import argparse
import logging
import os
import os.path as osp
import pprint
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml

from u2pl.dataset.builder import get_loader
from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.dist_helper import setup_distributed
from u2pl.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_rce_loss,
    get_criterion,
)
from u2pl.utils.lr_helper import get_optimizer, get_scheduler
from u2pl.utils.utils import (
    AverageMeter,
    cal_category_confidence,
    dynamic_copy_paste,
    generate_cutmix_mask,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    load_trained_model,
    sample_from_bank,
    set_random_seed,
    synchronize,
    update_cutmix_bank,
)

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--port", type=int, default=10682)
parser.add_argument("--seed", type=int, default=0)
logger = init_log("global", logging.INFO)
logger.propagate = 0


def main():
    global args, cfg, prototype
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    cudnn.enabled = True
    cudnn.benchmark = True

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
    if args.seed is not None:
        if rank == 0:
            print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:
        os.makedirs(cfg["saver"]["snapshot_dir"])

    # Create network.
    model = ModelBuilder(cfg["net"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    modules_back = [model.encoder]
    modules_head = [model.auxor, model.decoder]
    model.cuda()

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_teacher)
    model_teacher.cuda()
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    for p in model_teacher.parameters():
        p.requires_grad = False

    criterion = get_criterion(cfg)
    cons = cfg["criterion"].get("cons", False)
    sample = False
    if cons:
        sample = cfg["criterion"]["cons"].get("sample", False)
    if cons:
        criterion_cons = get_criterion(cfg, cons=True)
    else:
        criterion_cons = torch.nn.CrossEntropyLoss(ignore_index=255)

    trainloader_sup, trainloader_unsup, valloader = get_loader(cfg, seed=seed)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * 10)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    acp = cfg["dataset"].get("acp", False)
    acm = cfg["dataset"]["train"].get("acm", False)

    if acp or acm or sample:
        class_criterion = (
            torch.rand(3, cfg["net"]["num_classes"]).type(torch.float32).cuda()
        )
    if acm:
        cutmix_bank = torch.zeros(
            cfg["net"]["num_classes"], trainloader_unsup.dataset.__len__()
        ).cuda()

    # build class-wise memory bank
    memobank = []
    queue_ptrlis = []
    queue_size = []
    for i in range(cfg["net"]["num_classes"]):
        memobank.append([torch.zeros(0, 256)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # build prototype
    prototype = torch.zeros(
        (
            cfg["net"]["num_classes"],
            cfg["trainer"]["contrastive"]["num_queries"],
            1,
            256,
        )
    ).cuda()

    # Start to train model
    best_prec = 0
    labeled_epoch = 0

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["saver"]["snapshot_dir"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            if rank == 0:
                print("No checkpoint found in '{}'".format(lastest_model))
        else:
            if rank == 0:
                print(f"Resume model from: '{lastest_model}'")

            best_prec, labeled_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

            def map_func(storage, location):
                return storage.cuda()

            checkpoint = torch.load(lastest_model, map_location=map_func)
            class_criterion = checkpoint["class_criterion"].cuda()
            cutmix_bank = checkpoint["cutmix_bank"].cuda()

    elif cfg["saver"].get("pretrain", False):
        laod_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(trainloader_unsup), optimizer_start, start_epoch=labeled_epoch
    )

    for epoch in range(labeled_epoch, cfg_trainer["epochs"]):
        # Training
        t_start = time.time()
        if not acp and not acm and not sample:
            labeled_epoch = train(
                model,
                optimizer,
                lr_scheduler,
                criterion,
                trainloader_sup,
                epoch,
                labeled_epoch,
                model_teacher,
                trainloader_unsup,
                criterion_cons,
                memobank=memobank,
                queue_ptrlis=queue_ptrlis,
                queue_size=queue_size,
            )
        elif acm:
            labeled_epoch, class_criterion, cutmix_bank = train(
                model,
                optimizer,
                lr_scheduler,
                criterion,
                trainloader_sup,
                epoch,
                labeled_epoch,
                model_teacher,
                trainloader_unsup,
                criterion_cons,
                class_criterion,
                cutmix_bank,
                memobank=memobank,
                queue_ptrlis=queue_ptrlis,
                queue_size=queue_size,
            )
        else:
            labeled_epoch, class_criterion = train(
                model,
                optimizer,
                lr_scheduler,
                criterion,
                trainloader_sup,
                epoch,
                labeled_epoch,
                model_teacher,
                trainloader_unsup,
                criterion_cons,
                class_criterion,
                memobank=memobank,
                queue_ptrlis=queue_ptrlis,
                queue_size=queue_size,
            )
        # Validation
        if cfg_trainer["eval_on"]:
            prec = validate(model_teacher, model, valloader, epoch)
            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "teacher_state": model_teacher.state_dict(),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_miou": best_prec,
                    "class_criterion": class_criterion.cpu(),
                    "cutmix_bank": cutmix_bank.cpu(),
                }

                if prec > best_prec:
                    best_prec = prec
                    state["best_miou"] = prec
                    torch.save(
                        state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                    )

                torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt.pth"))

                logger.info(
                    " * Currently, the best val result is: {:.2f}".format(
                        best_prec * 100
                    )
                )

        t_end = time.time()
        if rank == 0:
            print("time for one epoch", t_end - t_start)


def train(
    model,
    optimizer,
    lr_scheduler,
    criterion,
    data_loader,
    epoch,
    labeled_epoch,
    model_teacher,
    data_loader_unsup,
    criterion_cons,
    class_criterion=None,
    cutmix_bank=None,
    memobank=None,
    queue_ptrlis=None,
    queue_size=None,
):
    global prototype
    model.train()
    model_teacher.train()

    data_loader.sampler.set_epoch(labeled_epoch)
    data_loader_unsup.sampler.set_epoch(epoch)
    data_loader_iter = iter(data_loader)
    data_loader_unsup_iter = iter(data_loader_unsup)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    ema_decay_origin = cfg["net"]["ema_decay"]
    consist_weight = cfg["criterion"].get("consist_weight", 1)
    contra_weight = cfg["criterion"].get("contra_weight", 1)
    threshold = cfg["criterion"].get("threshold", 0)
    cutmix = cfg["dataset"]["train"].get("cutmix", False)
    acm = cfg["dataset"]["train"].get("acm", False)
    acp = cfg["dataset"].get("acp", False)
    percent = cfg["trainer"]["contrastive"]["low_entropy_threshold"] * (
        1 - epoch / cfg["trainer"]["epochs"]
    )
    sample = False
    num_cat = 3
    if cfg["criterion"].get("cons", False):
        sample = cfg["criterion"]["cons"].get("sample", False)
    if sample:
        class_momentum = cfg["criterion"]["cons"].get("momentum", 0.999)
    if acp:
        all_cat = [i for i in range(num_classes)]
        ignore_cat = [0, 1, 2, 8, 10]
        target_cat = list(set(all_cat) - set(ignore_cat))
        class_momentum = cfg["dataset"]["acp"].get("momentum", 0.999)
        num_cat = cfg["dataset"]["acp"].get("number", 3)
    if acm:
        class_momentum = cfg["dataset"]["train"]["acm"].get("momentum", 0.999)
        area_thresh = cfg["dataset"]["train"]["acm"].get("area_thresh", 0.0001)
        no_pad = cfg["dataset"]["train"]["acm"].get("no_pad", False)
        no_slim = cfg["dataset"]["train"]["acm"].get("no_slim", False)
        if "area_thresh2" in cfg["dataset"]["train"]["acm"].keys():
            area_thresh2 = cfg["dataset"]["train"]["acm"]["area_thresh2"]
        else:
            area_thresh2 = area_thresh
    rank, world_size = get_rank(), get_world_size()
    if acp or acm or sample:
        conf = 1 - class_criterion[0]
        conf = conf[target_cat]
        conf = (conf ** 0.5).cpu().numpy()
        conf_print = np.exp(conf) / np.sum(np.exp(conf))
        if rank == 0:
            print("epoch [", epoch, ": ]", "sample_rate_target_class_conf", conf_print)
            print("epoch [", epoch, ": ]", "criterion_per_class", class_criterion[0])
            print(
                "epoch [",
                epoch,
                ": ]",
                "sample_rate_per_class_conf",
                (1 - class_criterion[0]) / (torch.max(1 - class_criterion[0]) + 1e-12),
            )

    sup_losses = AverageMeter(10)
    unsup_losses = AverageMeter(10)
    contra_losses = AverageMeter(10)
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    # for step, batch in enumerate(data_loader_unsup):
    for step in range(len(data_loader_unsup)):
        i_iter = epoch * len(data_loader_unsup) + step
        lr = lr_scheduler.get_lr()
        lr_scheduler.step()
        if acp or acm:
            conf = 1 - class_criterion[0]
            conf = conf[target_cat]
            conf = (conf ** 0.5).cpu().numpy()
            conf = np.exp(conf) / np.sum(np.exp(conf))
            query_cat = []
            for rc_idx in range(num_cat):
                query_cat.append(np.random.choice(target_cat, p=conf))
            query_cat = list(set(query_cat))
        # get labeled input
        if acp:
            try:
                labeled_inputs = data_loader_iter.next()
            except:
                labeled_epoch += 1
                data_loader.sampler.set_epoch(labeled_epoch)
                data_loader_iter = iter(data_loader)
                labeled_inputs = data_loader_iter.next()
            if len(labeled_inputs) > 2:
                images_sup, labels_sup, paste_img, paste_label = labeled_inputs
                images_sup = images_sup.cuda()
                labels_sup = labels_sup.long().cuda()
                paste_img = paste_img.cuda()
                paste_label = paste_label.long().cuda()
                images_sup, labels_sup = dynamic_copy_paste(
                    images_sup, labels_sup, paste_img, paste_label, query_cat
                )
                del paste_img, paste_label
            else:
                images_sup, labels_sup = labeled_inputs
                images_sup = images_sup.cuda()
                labels_sup = labels_sup.long().cuda()
                images_sup, labels_sup = dynamic_copy_paste(
                    images_sup, labels_sup, query_cat
                )
        else:
            try:
                images_sup, labels_sup = data_loader_iter.next()
            except:
                labeled_epoch += 1
                data_loader.sampler.set_epoch(labeled_epoch)
                data_loader_iter = iter(data_loader)
                images_sup, labels_sup = data_loader_iter.next()
            images_sup = images_sup.cuda()
            labels_sup = labels_sup.long().cuda()
        # get unlabeled input
        if not cutmix and not acm:
            (
                images_unsup_weak,
                _,
                images_unsup_strong,
                _,
                valid_mask,
            ) = data_loader_unsup_iter.next()
            images_unsup_weak = images_unsup_weak.cuda()
            images_unsup_strong = images_unsup_strong.cuda()
            valid_mask = valid_mask.long().cuda()
        elif acm:
            image_unsup, _, img_id = data_loader_unsup_iter.next()
            prob_im = random.random()
            if image_unsup.shape[0] > 1:
                if prob_im > 0.5:
                    image_unsup = image_unsup[0]
                    img_id = img_id[0]
                else:
                    image_unsup = image_unsup[1]
                    img_id = img_id[1]
            image_unsup = image_unsup.cuda()
            sample_id, sample_cat = sample_from_bank(cutmix_bank, class_criterion[0])
            image_unsup2, _, _ = data_loader_unsup.dataset.__getitem__(index=sample_id)
            image_unsup2 = image_unsup2.cuda()
            images_unsup = torch.cat(
                [image_unsup.unsqueeze(0), image_unsup2.unsqueeze(0)], dim=0
            )
            images_unsup_weak = images_unsup.clone()
        else:
            # cutmix for unlabeled input
            images_unsup, _, valid_masks = data_loader_unsup_iter.next()
            images_unsup = images_unsup.cuda()
            valid_masks = valid_masks.long().cuda()
            images_unsup_weak = images_unsup.clone()
            # construct strong and weak inputs for teacher and student model
            assert valid_masks.shape[0] == 2
            # images_unsup 2(B),3,H,W
            prob = random.random()
            if prob > 0.5:
                valid_mask_mix = valid_masks[0]  # H, W
                images_unsup_strong = images_unsup[0] * valid_mask_mix + images_unsup[
                    1
                ] * (1 - valid_mask_mix)
                images_unsup_strong = images_unsup_strong.unsqueeze(0)
            else:
                valid_mask_mix = valid_masks[1]
                images_unsup_strong = images_unsup[1] * valid_mask_mix + images_unsup[
                    0
                ] * (1 - valid_mask_mix)
                images_unsup_strong = images_unsup_strong.unsqueeze(0)

        # student model forward
        batch_size, c, h, w = images_sup.size()
        outs = model(images_sup)
        reps_student_sup = outs["rep"]
        batch_size, c, h_small, w_small = outs["pred"].size()

        preds_student_sup = [
            F.interpolate(outs["pred"], (h, w), mode="bilinear", align_corners=True),
            F.interpolate(outs["aux"], (h, w), mode="bilinear", align_corners=True),
        ]
        loss_sup_student = criterion(preds_student_sup, labels_sup)
        if cfg["trainer"].get("sym_ce_l", False):
            loss_sup_student += 0.1 * compute_rce_loss(preds_student_sup[0], labels_sup)
        loss_sup_student /= world_size

        # teacher model forward
        with torch.no_grad():
            outs = model_teacher(images_sup)
            reps_teacher_sup = outs["rep"].detach()
            preds_teacher_sup = outs["pred"].detach()
            preds_teacher_sup = F.interpolate(
                preds_teacher_sup, (h, w), mode="bilinear", align_corners=True
            )

            model_teacher.eval()
            outs = model_teacher(images_unsup_weak)
            preds_teacher_unsup = outs["pred"].detach()
            preds_teacher_unsup = F.interpolate(
                preds_teacher_unsup, (h, w), mode="bilinear", align_corners=True
            )

            if cutmix:
                if prob > 0.5:
                    preds_teacher_unsup = preds_teacher_unsup[
                        0
                    ] * valid_mask_mix + preds_teacher_unsup[1] * (1 - valid_mask_mix)
                else:
                    preds_teacher_unsup = preds_teacher_unsup[
                        1
                    ] * valid_mask_mix + preds_teacher_unsup[0] * (1 - valid_mask_mix)
                preds_teacher_unsup = preds_teacher_unsup.unsqueeze(0)
            if acm:
                valid_mask_mix = generate_cutmix_mask(
                    preds_teacher_unsup[1].max(0)[1].cpu().numpy(),
                    sample_cat,
                    area_thresh,
                    no_pad=no_pad,
                    no_slim=no_slim,
                )
                images_unsup_strong = (
                    images_unsup[0] * (1 - valid_mask_mix)
                    + images_unsup[1] * valid_mask_mix
                )
                # update cutmix bank
                cutmix_bank = update_cutmix_bank(
                    cutmix_bank, preds_teacher_unsup, img_id, sample_id, area_thresh2
                )
                preds_teacher_unsup = (
                    preds_teacher_unsup[0] * (1 - valid_mask_mix)
                    + preds_teacher_unsup[1] * valid_mask_mix
                )

                preds_teacher_unsup = preds_teacher_unsup.unsqueeze(0)
                images_unsup_strong = images_unsup_strong.unsqueeze(0)

            # compute consistency loss
            logits_teacher_sup = preds_teacher_sup.max(1)[1]
            conf_sup = F.softmax(preds_teacher_sup, dim=1).max(1)[0]
            conf_teacher_sup_map = conf_sup
            logits_teacher_sup[conf_teacher_sup_map < threshold] = 255

            probs_teacher_unsup = F.softmax(preds_teacher_unsup, dim=1)
            entropy_teacher_unsup = -torch.sum(
                probs_teacher_unsup * torch.log(probs_teacher_unsup + 1e-10), dim=1
            )
            thresh = np.percentile(
                entropy_teacher_unsup.detach().cpu().numpy().flatten(), percent
            )
            conf_unsup = F.softmax(preds_teacher_unsup, dim=1).max(1)[0]
            logits_teacher_unsup = preds_teacher_unsup.max(1)[1]
            if not cutmix and not acm:
                logits_teacher_unsup += valid_mask
                logits_teacher_unsup[logits_teacher_unsup > 20] = 255

            logits_teacher_unsup[entropy_teacher_unsup < thresh] = 255

            model_teacher.train()
            reps_teacher_unsup = model_teacher(images_unsup_strong)["rep"].detach()
            prob_l_teacher = F.softmax(
                F.interpolate(
                    preds_teacher_sup,
                    (h_small, w_small),
                    mode="bilinear",
                    align_corners=True,
                ),
                dim=1,
            ).detach()
            prob_u_teacher = F.softmax(
                F.interpolate(
                    preds_teacher_unsup,
                    (h_small, w_small),
                    mode="bilinear",
                    align_corners=True,
                ),
                dim=1,
            ).detach()

        outs = model(images_unsup_strong)
        reps_student_unsup = outs["rep"]
        preds_student_unsup = [
            F.interpolate(outs["pred"], (h, w), mode="bilinear", align_corners=True),
            F.interpolate(outs["aux"], (h, w), mode="bilinear", align_corners=True),
        ]

        # consistency loss
        with torch.no_grad():
            if acp or acm or sample:
                category_entropy = cal_category_confidence(
                    preds_student_sup[0].detach(),
                    preds_student_unsup[0].detach(),
                    labels_sup,
                    preds_teacher_unsup,
                    num_classes,
                )
                # perform momentum update
                class_criterion = (
                    class_criterion * class_momentum
                    + category_entropy.cuda() * (1 - class_momentum)
                )
        if isinstance(criterion_cons, torch.nn.CrossEntropyLoss):
            loss_consistency1 = (
                criterion_cons(preds_student_sup[0], logits_teacher_sup) / world_size
            )
            loss_consistency2 = (
                criterion_cons(preds_student_unsup[0], logits_teacher_unsup)
                / world_size
            )

        elif sample:
            loss_consistency1 = (
                criterion_cons(
                    preds_student_sup[0],
                    conf_sup,
                    logits_teacher_sup,
                    class_criterion[0],
                )
                / world_size
            )
            loss_consistency2 = (
                criterion_cons(
                    preds_student_unsup[0],
                    conf_unsup,
                    logits_teacher_unsup,
                    class_criterion[0],
                )
                / world_size
            )

        else:
            loss_consistency1 = (
                criterion_cons(preds_student_sup[0], conf_sup, logits_teacher_sup)
                / world_size
            )
            loss_consistency2 = (
                criterion_cons(preds_student_unsup[0], conf_unsup, logits_teacher_unsup)
                / world_size
            )

        if cfg["trainer"].get("sym_ce_u", False):
            loss_consistency1 += (
                0.1
                * compute_rce_loss(preds_student_sup[0], logits_teacher_sup)
                / world_size
            )
            loss_consistency2 += (
                0.1
                * compute_rce_loss(preds_student_unsup[0], logits_teacher_unsup)
                / world_size
            )

        loss_consistency = loss_consistency1 + loss_consistency2

        # contrastive loss (U2PL)
        contra_flag = "none"
        if epoch >= cfg["trainer"]["contrastive"].get("start_epoch", 20):
            cfg_contra = cfg["trainer"]["contrastive"]
            contra_flag = "{}:{}".format(
                cfg_contra["low_rank"], cfg_contra["high_rank"]
            )

            with torch.no_grad():
                entropy = entropy_teacher_unsup

                low_thresh = np.percentile(
                    entropy.detach().cpu().numpy().flatten(),
                    cfg_contra["low_entropy_threshold"],
                )
                low_entropy_mask = (
                    entropy.le(low_thresh).float()
                    * (logits_teacher_unsup != 255).bool()
                )

                high_thresh = np.percentile(
                    entropy.detach().cpu().numpy().flatten(),
                    cfg_contra["unsupervised_entropy_ignore"],
                )
                high_entropy_mask = (
                    entropy.ge(high_thresh).float()
                    * (logits_teacher_unsup != 255).bool()
                )

                low_mask_all = torch.cat(
                    (
                        (labels_sup.unsqueeze(1) != 255).float(),
                        low_entropy_mask.unsqueeze(1),
                    )
                )
                low_mask_all = F.interpolate(
                    low_mask_all, size=(h_small, w_small), mode="nearest"
                )

                if cfg_contra.get("negative_high_entropy", True):
                    contra_flag += " high"
                    high_mask_all = torch.cat(
                        (
                            (labels_sup.unsqueeze(1) != 255).float(),
                            high_entropy_mask.unsqueeze(1),
                        )
                    )
                else:
                    contra_flag += " low"
                    high_mask_all = torch.cat(
                        (
                            (labels_sup.unsqueeze(1) != 255).float(),
                            torch.ones(logits_teacher_unsup.shape)
                            .float()
                            .unsqueeze(1)
                            .cuda(),
                        ),
                    )
                high_mask_all = F.interpolate(
                    high_mask_all, size=(h_small, w_small), mode="nearest"
                )  # down sample

                # down sample and concat
                label_l_small = F.interpolate(
                    label_onehot(labels_sup, num_classes),
                    size=(h_small, w_small),
                    mode="nearest",
                )
                label_u_small = F.interpolate(
                    label_onehot(logits_teacher_unsup, num_classes),
                    size=(h_small, w_small),
                    mode="nearest",
                )

            if not cfg_contra.get("anchor_ema", False):
                new_keys, contra_loss = compute_contra_memobank_loss(
                    torch.cat((reps_student_sup, reps_student_unsup)),
                    label_l_small.long(),
                    label_u_small.long(),
                    prob_l_teacher.detach(),
                    prob_u_teacher.detach(),
                    low_mask_all,
                    high_mask_all,
                    cfg_contra,
                    memobank,
                    queue_ptrlis,
                    queue_size,
                    torch.cat((reps_teacher_sup, reps_teacher_unsup)).detach(),
                    conf_weight=cfg_contra.get("conf_weight", False),
                )
            else:
                prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                    torch.cat((reps_student_sup, reps_student_unsup)),
                    label_l_small.long(),
                    label_u_small.long(),
                    prob_l_teacher.detach(),
                    prob_u_teacher.detach(),
                    low_mask_all,
                    high_mask_all,
                    cfg_contra,
                    memobank,
                    queue_ptrlis,
                    queue_size,
                    torch.cat((reps_teacher_sup, reps_teacher_unsup)).detach(),
                    prototype,
                    i_iter
                    - len(data_loader_unsup)
                    * cfg["trainer"]["contrastive"].get("start_epoch", 20)
                    + 1,
                )

            dist.all_reduce(contra_loss)
            contra_loss = contra_loss / world_size / world_size
        else:
            contra_loss = 0 * reps_student_sup.sum()

        # gather all loss from different gpus
        reduced_loss = loss_sup_student.clone().detach()
        dist.all_reduce(reduced_loss)
        sup_losses.update(reduced_loss.item())

        reduced_loss = loss_consistency.clone().detach()
        dist.all_reduce(reduced_loss)
        unsup_losses.update(reduced_loss.item())

        reduced_loss = contra_loss.clone().detach()
        dist.all_reduce(reduced_loss)
        contra_losses.update(reduced_loss.item())

        loss = (
            loss_sup_student
            + consist_weight * loss_consistency
            + contra_weight * contra_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get the output produced by model
        output = (
            preds_student_sup[0]
            if cfg["net"].get("aux_loss", False)
            else preds_student_sup
        )
        output = output.data.max(1)[1].cpu().numpy()
        target = labels_sup.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())

        # update teacher model with EMA
        ema_decay = min(1 - 1 / (i_iter + 1), ema_decay_origin)
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.data = ema_decay * t_params.data + (1 - ema_decay) * s_params.data

        if i_iter % 10 == 0 and rank == 0:
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            logger.info(
                "[{}] iter [{}/{}]\tLR {:.5f}\tSup {:.4f}\tUnsup {:.4f}\t Contra {:.4f}\tmIoU {:.2f}".format(
                    contra_flag,
                    i_iter,
                    cfg["trainer"]["epochs"] * len(data_loader_unsup),
                    lr[0],
                    sup_losses.avg,
                    unsup_losses.avg,
                    contra_losses.avg,
                    mIoU * 100,
                )
            )

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    if rank == 0:
        logger.info(" * Epoch [{}]\tTrain mIoU {:.2f}".format(epoch, mIoU * 100))
    if class_criterion is not None and cutmix_bank is None:
        return labeled_epoch, class_criterion
    elif cutmix_bank is not None:
        return labeled_epoch, class_criterion, cutmix_bank
    else:
        return labeled_epoch


def validate(model_teacher, model_student, data_loader, epoch):
    model_teacher.eval()
    model_student.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = get_rank(), get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # meters for student
    intersection_meter_student = AverageMeter()
    union_meter_student = AverageMeter()
    target_meter_student = AverageMeter()

    for step, batch in enumerate(data_loader):
        batch_start = time.time()

        images, labels = batch
        b, c, h, w = images.size()
        images = images.cuda()
        labels = labels.long().cuda()
        with torch.no_grad():
            preds = model_teacher(images)
            preds_student = model_student(images)

        # get the output produced by model_teacher
        output = F.interpolate(
            preds["pred"], (h, w), mode="bilinear", align_corners=True
        )
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())

        # get the output produced by model_student
        output_student = F.interpolate(
            preds_student["pred"], (h, w), mode="bilinear", align_corners=True
        )
        output_student = output_student.data.max(1)[1].cpu().numpy()
        intersection_s, union_s, target_s = intersectionAndUnion(
            output_student, target_origin, num_classes, ignore_label
        )
        reduced_intersection_s = torch.from_numpy(intersection_s).cuda()
        reduced_union_s = torch.from_numpy(union_s).cuda()
        reduced_target_s = torch.from_numpy(target_s).cuda()

        dist.all_reduce(reduced_intersection_s)
        dist.all_reduce(reduced_union_s)
        dist.all_reduce(reduced_target_s)
        intersection_meter_student.update(reduced_intersection_s.cpu().numpy())
        union_meter_student.update(reduced_union_s.cpu().numpy())
        target_meter_student.update(reduced_target_s.cpu().numpy())

        if step % 10 == 0 and rank == 0:
            logger.info(
                "Test [{}/{}]\tTime {:.3f}".format(
                    step,
                    len(data_loader),
                    time.time() - batch_start,
                )
            )

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    iou_class_student = intersection_meter_student.sum / (
        union_meter_student.sum + 1e-10
    )
    accuracy_class_student = intersection_meter_student.sum / (
        target_meter_student.sum + 1e-10
    )
    mIoU_student = np.mean(iou_class_student)

    if rank == 0:
        for i, IoU in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, IoU * 100))

        logger.info(
            " * Epoch [{}], Val_Teacher mIoU = {:.2f}".format(epoch, mIoU * 100)
        )
        logger.info(
            " * Epoch [{}], Val_Student mIoU = {:.2f}".format(epoch, mIoU_student * 100)
        )

    return mIoU


if __name__ == "__main__":
    main()
