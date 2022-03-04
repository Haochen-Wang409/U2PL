import copy
import os
import os.path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from . import augmentation as psp_trsform
from .base import BaseDataset


class voc_dset(BaseDataset):
    def __init__(
        self,
        data_root,
        data_list,
        trs_form,
        seed=0,
        n_sup=10582,
        split="val",
        unsup=False,
        fm=False,
        acp=False,
        paste_trs=None,
        prob=0.5,
        acm=False,
    ):
        super(voc_dset, self).__init__(data_list)
        self.data_root = data_root
        self.transform = trs_form
        self.paste_trs = paste_trs
        self.fm = fm
        self.acp = acp and split == "train"
        self.prob = prob
        self.acm = acm
        random.seed(seed)
        if (len(self.list_sample) >= n_sup) and split == "train":
            self.list_sample_new = random.sample(self.list_sample, n_sup)
            if unsup:
                # transform to tuple
                for i in range(len(self.list_sample)):
                    self.list_sample[i] = tuple(self.list_sample[i])
                for i in range(len(self.list_sample_new)):
                    self.list_sample_new[i] = tuple(self.list_sample_new[i])
                self.list_sample_unsup = list(
                    set(self.list_sample) - set(self.list_sample_new)
                )
                self.list_sample_new = self.list_sample_unsup
        else:
            self.list_sample_new = self.list_sample

    def __getitem__(self, index):
        # load VOC image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
        image = self.img_loader(image_path, "RGB")
        label = self.img_loader(label_path, "P")
        # loader paste img and mask
        if self.acp:
            if random.random() > self.prob:
                paste_idx = random.randint(0, self.__len__() - 1)
                paste_img_path = os.path.join(
                    self.data_root, self.list_sample_new[paste_idx][0]
                )
                paste_img = self.img_loader(paste_img_path, "RGB")
                paste_label_path = os.path.join(
                    self.data_root, self.list_sample_new[paste_idx][1]
                )
                paste_label = self.img_loader(paste_label_path, "L")
                paste_img, paste_label = self.paste_trs(paste_img, paste_label)
            else:
                paste_img, paste_label, instance_label = None, None, None

        if self.fm:
            inputs = self.transform(image, label)
            if len(inputs) == 5:
                image_weak, label_weak, image_strong, label_strong, valid = inputs
                return (
                    image_weak[0],
                    label_weak[0, 0].long(),
                    image_strong[0],
                    label_strong[0, 0].long(),
                    valid[0, 0].long(),
                )
            else:
                image, label, valid = inputs
                return image[0], label[0, 0].long(), valid[0, 0].long()
        elif self.acm:
            image, label = self.transform(image, label)
            return image[0], label[0, 0].long(), index
        else:
            image, label = self.transform(image, label)

        if self.acp:
            if paste_img is not None:
                return torch.cat((image[0], paste_img[0]), dim=0), torch.cat(
                    [label[0, 0].long(), paste_label[0, 0].long()], dim=0
                )
            else:
                h, w = image[0].shape[1], image[0].shape[2]
                paste_img = torch.zeros(3, h, w)
                paste_label = torch.zeros(h, w)
                return torch.cat((image[0], paste_img), dim=0), torch.cat(
                    [label[0, 0].long(), paste_label.long()], dim=0
                )

        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)


def build_transfrom(cfg, fm=False, acp=False):
    trs_form = []
    mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        if not acp:
            trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
        else:
            trs_form.append(psp_trsform.RandResize(cfg["acp"]["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
    if fm and cfg.get("cutout", False):
        n_holes, length = cfg["cutout"]["n_holes"], cfg["cutout"]["length"]
        trs_form.append(psp_trsform.Cutout(n_holes=n_holes, length=length))
    if fm and cfg.get("cutmix", False):
        n_holes, prop_range = cfg["cutmix"]["n_holes"], cfg["cutmix"]["prop_range"]
        trs_form.append(psp_trsform.Cutmix(prop_range=prop_range, n_holes=n_holes))
    return psp_trsform.Compose(trs_form)


def build_voc_semi_loader_cp(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]
    cfg_trainer = all_cfg["trainer"]
    fm = (
        True
        if "cutout" in cfg_dset["train"].keys() or "cutmix" in cfg_dset["train"].keys()
        else False
    )
    acp = True if "acp" in cfg_dset.keys() else False
    acm = cfg_dset["train"].get("acm", False)
    # print('fm',fm)
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 10582)
    prob = cfg["acp"].get("prob", 0.5)
    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg, fm=fm)
    if acp:
        paste_trs = build_transfrom(cfg, acp=True)
    else:
        paste_trs = None
    dset = voc_dset(
        cfg["data_root"],
        cfg["data_list"],
        trs_form,
        seed,
        n_sup,
        split,
        acp=acp,
        paste_trs=paste_trs,
        prob=prob,
    )

    # build sampler
    sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
    )

    # build sampler for unlabeled set
    dset_unsup = voc_dset(
        cfg["data_root"],
        cfg["data_list"],
        trs_form_unsup,
        seed,
        n_sup,
        split,
        unsup=True,
        fm=fm,
        acm=acm,
    )
    if split == "train":
        sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_unsup,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
        )
        return loader, loader_unsup
    return loader
