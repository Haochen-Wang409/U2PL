import logging

from .cityscapes import build_cityloader
from .cityscapes_semi_cp import build_city_semi_loader_cp
from .pascal_voc import build_vocloader
from .pascal_voc_semi_cp import build_voc_semi_loader_cp

logger = logging.getLogger("global")


def get_loader(cfg, seed=0):
    cfg_dataset = cfg["dataset"]

    if cfg_dataset["type"] == "cityscapes_semi_cp":
        trainloader_sup, trainloader_unsup = build_city_semi_loader_cp(
            "train", cfg, seed=seed
        )
        valloader = build_cityloader("val", cfg)
    elif cfg_dataset["type"] == "pascal_voc_semi_cp":
        trainloader_sup, trainloader_unsup = build_voc_semi_loader_cp(
            "train", cfg, seed=seed
        )
        valloader = build_vocloader("val", cfg)
    else:
        raise NotImplementedError(
            "dataset type {} is not supported".format(cfg_dataset)
        )
    logger.info("Get loader Done...")

    return trainloader_sup, trainloader_unsup, valloader
