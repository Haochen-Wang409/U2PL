import importlib

import torch.nn as nn
from torch.nn import functional as F
from .decoder import Aux_Module
from u2pl.models import swin


class ModelBuilder(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder, self).__init__()
        self._sync_bn = net_cfg["sync_bn"]
        self._num_classes = net_cfg["num_classes"]

        self.encoder = self._build_encoder(net_cfg["encoder"])
        self.decoder = self._build_decoder(net_cfg["decoder"])

        self._use_auxloss = True if net_cfg.get("aux_loss", False) else False
        self.fpn = True if net_cfg["encoder"]["kwargs"].get("fpn", False) else False
        if self._use_auxloss:
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(
                cfg_aux["aux_plane"], self._num_classes, self._sync_bn
            )

    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update(
            {
                "in_planes": self.encoder.get_outplanes(),
                "sync_bn": self._sync_bn,
                "num_classes": self._num_classes,
            }
        )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x):
        if self._use_auxloss:
            if self.fpn:
                # feat1 used as dsn loss as default, f1 is layer2's output as default
                f1, f2, feat1, feat2 = self.encoder(x)
                outs = self.decoder([f1, f2, feat1, feat2])
            else:
                feat1, feat2 = self.encoder(x)
                outs = self.decoder(feat2)

            pred_aux = self.auxor(feat1)

            outs.update({"aux": pred_aux})
            return outs
        else:
            feat = self.encoder(x)
            outs = self.decoder(feat)
            return outs


class ModelBuilder_Sunyong(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder_Sunyong, self).__init__()
        self._sync_bn = net_cfg["sync_bn"]
        self._num_classes = net_cfg["num_classes"]

        self.encoder = swin.SwinTransformer(in_chans=3,
                                            embed_dim=96,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=7,
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0.,
                                            attn_drop_rate=0.,
                                            drop_path_rate=0.3,
                                            ape=False,
                                            patch_norm=True,
                                            out_indices=(0, 1, 2, 3),
                                            use_checkpoint=False)
        self.decoder = swin.M_UPerHead(in_channels=[96, 192, 384, 768],
                                       in_index=[0, 1, 2, 3],
                                       pool_scales=(1, 2, 3, 6),
                                       channels=512,
                                       dropout_ratio=0.1,
                                       num_classes=2,
                                       align_corners=False,)

        self._use_auxloss = True if net_cfg.get("aux_loss", False) else False
        self.fpn = True if net_cfg["encoder"]["kwargs"].get("fpn", False) else False
        if self._use_auxloss:
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(
                cfg_aux["aux_plane"], self._num_classes, self._sync_bn
            )

    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update(
            {
                "in_planes": self.encoder.get_outplanes(),
                "sync_bn": self._sync_bn,
                "num_classes": self._num_classes,
            }
        )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x):
        x_size = x.shape[2:]
        if self._use_auxloss:
            if self.fpn:
                feat = self.encoder(x)  # list of feature pyramid
                feat_out = self.decoder(feat)
                outs = swin.Upsample(feat_out, x_size)
            else:
                feat1, feat2 = self.encoder(x)
                outs = self.decoder(feat2)

            pred_aux = self.auxor(feat[2])

            outs.update({"aux": pred_aux})
            return outs
        else:
            feat = self.encoder(x)  # list of feature pyramid
            feat_out = self.decoder(feat)
            outs = swin.Upsample(feat_out, x_size)
            return outs
