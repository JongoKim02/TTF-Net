# BSD 3-Clause License
# Copyright (c) 2025, System Level Design Group
"""
Created on Sat June  21 11:06:19 2025

@author: mostafij.rahman
python main_finetune_BTCV_TU.py --root /research/data/amos_trns/ --output output_folder/run1 --dataset BTCV13 --img_size 96 96 96 --n_channels 1 --network 3DUXNET_EffiDec3D --channels 48 96 192 384 --n_decoder_channels 48 --ds False --mode train --pretrain False --batch_size 1 --crop_sample 4 --lr 0.001 --optim AdamW --max_iter 45000  --eval_step 250 --val_batch 1 --gpu 0 --cache_rate 1.0 --num_workers 4 --overlap 0.7 > output_folder/BTCV13_3DUXNET_EffiDec3D_loss_dsFalse_1out_96x96x96_lr1e3_itr45000_overlap070_run1.txt

"""

import os, argparse

# --- EARLY PARSE: get --gpu before importing torch/monai ---
_early = argparse.ArgumentParser(add_help=False)
_early.add_argument('--gpu', type=str, default=None)
_early_args, _ = _early.parse_known_args()
if _early_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_args.gpu
# ------------------------------------------------------------

# from torch.cuda.amp import autocast, GradScaler

import torch


try:

    from torch.amp import autocast as _autocast_new, GradScaler as _GradScaler_new
    def autocast_ctx(enabled: bool):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return _autocast_new(dev, enabled=enabled)
    def make_scaler():
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return _GradScaler_new(dev)
except Exception:

    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScaler_old
    def autocast_ctx(enabled: bool):
        return _autocast_old(enabled=enabled)
    def make_scaler():
        return _GradScaler_old()
# ------------------------------------------------------------

_projector_checked = False

from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.networks.nets import UNETR, SwinUNETR
from networks.swin_unetr_effidec3d import SwinUNETR as SwinUNETRv2
from networks.MedNeXt.mednextv1.create_mednext_v1 import create_mednext_v1
from networks.UXNet_3D.network_backbone import UXNET, UXNET_EffiDec3D
from networks.swin_unetr_effidec3d import SwinUNETR_EffiDec3D
from networks.MedNeXt.mednextv1.create_mednextv1_effidec3d import create_mednextv1_effidec3d

from networks.unetr_pp.synapse.unetr_pp_synapse import UNETR_PP
from networks.nnunet.network_architecture.generic_UNet import Generic_UNet
from networks.SegFormer3D.segformer3d import SegFormer3D
from networks.SlimUNETR.SlimUNETR import SlimUNETR
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceCELoss
from monai_utils.inferers.utils import sliding_window_inference_1out
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    Activations,
    )



import torch
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms import data_loader, data_transforms
from monai.networks.blocks import UnetOutBlock
import torch.nn as nn
import torch.nn.functional as F

from ptflops import get_model_complexity_info

import csv
import os
import numpy as np
import scipy.ndimage as ndimage
from medpy import metric
from tqdm import tqdm
import argparse

import resource

import re
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = argparse.ArgumentParser(description='3DUXNET w/ EffiDec3D hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='data', required=True, help='Path to the dataset root (expects imagesTr/labelsTr/imagesVal/labelsVal).')
parser.add_argument('--output', type=str, default='', required=True, help='Path to the output directory for logs, checkpoints, and metrics.')
parser.add_argument('--dataset', type=str, default='flare', required=True, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
parser.add_argument('--img_size', type=int, nargs='+', default=[96,96,96], help='3D ROI size, e.g., [96, 96, 96]') #500
parser.add_argument('--n_channels', type=int, default=1, help='number of channels in input image') #500

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='SwinUNETRv2', help='Network models: {3DUXNET_EffiDec3D, SwinUNETR_EffiDec3D, SwinUNETRv2_EffiDec3D, MedNeXt_M_EffiDec3D, TransBTS, nnFormer, UNETR, SlimUNETR, SwinUNETR, 3DUXNET, MedNeXt_M}')
parser.add_argument('--channels', type=int, nargs='+', default=[48, 96, 192, 384], help='Number of channels in the 3DUXNet network, e.g., [48, 96, 192, 384]') #500
parser.add_argument('--feature_size', type=int, default=48, help='Feature size for SwinTransformer') #500
parser.add_argument('--n_decoder_channels', type=str, default="48", help='Number of channels in each satge of the decoder') #500
parser.add_argument('--resolution_factor', type=int, default=2, help='Resolution factor to control high-resolution operations, e.g., [1,2,4,8,16]') #500
parser.add_argument('--skip_aggregation', type=str, default='addition', required=False, help='Aggregation in skip connection: {addition, concatenation}')

parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--pretrain', action='store_true', help='Have pretrained weights or not')

parser.add_argument('--ds', default=True, help='Use of deep supervision (ds) or not')
parser.add_argument('--pretrained_weights', default='', help='Path to pretrained model weights.')
parser.add_argument('--pretrain_classes', type=int, default=5,
                    help='Number of classes output from pretrained model')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=40000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=50, help='Per steps to perform validation') #500
parser.add_argument('--val_batch', type=int, default=1, help='Validation batch size') #500
parser.add_argument('--overlap', type=float, default=0.5, help='Amount of overlap between scans') #500
parser.add_argument('--overlap_mode', type=str, default='constant', help='overlap mode') #500

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='1', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

# --- Text-Contrast options ---
parser.add_argument(
    "--text_bank_npz",
    type=str,
    default="",
    help="Path to the training text memory bank .npz file (contains 'ids' and 'embeddings').",
)
parser.add_argument("--text_dim", type=int, default=1024)   # Qwen3-Emb-0.6B
parser.add_argument("--contrastive_w", type=float, default=1)
parser.add_argument("--contrastive_t", type=float, default=0.07)
parser.add_argument("--id_regex", type=str, default=r"train_(\d+)_0000\.nii\.gz")


parser.add_argument("--triplane_dim", type=int, default=64,
    help="Channel dim d for tri-plane Q.")
parser.add_argument(
    "--eval_text_bank_npz",
    type=str,
    default="",
    help="Path to the evaluation text memory bank .npz file. If empty, --text_bank_npz is used.",
)

parser.add_argument("--debug_text", action="store_true",
    help="")

args = parser.parse_args()
if isinstance(args.ds, str):
    args.ds = args.ds.lower() in ("1", "true", "t", "yes", "y")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

train_samples, valid_samples, out_classes = data_loader(args)

train_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
]

val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
]

print(train_files)
print(val_files)

set_determinism(seed=0)

train_transforms, val_transforms = data_transforms(args)

## Train Pytorch Data Loader and Caching
print('Start caching datasets!')
train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=args.cache_rate, num_workers=args.num_workers)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

## Valid Pytorch Data Loader and Caching
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)

val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

if args.n_decoder_channels == 'None':
    args.n_decoder_channels = None
else:
    args.n_decoder_channels = int(args.n_decoder_channels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load Networks
if device.type == "cuda":
    torch.cuda.set_device(0)

if args.network == '3DUXNET_EffiDec3D':
    model = UXNET_EffiDec3D(
        in_chans=args.n_channels,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=args.channels,
        n_decoder_channels=args.n_decoder_channels,
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
        skip_aggregation=args.skip_aggregation,
        resolution_factor=args.resolution_factor
    ).to(device)

    if args.pretrain and args.pretrained_weights:
        try:
            sd = torch.load(args.pretrained_weights, map_location='cpu', weights_only=True)
        except TypeError:
            sd = torch.load(args.pretrained_weights, map_location='cpu')

        if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}

        head_prefixes = ('out.', 'out.conv.', 'out.conv.conv.')
        for k in list(sd.keys()):
            if k.startswith(head_prefixes):
                sd.pop(k)


        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[pretrain] backbone loaded | missing={len(missing)} unexpected={len(unexpected)}")

        try:
            in_ch = model.out.conv.conv.weight.shape[1]  # (C_out, C_in, 1,1,1) => C_in
        except Exception:

            if getattr(args, 'n_decoder_channels', None) not in (None, 'None'):
                in_ch = int(args.n_decoder_channels)
            elif isinstance(args.channels, (list, tuple)) and len(args.channels) > 0:
                in_ch = int(args.channels[0])
            else:
                in_ch = 48

        from monai.networks.blocks import UnetOutBlock

        model.out = UnetOutBlock(
            spatial_dims=3,
            in_channels=int(in_ch),
            out_channels=out_classes
        ).to(device)

        print(f"[pretrain] head reset -> in_channels={int(in_ch)}, out_channels={out_classes}")
        print("[pretrain] Pretrained weights loaded and head replaced. Ready to train.")
    else:
        print("[pretrain] (skip) No pretrained weights requested.")





elif args.network == 'SwinUNETR_EffiDec3D':
	model = SwinUNETR_EffiDec3D(
        img_size=args.img_size,
        in_channels=args.n_channels, #1
        out_channels=out_classes,
        feature_size=args.feature_size,
        n_decoder_channels=args.n_decoder_channels,
        resolution_factor=args.resolution_factor,
        use_checkpoint=False,
        skip_aggregation=args.skip_aggregation,
        use_v2=False
   	).to(device)

elif args.network == 'SwinUNETRv2_EffiDec3D':
	model = SwinUNETR_EffiDec3D(
        img_size=args.img_size,
        in_channels=args.n_channels, #1
        out_channels=out_classes,
        feature_size=args.feature_size,
        n_decoder_channels=args.n_decoder_channels,
        resolution_factor=args.resolution_factor,
        use_checkpoint=False,
        skip_aggregation=args.skip_aggregation,
        use_v2=True
   	).to(device)

elif args.network == 'MedNeXt_M_EffiDec3D':
    model = create_mednextv1_effidec3d(
        args.n_channels,
        out_classes,
        'M',
        n_channels=args.feature_size,
        kernel_size=3,
        deep_supervision=args.ds #True
    ).to(device)

## 3D UX-Net
elif args.network == '3DUXNET':
    #print(args.pretrain)
    if args.pretrain == True:
        #print('here')
        model = UXNET(
            in_chans=args.n_channels,
            out_chans=args.pretrain_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        )
        model.load_state_dict(torch.load(args.pretrained_weights))
        model.out = UnetOutBlock(spatial_dims=3, in_channels=48, out_channels=out_classes)
        model = model.to(device)
    else:
        model = UXNET(
            in_chans=args.n_channels, #1
            out_chans=out_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        ).to(device)

elif args.network == 'SlimUNETR':
    if args.img_size[0] == 96:
        embedding_dim = 27
    else:
        embedding_dim = 64

    model = SlimUNETR(
        in_channels=args.n_channels,
        out_channels=out_classes,
        embed_dim=96,
        embedding_dim=embedding_dim,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ).to(device)

## SwinUNETR
elif args.network == 'SwinUNETR':
    if args.pretrain == True:
        model = SwinUNETR(
            img_size=args.img_size,
            in_channels=args.n_channels,
            out_channels=args.pretrain_classes,
            feature_size=args.feature_size,
            use_checkpoint=False,
        )
        model.load_state_dict(torch.load(args.pretrained_weights))
        model.out = UnetOutBlock(spatial_dims=3, in_channels=48, out_channels=out_classes)
        model = model.to(device)
    else:
        model = SwinUNETR(
            img_size=args.img_size,
            in_channels=args.n_channels, #1
            out_channels=out_classes,
            feature_size=args.feature_size, #48
            use_checkpoint=False,
        ).to(device)

elif args.network == 'SwinUNETRv2':
    model = SwinUNETRv2(
            img_size=args.img_size,
            in_channels=args.n_channels, #1
            out_channels=out_classes,
            feature_size=args.feature_size, #48
            use_checkpoint=False,
            use_v2=True
        ).to(device)

elif args.network == 'nnUNet':
    model = Generic_UNet(
        input_channels=args.n_channels,
        base_num_features=48,
        num_classes=out_classes,
        num_pool=4,
        num_conv_per_stage=2,
        conv_op=nn.Conv3d,
        norm_op=nn.BatchNorm3d,
        dropout_op=nn.Dropout3d,
        max_num_features=512,
        deep_supervision=False,
    ).to(device)

## nnFormer
elif args.network == 'nnFormer':
    if args.pretrain == True:
        from networks.nnFormer.nnFormer_seg import final_patch_expanding
        final_layer = []
        model = nnFormer(input_channels=args.n_channels, num_classes=args.pretrain_classes)
        model.load_state_dict(torch.load(args.pretrained_weights))
        final_layer.append(final_patch_expanding(192, out_classes, patch_size=[2,4,4]))
        model.final = nn.ModuleList(final_layer)
        model = model.to(device)
    else:
        model = nnFormer(input_channels=args.n_channels, num_classes=out_classes).to(device) #1

## UNETR
elif args.network == 'UNETR':
    if args.pretrain == True:
        model = UNETR(
            in_channels=args.n_channels,
            out_channels=args.pretrain_classes,
            img_size=args.img_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
        model.load_state_dict(torch.load(args.pretrained_weights))
        model.out = UnetOutBlock(spatial_dims=3, in_channels=48, out_channels=out_classes)
        model = model.to(device)
    else:
        model = UNETR(
            in_channels=args.n_channels, #1
            out_channels=out_classes,
            img_size=args.img_size,
            feature_size=args.feature_size, #16
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)

elif args.network == 'UNETR_PP':
    model = UNETR_PP(
        in_channels=args.n_channels, #1
        out_channels=out_classes,
        img_size=args.img_size,
        feature_size=args.feature_size, #16
        hidden_size=256,
        dims=[32, 64, 128, 256],
        num_heads=4,
        pos_embed="perceptron",
        norm_name="instance",
        dropout_rate=0.0,
        do_ds=False
    ).to(device)

elif args.network == 'MedNeXt_S':
    model = create_mednext_v1(
        args.n_channels,
        out_classes,
        'S',
        kernel_size=5,
        n_channels=args.feature_size,
        deep_supervision=args.ds #False
    ).to(device)

elif args.network == 'MedNeXt_B':
    model = create_mednext_v1(
        args.n_channels,
        out_classes,
        'B',
        kernel_size=5,
        n_channels=args.feature_size,
        deep_supervision=args.ds #False
    ).to(device)

elif args.network == 'MedNeXt_M':
    model = create_mednext_v1(
        args.n_channels,
        out_classes,
        'M',
        kernel_size=5,
        n_channels=args.feature_size,
        deep_supervision=args.ds #False
    ).to(device)

elif args.network == 'MedNeXt_L':
    model = create_mednext_v1(
        args.n_channels,
        out_classes,
        'L',
        kernel_size=5,
        n_channels=args.feature_size,
        deep_supervision=args.ds #False
    ).to(device)

elif args.network == 'SegFormer3D':
    model = SegFormer3D(
        in_channels = args.n_channels,
        sr_ratios = [4, 2, 1, 1],
        embed_dims = [32, 64, 160, 256],
        patch_kernel_size = [7, 3, 3, 3],
        patch_stride = [4, 2, 2, 2],
        patch_padding = [3, 1, 1, 1],
        mlp_ratios = [4, 4, 4, 4],
        num_heads = [1, 2, 5, 8],
        depths = [2, 2, 2, 2],
        decoder_head_embedding_dim = 256,
        num_classes = out_classes,
        decoder_dropout = 0.0,
    ).to(device)

## TransBTS
elif args.network == 'TransBTS':
    if args.pretrain == True:
        #_, model = TransBTS(dataset='flare', _conv_repr=True, _pe_type='learned')
        _, model = TransBTS(num_classes=2, num_channels=1, img_dim=96, _conv_repr=True, _pe_type='learned')
        model.load_state_dict(torch.load(args.pretrained_weights))
        model.endconv = nn.Conv3d(512 // 32, out_classes, kernel_size=1)
        model = model.to(device)
    else:
        #_, model = TransBTS(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
        _, model = TransBTS(num_classes=out_classes, num_channels=args.n_channels, img_dim=args.img_size[0], _conv_repr=True, _pe_type='learned')

        model = model.to(device)

print('Chosen Network Architecture: {}'.format(args.network))


macs, params = get_model_complexity_info(model, (args.n_channels, args.img_size[0], args.img_size[1], args.img_size[2]), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

print("Computing model complexity with fvcore.nn ...")


model.eval()


input_shape = (1, args.n_channels, *args.img_size)
inputs = (torch.randn(input_shape).to(device),)

try:
    flops = FlopCountAnalysis(model, inputs)

    print("=" * 70)
    print("[Per-layer FLOPs]")
    print(flop_count_table(flops))
    print("=" * 70)
    print("[Parameter Summary]")
    print(parameter_count_table(model))
    print("=" * 70)
    print(f"Total FLOPs : {flops.total() / 1e9:.3f} G")
    print(f"Total Params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    print("=" * 70)

except Exception as e:
    print(f"[WARN] fvcore FLOPs calculation failed: {e}")
    print("Use ptflops as fallback after confirming 3D Conv compatibility.")

## Define Loss function and optimizer
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('Loss for training: {}'.format('DiceCELoss'))

# Define Loss function and optimizer ...
if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


from types import SimpleNamespace
feat_cache = SimpleNamespace(bottleneck=None, decoder_feats={}, tri_kernels={})

# --- Eval bank load ---
eval_text_bank = None
eval_bank_ids, eval_bank_id2idx, eval_bank_mat = [], {}, None

_eval_npz_path = args.eval_text_bank_npz or args.text_bank_npz
if _eval_npz_path and os.path.isfile(_eval_npz_path):
    _npz = np.load(_eval_npz_path, allow_pickle=True)
    _ids = list(map(str, _npz["ids"].tolist()))
    _emb = _npz["embeddings"].astype(np.float32)
    eval_text_bank = { i: torch.from_numpy(v) for i, v in zip(_ids, _emb) }
    print(f"[EvalTextBank] loaded: {len(eval_text_bank)} items, dim={_emb.shape[1]}")
    eval_bank_ids = list(eval_text_bank.keys())
    eval_bank_id2idx = {k: i for i, k in enumerate(eval_bank_ids)}
    eval_bank_mat = torch.stack([eval_text_bank[k] for k in eval_bank_ids], dim=0).float()
    eval_bank_mat = F.normalize(eval_bank_mat, p=2, dim=1)
    try:
        eval_bank_mat = eval_bank_mat.to(device, non_blocking=True)
    except RuntimeError:
        pass
else:
    print(f"[EvalTextBank] NOT found: {_eval_npz_path}")



text_bank = None
if args.text_bank_npz and os.path.isfile(args.text_bank_npz):
    _npz = np.load(args.text_bank_npz, allow_pickle=True)
    _ids = list(map(str, _npz["ids"].tolist()))
    _emb = _npz["embeddings"].astype(np.float32)
    # id -> torch.tensor
    text_bank = { i: torch.from_numpy(v) for i, v in zip(_ids, _emb) }
    print(f"[TextBank] loaded: {len(text_bank)} items, dim={_emb.shape[1]}")
else:
    print(f"[TextBank] NOT found: {args.text_bank_npz}")


# ---- Build full-bank tensor & id<->index map (minimal add) ----
if text_bank:
    bank_ids = list(text_bank.keys())                                  # [N]
    bank_id2idx = {k: i for i, k in enumerate(bank_ids)}               # id(str) -> abs index
    bank_mat = torch.stack([text_bank[k] for k in bank_ids], dim=0).float()  # [N, D]
    bank_mat = F.normalize(bank_mat, p=2, dim=1)
    try:
        bank_mat = bank_mat.to(device, non_blocking=True)
    except RuntimeError:
        pass
else:
    bank_ids, bank_id2idx, bank_mat = [], {}, None


def nt_xent_multi(z, y, temp=0.07):
    """
    z: [B, D] (
    """
    sim = (z @ z.t()) / temp                  # [B, B]
    B = z.size(0)
    mask_self = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask_self, float('-inf'))


    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & (~mask_self)  # [B, B]



    log_denom = torch.logsumexp(sim, dim=1)                     # [B]

    sim_pos = sim.masked_fill(~pos_mask, float('-inf'))
    log_pos = torch.logsumexp(sim_pos, dim=1)                   # [B]

    has_pos = pos_mask.any(dim=1)
    loss = -(log_pos - log_denom)[has_pos].mean() if has_pos.any() else z.new_tensor(0.0)
    return loss


def _bottleneck_hook(_m, _in, _out):

    feat_cache.bottleneck = _out


    if bank_mat is None:
        return
    try:
        Cb = int(_out.shape[1])
        global projector, proj_group
        if (projector is None) or (getattr(projector.fc, "in_features", None) != Cb):
            projector = GAPProjector(in_ch=Cb, out_dim=args.text_dim).to(device)

            if proj_group is None:
                base = optimizer.param_groups[0]
                ng = {'params': list(projector.parameters())}
                for k in ("lr", "weight_decay", "betas", "eps"):
                    if k in base: ng[k] = base[k]
                optimizer.add_param_group(ng)
                proj_group = optimizer.param_groups[-1]
            else:
                proj_group['params'] = list(projector.parameters())
                g0 = optimizer.param_groups[0]
                for k in ("lr", "weight_decay", "betas", "eps"):
                    if k in g0: proj_group[k] = g0[k]

        with torch.no_grad():
            img_vec = projector(_out)          # [B, text_dim], L2 norm
            sims   = img_vec @ bank_mat.t()    # [B, N]
            top_ix = sims.argmax(dim=1)        # [B]
            feat_cache.sel_text = bank_mat.index_select(0, top_ix)  # [B, text_dim]
            feat_cache.top_idx  = top_ix
    except Exception:
        feat_cache.sel_text = None
        feat_cache.top_idx  = None



def attach_bottleneck_hook_by_network(model, net_name: str):
    """
    Attach a bottleneck forward hook based on network name.
    Returns (hook_handle, attr_name) on success, otherwise (None, None).
    """
    hook_targets = {
        "3DUXNET": "encoder5",
        "SwinUNETRv2": "encoder10",
    }

    attr = hook_targets.get(net_name)
    if attr is None:
        print(f"[hook] WARN: no hook target mapped for network '{net_name}'")
        return None, None
    if not hasattr(model, attr):
        print(f"[hook] WARN: model has no attribute '{attr}', skipping hook")
        return None, None
    mod = getattr(model, attr)
    if not isinstance(mod, torch.nn.Module):
        print(f"[hook] WARN: model.{attr} is not a torch.nn.Module")
        return None, None

    handle = mod.register_forward_hook(_bottleneck_hook)
    print(f"[hook] attached on model.{attr} for network '{net_name}'")
    return handle, attr


hook_handle, hook_target = (None, None)
if args.mode == 'train':
    hook_handle, hook_target = attach_bottleneck_hook_by_network(model, args.network)

# projector: 3D -> GAP -> Linear(text_dim) -> LN -> L2Norm
class GAPProjector(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc   = nn.Linear(in_ch, out_dim)
        self.ln   = nn.LayerNorm(out_dim)
    def forward(self, x):
        v = self.pool(x).flatten(1)       # [B, C]
        z = self.ln(self.fc(v))           # [B, out_dim]
        z = F.normalize(z, p=2, dim=1)
        return z


proj_in_ch = getattr(model, "n_decoder_channels", None)
if proj_in_ch is None:

    proj_in_ch = getattr(model, "cls_head_in_channels", 48)


projector = None

scaler = make_scaler()


_id_re = re.compile(args.id_regex)

def extract_id_from_batch_meta(batch):

    meta = batch.get("image_meta_dict", {})
    fn = meta.get("filename_or_obj", None)
    if fn is None:
        return None

        fns = [str(x) for x in fn]
    else:
        fns = [str(fn)]
    ids = []
    for p in fns:
        m = _id_re.search(os.path.basename(p))
        ids.append(m.group(1) if m else None)
    return ids

def _pos_idx_from_batch_ids(batch_ids):
    """Convert batch ids from filenames to memory-bank indices."""
    pos_idx, rows = [], []
    for i, cid in enumerate(batch_ids or []):
        j = bank_id2idx.get(str(cid))
        if j is not None:
            pos_idx.append(j)
            rows.append(i)
    if not pos_idx:
        return None, None
    pos_idx = torch.tensor(pos_idx, device=device, dtype=torch.long)
    return pos_idx, rows


def build_train_sel_text_from_batch(batch):
    """
    Build [B, D] text embeddings from training batch ids using the memory bank.
    Return None if any id is missing (safe fallback: skip fusion for this batch).
    """
    if bank_mat is None:
        return None
    ids = extract_id_from_batch_meta(batch) or []
    if len(ids) == 0:
        return None

    idxs = []
    for sid in ids:
        j = bank_id2idx.get(str(sid))
        if j is None:
            return None
        idxs.append(j)

    sel = bank_mat.index_select(
        0, torch.tensor(idxs, device=bank_mat.device, dtype=torch.long)
    )
    return sel

def build_eval_sel_text_from_batch(batch, fallback_dim=None):
    if eval_bank_mat is None:
        return None
    ids = extract_id_from_batch_meta(batch) or []

    if len(ids) == 0:
        return None
    idxs = []
    for sid in ids:
        j = eval_bank_id2idx.get(str(sid))
        if j is None:
            return None
        idxs.append(j)
    sel = eval_bank_mat.index_select(0, torch.tensor(idxs, device=eval_bank_mat.device, dtype=torch.long))

    return sel


# InfoNCE
def info_nce(img_z, txt_z, temp=0.07):
    logits_i2t = (img_z @ txt_z.t()) / temp
    targets = torch.arange(img_z.size(0), device=img_z.device)
    loss_i2t = F.cross_entropy(logits_i2t, targets)
    logits_t2i = (txt_z @ img_z.t()) / temp
    loss_t2i = F.cross_entropy(logits_t2i, targets)
    return 0.5 * (loss_i2t + loss_t2i)

args.output = os.path.abspath(args.output)

root_dir = os.path.join(args.output,args.network,args.dataset)
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)

last_ckpt = os.path.join(root_dir, "last_model.pth")

ckpt = None
if os.path.isfile(last_ckpt):
    size = os.path.getsize(last_ckpt)
    if size > 0:
        try:
            ckpt = torch.load(last_ckpt, map_location=device)
            print(f"=> Loaded checkpoint (step {ckpt.get('global_step', '?')})")
        except EOFError:
            print(f"[WARN] Checkpoint file {last_ckpt!r} is corrupted (EOF). Removing it.")
            os.remove(last_ckpt)
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e!r}. Removing it.")
            os.remove(last_ckpt)
    else:
        print(f"[WARN] Checkpoint file {last_ckpt!r} is empty. Removing it.")
        os.remove(last_ckpt)

# if ckpt was loaded, restore; otherwise default init
if ckpt:

    ckpt_opt = ckpt.get("optimizer_state_dict", {})
    ckpt_groups = len(ckpt_opt.get("param_groups", []))



    while len(optimizer.param_groups) < ckpt_groups:
        base = optimizer.param_groups[0]
        placeholder = {'params': []}
        for k in ("lr", "weight_decay", "betas", "eps"):
            if k in base:
                placeholder[k] = base[k]
        optimizer.add_param_group(placeholder)


    proj_group = optimizer.param_groups[-1]


    model.load_state_dict(ckpt["model_state_dict"])


    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except ValueError as e:
        print(f"[optimizer] load_state_dict failed -> using fresh optimizer: {e}")

        Optim = torch.optim.AdamW if args.optim == 'AdamW' else torch.optim.Adam
        optimizer = Optim(model.parameters(), lr=args.lr)

        proj_group = {'params': []}
        for k in ("lr", "weight_decay", "betas", "eps"):
            if k in base: proj_group[k] = base[k]
        optimizer.add_param_group(proj_group)
        proj_group = optimizer.param_groups[-1]


    scaler.load_state_dict(ckpt["scaler_state_dict"])

    global_step = ckpt.get("global_step", 0)
    dice_val_best = ckpt.get("dice_val_best", 0.0)
    global_step_best = ckpt.get("global_step_best", 0)

else:
    print("=> No valid checkpoint, training from scratch")
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0


    base = optimizer.param_groups[0]
    proj_group = {'params': []}
    for k in ("lr", "weight_decay", "betas", "eps"):
        if k in base:
            proj_group[k] = base[k]
    optimizer.add_param_group(proj_group)
    t_dir = os.path.join(root_dir, 'tensorboard')
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)

def validation(epoch_iterator_val):
    model.eval()
    dice_vals = []
    per_class_accumulator = []
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val, start=1):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            x, y = (batch["image"].to(device), batch["label"].to(device))

            feat_cache.bottleneck = None
            feat_cache.sel_text = None
            feat_cache.top_idx = None


            if out_classes == 9:
                val_labels[val_labels == 5]  = 0
                val_labels[val_labels == 9]  = 0
                val_labels[val_labels == 10] = 0
                val_labels[val_labels == 12] = 0
                val_labels[val_labels == 13] = 0
                val_labels[val_labels == 11] = 5


            sel_text_batch = build_eval_sel_text_from_batch(batch)


            def net_with_text(x):

                if sel_text_batch is None:
                    return model(x)
                Bwin = x.shape[0]
                if sel_text_batch.shape[0] == Bwin:
                    sel = sel_text_batch
                elif sel_text_batch.shape[0] == 1:
                    sel = sel_text_batch.expand(Bwin, -1)
                else:

                    return model(x)
                return model(x, sel_text=sel)

            with autocast_ctx(enabled=False):
                val_outputs = sliding_window_inference_1out(
                    val_inputs,
                    (args.img_size[0], args.img_size[1], args.img_size[2]),
                    args.val_batch,
                    net_with_text,
                    overlap=args.overlap
                )


                val_labels_list  = decollate_batch(val_labels)
                val_outputs_list = decollate_batch(val_outputs)
                val_labels_convert  = [post_label(v) for v in val_labels_list]
                val_outputs_convert = [post_pred(v)  for v in val_outputs_list]


                dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                dice_metric.reset()


                dice_metric_pc(y_pred=val_outputs_convert, y=val_labels_convert)
                pc = dice_metric_pc.aggregate().detach().cpu().numpy()
                dice_metric_pc.reset()


                pc = np.asarray(pc, dtype=np.float32).ravel()
                per_class_accumulator.append(pc)


                show_k = min(5, len(pc))
                brief = ", ".join([f"{class_names[i]}:{float(pc[i]):.3f}" for i in range(show_k)])
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f mean_dice=%2.5f | %s)"
                    % (global_step, eval_num, dice, np.mean(dice_vals), brief)
                )

    mean_dice_val = float(np.mean(dice_vals)) if dice_vals else 0.0
    writer.add_scalar('Validation/MeanDice', mean_dice_val, global_step)


    if per_class_accumulator:
        pc_mean = np.mean(np.stack(per_class_accumulator, axis=0), axis=0).astype(np.float32)  # shape: [K-1]

        for i, name in enumerate(class_names):
            writer.add_scalar(f"Validation/Dice_cls_{name}", float(pc_mean[i]), global_step)
        writer.add_scalars(
            "Validation/Dice_per_class",
            {name: float(pc_mean[i]) for i, name in enumerate(class_names)},
            global_step
        )
    else:
        pc_mean = np.zeros(len(class_names), dtype=np.float32)

    return mean_dice_val, pc_mean

# id matching
def _forward_train(m, x, sel_override=None):

    sel = sel_override if sel_override is not None else getattr(feat_cache, "sel_text", None)

    try:
        return m(x, sel_text=sel)
    except TypeError:
        pass
    try:
        return m(x, mode='train')
    except TypeError:
        pass
    return m(x)


# def _forward_train(m, x):

#     sel = getattr(feat_cache, "sel_text", None)
#

#     try:
#         return m(x, sel_text=sel)
#     except TypeError:
#         pass
#

#     try:
#         return m(x, mode='train')
#     except TypeError:
#         pass
#

#     return m(x)
def train(global_step, train_loader, dice_val_best, global_step_best):
    from collections import Counter
    global projector, _projector_checked, optimizer, proj_group

    model.train()
    if projector is not None:
        projector.train()

    epoch_loss = 0.0
    ema_seg = None
    ema_ctr = None

    epoch_iterator = tqdm(train_loader, desc="Training (X / X) (total=NA, seg=NA, ctr=NA)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator, start=1):
        x, y = (batch["image"].to(device), batch["label"].to(device))
        feat_cache.bottleneck = None


        _SEGON = True
        _CTRON = True

        with autocast_ctx(enabled=False):

            sel_text_batch = build_train_sel_text_from_batch(batch)
            p = _forward_train(model, x, sel_override=sel_text_batch)

            # === Print GAP projector params only at first iteration ===
            # if global_step == 0:
            #     if projector is not None:
            #         total_params = sum(p.numel() for p in projector.parameters())
            #         print("\n===== GAP Projector Parameters =====")
            #         for name, p in projector.named_parameters():
            #             print(f"{name:20s} : {p.numel():8d}")
            #         print(f"Total Parameters: {total_params}")
            #         print("====================================\n")
            #     else:
            #         print("[WARN] projector is None at step 0")

        # with autocast_ctx(enabled=False):

            # p = _forward_train(model, x)



            P = p if isinstance(p, list) else [p]
            if out_classes == 9:
                y[y == 5]  = 0
                y[y == 9]  = 0
                y[y == 10] = 0
                y[y == 12] = 0
                y[y == 13] = 0
                y[y == 11] = 5

            # -------------------------

            # -------------------------
            loss_terms = []
            seg_loss_only = torch.tensor(0.0, device=y.device)

            # (A) Segmentation loss
            if _SEGON:
                ss = [[i] for i in range(len(P))] if (args.ds and len(P) > 1) else [[0]]
                _seg = 0.0
                target_size = (y.shape[-3], y.shape[-2], y.shape[-1])
                for s in ss:
                    if not s:
                        continue
                    iout = None
                    for idx in s:
                        up = F.interpolate(P[idx], size=target_size, mode='trilinear', align_corners=False)
                        iout = up if iout is None else iout + up
                    _seg = _seg + loss_function(iout, y)
                loss_terms.append(_seg)
                seg_loss_only = _seg.detach()


                if _CTRON and (hook_handle is not None) and (bank_mat is not None) and \
                        (feat_cache.bottleneck is not None) and (args.contrastive_w > 0):

                    img_feat_3d = feat_cache.bottleneck
                    C = int(img_feat_3d.shape[1])

                    # projector ensure
                    if (projector is None) or (projector.fc.in_features != C):
                        projector = GAPProjector(in_ch=C, out_dim=args.text_dim).to(device)
                        proj_group['params'] = list(projector.parameters())
                        g0 = optimizer.param_groups[0]
                        for k in ("lr", "weight_decay", "betas", "eps"):
                            if k in g0: proj_group[k] = g0[k]
                        print(f"[Projector] (re)create: in_ch -> {C}")

                    # image embedding
                    img_z = projector(img_feat_3d)  # [B, D], L2-norm

                    # targets: only rows whose id exists in memory bank
                    batch_ids = extract_id_from_batch_meta(batch) or []
                    pos_idx, valid_rows = _pos_idx_from_batch_ids(batch_ids)

                    if (pos_idx is not None) and (valid_rows is not None) and (len(valid_rows) > 0):
                        row_ix = torch.as_tensor(valid_rows, device=img_z.device, dtype=torch.long)
                        img_sel = img_z.index_select(0, row_ix)  # [Bv, D]

                        bank_local = bank_mat if bank_mat.device == img_sel.device else \
                            bank_mat.to(img_sel.device, non_blocking=True)
                        bank_local = bank_local.detach()  # no grad into bank

                        logits = (img_sel @ bank_local.t()) / args.contrastive_t
                        targets = pos_idx.to(img_sel.device, dtype=torch.long)  # [Bv]
                        c_loss = F.cross_entropy(logits, targets)

                        loss_terms.append(args.contrastive_w * c_loss)

                        # logging / EMA
                        seg_val = float(seg_loss_only.item()) if _SEGON else 0.0
                        ctr_val = float(c_loss.item())
                        ema_seg = seg_val if ema_seg is None else 0.9 * ema_seg + 0.1 * seg_val
                        ema_ctr = ctr_val if ema_ctr is None else 0.9 * ema_ctr + 0.1 * ctr_val
                        if _SEGON:
                            writer.add_scalar('Training/SegLoss', seg_val, global_step)
                        writer.add_scalar('Training/ContrastiveLoss', ctr_val, global_step)
                    else:
                        # no valid ids in bank -> contrastive skipped (seg only)
                        if (global_step % 50) == 0:
                            print("[InfoNCE] skipped (0 valid ids in bank for this batch)")

        # === backward/step ===
        if not loss_terms:

            optimizer.zero_grad(set_to_none=True)
            epoch_iterator.set_description(
                f"Training ({global_step} / {max_iterations}) "
                f"(total=skip, seg={(f'{ema_seg:.5f}' if ema_seg is not None else '0.00000')}, "
                f"ctr={(f'{ema_ctr:.5f}' if ema_ctr is not None else '0.00000')} | no loss terms)"
            )
            writer.add_scalar('Training/TotalLoss', 0.0, global_step)
            global_step += 1
            continue

        loss = sum(loss_terms)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_loss += float(loss.item())

        epoch_iterator.set_description(
            f"Training ({global_step} / {max_iterations}) "
            f"(total={float(loss.item()):.5f}, seg={(f'{ema_seg:.5f}' if ema_seg is not None else '0.00000')}, "
            f"ctr={(f'{ema_ctr:.5f}' if ema_ctr is not None else '0.00000')})"
        )


        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val, pc_mean = validation(epoch_iterator_val)

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)

            pc_mean = np.asarray(pc_mean, dtype=np.float32).ravel()
            show_k = min(5, len(pc_mean))
            cls_summary = ", ".join([f"{class_names[i]}:{float(pc_mean[i]):.4f}" for i in range(show_k)])

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print("Model Was Saved ! Current Best Avg. Dice: %.6f | Current Avg. Dice: %.6f | Per-class (top-%d): [%s]"
                      % (dice_val_best, dice_val, show_k, cls_summary))
            else:
                print("Model Was Not Saved ! Current Best Avg. Dice: %.6f | Current Avg. Dice: %.6f | Per-class (top-%d): [%s]"
                      % (dice_val_best, dice_val, show_k, cls_summary))

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "global_step": global_step,
                "dice_val_best": dice_val_best,
                "global_step_best": global_step_best,
            }, last_ckpt)

        writer.add_scalar('Training/TotalLoss', float(loss.item()), global_step)
        global_step += 1

    return global_step, dice_val_best, global_step_best

max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
# dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)



dice_metric_pc = DiceMetric(include_background=False, reduction="none", get_not_nans=False)

class_names = [str(i) for i in range(1, out_classes)]

# global_step = 1
# dice_val_best = 0.0
# global_step_best = 1
epoch_loss_values = []
metric_values = []

#args.mode = 'test'
#args.overlap = 0.5
#args.overlap_mode = 'gaussian'

if args.mode == 'train':
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )


best_pth = os.path.join(root_dir, "best_metric_model.pth")
if os.path.isfile(best_pth) and os.path.getsize(best_pth) > 0:
    model.load_state_dict(torch.load(best_pth, map_location=device))
    model.eval()
else:
    print(f"[WARN] {best_pth} not found or empty. Skipping final load.")

epoch_iterator_val = tqdm(
    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
)

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        #jaccard = metric.binary.jc(pred, gt)
        #asd = metric.binary.assd(pred, gt)
        return dice, hd95#, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0#, 1, 0
    else:
        return 0, 0#, 0, 0

def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from matplotlib import cm

def generate_colormap(num_classes):
    """
    Generate a colormap for a given number of classes using specified colors.
    Args:
        num_classes: Number of classes including the background.
    Returns:
        colormap: Dictionary mapping class indices to RGB values.
    """
    # List of preferred colors for classes
    color_list = [
        (0, 0, 0),          # Background (black)
        (255, 0, 0),        # Red
        (0, 255, 0),        # Green
        (135, 206, 250),    # Skyblue
        (255, 165, 0),      # Orange
        (255, 0, 255),      # Magenta
        (128, 0, 128),      # Purple
        (255, 255, 0),      # Yellow
        (205, 133, 63),     # Peru
        (128, 128, 0),      # Olive
        (75, 0, 130),       # Indigo
        (0, 255, 128),      # Lime
        (0, 0, 255),        # Blue
        (255, 20, 147)      # Deeppink
    ]

    # Ensure we have enough colors for the number of classes
    assert num_classes <= len(color_list), "Not enough colors specified for the number of classes."

    # Map each class index to its corresponding RGB value
    colormap = {cls: color_list[cls] for cls in range(num_classes)}
    return colormap


def overlay_segmentation(image, segmentation, colormap):
    """
    Overlay segmentation on the input image using a specified colormap.
    Args:
        image: Original 8-bit image (2D numpy array).
        segmentation: Segmentation mask (2D numpy array with class indices).
        colormap: Dictionary mapping class indices to RGB values.
    Returns:
        overlay: RGB image with segmentation overlay.
    """
    overlay = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
    for cls, color in colormap.items():
        if cls == 0:  # Skip background (keep original image pixels)
            continue
        overlay[segmentation == cls] = np.array(color)
    return overlay

# Define class labels dictionary

def save_metrics_to_csv(trained_weights, dataset_name, network_name, overlap, overlap_mode,
                        per_class_dice, per_class_hd, mean_dice_val, mean_hd_val, csv_filename, out_classes):

    file_exists = os.path.isfile(csv_filename)


    fg_classes = [str(i) for i in range(1, out_classes)]


    header = ["Trained_Weights", "Dataset", "Network", "Overlap", "OverlapMode"]
    header += [f"Dice_{cid}" for cid in fg_classes] + ["Mean_Dice"]
    header += [f"HD_{cid}"   for cid in fg_classes] + ["Mean_HD"]


    row = [
        trained_weights,
        dataset_name,
        network_name,
        overlap,
        overlap_mode
    ]


    pc_hd_mean   = per_class_hd.mean(axis=0)   if per_class_hd.size   else np.array([])


    row += [f"{float(d):.4f}" for d in pc_dice_mean]

    row.append(f"{float(mean_dice_val):.4f}")

    row += [f"{float(h):.4f}" for h in pc_hd_mean]

    row.append(f"{float(mean_hd_val):.4f}")


    with open(csv_filename, mode='a', newline='') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)

# Example usage:
csv_filename = "last_validation_metrics_btcv.csv"


# Save the metrics into the CSV file
# save_metrics_to_csv(args.output, args.dataset, args.network, args.overlap, args.overlap_mode, class_labels, per_class_dice, per_class_hd, mean_dice_val, mean_hd_val, csv_filename)
mean_dice_val, pc_mean = validation(tqdm(val_loader, desc="Validate (final)", dynamic_ncols=True))


per_class_dice = np.expand_dims(np.asarray(pc_mean, dtype=np.float32).ravel(), axis=0)
per_class_hd   = np.zeros_like(per_class_dice, dtype=np.float32)

save_metrics_to_csv(
    trained_weights=args.output,
    dataset_name=args.dataset,
    network_name=args.network,
    overlap=args.overlap,
    overlap_mode=args.overlap_mode,
    per_class_dice=per_class_dice,
    per_class_hd=per_class_hd,
    mean_dice_val=float(mean_dice_val),
    mean_hd_val=0.0,
    csv_filename=csv_filename,
    out_classes=out_classes
)
