from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric
from monai.data.utils import no_collation
import torch
from load_datasets_transforms import data_loader, data_transforms, infer_post_transforms
import os
import argparse
from collections import OrderedDict
from networks.swin_unetr_effidec3d import SwinUNETR as SwinUNETRv2
from networks.UXNet_3D.network_backbone import UXNET_EffiDec3D

# -----------------------------

# -----------------------------
parser = argparse.ArgumentParser(description='3D UX-Net inference hyperparameters for medical image segmentation')
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--dataset', type=str, default='flare', required=True)
parser.add_argument('--network', type=str, default='3DUXNET_EffiDec3D', required=True)
parser.add_argument('--trained_weights', default='', required=True)
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--sw_batch_size', type=int, default=4)
parser.add_argument('--overlap', type=float, default=0.5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--cache_rate', type=float, default=0.1)
parser.add_argument('--num_workers', type=int, default=2)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# -----------------------------

# -----------------------------
test_samples, out_classes = data_loader(args)
test_files = [{"image": i, "label": l} for i, l in zip(test_samples["images"], test_samples["labels"])]

set_determinism(seed=0)
test_transforms = data_transforms(args)
post_transforms = infer_post_transforms(args, test_transforms, out_classes)

test_ds = CacheDataset(
    data=test_files, transform=test_transforms,
    cache_rate=args.cache_rate, num_workers=args.num_workers
)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">>> device: {device}, cuda_available={torch.cuda.is_available()}")

# -----------------------------

# -----------------------------
if args.network == '3DUXNET':
    model = UXNET(
        in_chans=1, out_chans=out_classes,
        depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
        drop_path_rate=0, layer_scale_init_value=1e-6, spatial_dims=3,
    ).to(device)

elif args.network == '3DUXNET_EffiDec3D':
    model = UXNET_EffiDec3D(
        in_chans=1, out_chans=out_classes,
        depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
        n_decoder_channels=48,
        drop_path_rate=0, layer_scale_init_value=1e-6,
        spatial_dims=3,
        skip_aggregation="addition",
        resolution_factor=2
    ).to(device)

elif args.network == 'SwinUNETR':
    model = SwinUNETR(
        img_size=(96, 96, 96), in_channels=1, out_channels=out_classes,
        feature_size=48, use_checkpoint=False,
    ).to(device)

elif args.network == 'nnFormer':
    model = nnFormer(input_channels=1, num_classes=out_classes).to(device)

elif args.network == 'UNETR':
    model = UNETR(
        in_channels=1, out_channels=out_classes, img_size=(96, 96, 96),
        feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
        pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0,
    ).to(device)

elif args.network == 'SwinUNETRv2':
    model = SwinUNETRv2(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=out_classes,
        feature_size=48,
        use_checkpoint=False,
        use_v2=True
    ).to(device)

elif args.network == 'TransBTS':
    _, model = TransBTS(_conv_repr=True, _pe_type='learned')
    model = model.to(device)

else:
    raise ValueError(f"Unknown network name: {args.network}")


# -----------------------------

# -----------------------------
def load_state_dict_safely(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_sd[nk] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f">>> load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing: print("    missing keys:", list(missing)[:8])
    if unexpected: print("    unexpected keys:", list(unexpected)[:8])

load_state_dict_safely(model, args.trained_weights)
model.eval()

# -----------------------------

# -----------------------------

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
post_label = AsDiscrete(to_onehot=out_classes)


# -----------------------------

# -----------------------------
roi_size = (96, 96, 96)
case_dices = []

with torch.no_grad():
    print(f">>> Start inference: {len(test_ds)} cases, sw_batch_size={args.sw_batch_size}, overlap={args.overlap}")
    for i, batch in enumerate(test_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        preds = sliding_window_inference(
            images, roi_size, args.sw_batch_size, model, overlap=args.overlap
        )


        preds_post = [post_pred(p) for p in decollate_batch(preds)]
        labels_post = [post_label(l) for l in decollate_batch(labels)]
        dice_metric(y_pred=preds_post, y=labels_post)
        dice_value = dice_metric.aggregate().item()
        dice_metric.reset()

        case_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])
        case_dices.append(dice_value)
        print(f"[{i+1}/{len(test_loader)}] {case_name} Dice: {dice_value:.4f}")


        batch["pred"] = preds
        _ = [post_transforms(b) for b in decollate_batch(batch)]

mean_dice = sum(case_dices) / len(case_dices)
print(">>> ----------------------------------------------------")
print(f">>> Mean Dice over {len(case_dices)} cases: {mean_dice:.4f}")
print(">>> Inference done.")
