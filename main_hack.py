import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from model_chestray import ChestRayNet
from chestray_labels import LABELS, NUM_CLASSES
from preprocess import build_loaders, compute_pos_weight
from post_process import (
    evaluate as eval_fn,
    compute_thresholds,
)
from report_builder import build_card  

_CAM_AVAILABLE = True
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:
    _CAM_AVAILABLE = False


# ----------------------------- small utils ----------------------------------
def _load_image(image_path, img_size):
    im = Image.open(image_path).convert('RGB')
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    t = tfm(im).unsqueeze(0)
    return im, t


def _maybe_make_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def _amp_context():
    """Return an autocast context and a GradScaler that work across PyTorch versions."""
    use_cuda = torch.cuda.is_available()
    try:
        from torch import amp as _amp  # new API
        scaler = _amp.GradScaler(enabled=use_cuda)
        def _ctx():
            return _amp.autocast(device_type='cuda' if use_cuda else 'cpu', enabled=use_cuda)
        return _ctx, scaler
    except Exception:
        from torch.cuda import amp as _camp  # legacy API
        scaler = _camp.GradScaler(enabled=use_cuda)
        def _ctx():
            return _camp.autocast(enabled=use_cuda)
        return _ctx, scaler


# --------------------------- single-image predict ----------------------------
@torch.no_grad()
def predict_once(weights, image_path, img_size=320, backbone='efficientnet_b0',
                 thresholds=None, make_cam=False, cam_label=None, cam_out_dir='cam_outputs'):
    """
    Returns: (card_dict, cam_path_or_None)
    - If make_cam=True, writes a Grad-CAM overlay (top-1 or specified label) and
      sets card['cam_image_path'] accordingly.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = ChestRayNet(backbone=backbone, pretrained=False, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    # Image
    im, t = _load_image(image_path, img_size)
    t = t.to(device)

    # Forward
    probs = torch.sigmoid(model(t))[0].cpu().numpy()

    # Build screening card
    card = build_card(
        probs_or_scores=probs,
        thresholds_path=thresholds,   # None -> safe defaults inside build_card
        lang="en",
        report_labels=None,
        cam_image_path=None,
        frontal_view=True
    )

    cam_path = None
    if make_cam:
        if not _CAM_AVAILABLE:
            card['warnings'] = card.get('warnings', []) + ["Grad-CAM not available: install grad-cam>=1.5.0"]
        else:
            # Which index to visualize
            if cam_label and cam_label in LABELS:
                idx = LABELS.index(cam_label)
            else:
                idx = int(np.argmax(probs))

            target_layer = model.gradcam_target_layer()
            cam = GradCAM(model=model, target_layers=[target_layer])
            heat = cam(input_tensor=t, targets=[ClassifierOutputTarget(idx)], eigen_smooth=False)[0]
            h, w = heat.shape
            im_rs = im.resize((w, h))
            rgb = np.asarray(im_rs).astype(np.float32) / 255.0
            overlay = show_cam_on_image(rgb, heat, use_rgb=True)

            _maybe_make_dir(cam_out_dir)
            cam_path = os.path.join(cam_out_dir, f"cam_{LABELS[idx].replace(' ','_')}.png")
            Image.fromarray(overlay).save(cam_path)
            card['cam_image_path'] = cam_path

    return card, cam_path


# --------------------------------- train -------------------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_ctx, scaler = _amp_context()

    train_loader, val_loader, *_ = build_loaders(
        args.data_root, args.img_size, args.batch_size,
        args.num_workers, args.policy, not args.no_frontal_only
    )

    model = ChestRayNet(
        backbone=args.backbone, pretrained=True, use_cbam=True, num_classes=NUM_CLASSES
    ).to(device)

    # pos_weight for class imbalance
    pw_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    pos_weight = compute_pos_weight(pw_loader, device)

    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best = -1.0
    bad = 0
    _maybe_make_dir(args.out_dir)

    # Safe args dump
    safe_args = {k: v for k, v in vars(args).items() if not callable(v)}
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(safe_args, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for x, y, m, _ in loop:
            x = x.to(device, non_blocking=True)
            y = y.to(device)
            m = m.to(device)

            optim.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = model(x)
                loss_mat = criterion(logits, y)  # [B, C]
                loss = (loss_mat * m).sum() / (m.sum() + 1e-8)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            loop.set_postfix(loss=f'{loss.item():.4f}',
                             lr=f'{optim.param_groups[0]["lr"]:.2e}')

        sched.step()

        # Validation
        metrics = eval_fn(model, val_loader, device)
        mean_auc = metrics['macro_auroc']
        print(f'Val macro AUROC: {mean_auc:.4f}')
        with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Checkpoint / early stop
        score = mean_auc if not np.isnan(mean_auc) else -1.0
        if score > best:
            best = score
            bad = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'chestray_best.pt'))
            print('Saved new best model.')
        else:
            bad += 1
            if bad >= args.patience:
                print('Early stopping.')
                break


# ---------------------------------- eval -------------------------------------
def evaluate_cmd(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader, *_ = build_loaders(
        args.data_root, args.img_size, args.batch_size,
        args.num_workers, args.policy, not args.no_frontal_only
    )
    model = ChestRayNet(backbone=args.backbone, pretrained=False, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    metrics = eval_fn(model, val_loader, device)
    _maybe_make_dir(os.path.dirname(args.out_metrics))
    with open(args.out_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    thr = compute_thresholds(model, val_loader, device)
    _maybe_make_dir(os.path.dirname(args.out_thresholds))
    with open(args.out_thresholds, 'w') as f:
        json.dump(thr, f, indent=2)
    print(f"Saved thresholds to {args.out_thresholds}")


# --------------------------------- infer -------------------------------------
def infer_cmd(args):
    card, cam_path = predict_once(
        weights=args.weights,
        image_path=args.image,
        img_size=args.img_size,
        backbone=args.backbone,
        thresholds=args.thresholds,
        make_cam=args.cam_top1,
        cam_label=None,
        cam_out_dir=args.cam_out_dir
    )
    if args.save_card:
        _maybe_make_dir(os.path.dirname(args.save_card))
        with open(args.save_card, 'w') as f:
            json.dump(card, f, indent=2)
    print(json.dumps(card, indent=2))


# ---------------------------------- cam --------------------------------------
def cam_cmd(args):
    if not _CAM_AVAILABLE:
        raise RuntimeError("Grad-CAM not available. Please `pip install grad-cam>=1.5.0`")
    # Reuse predict_once with make_cam=True
    card, cam_path = predict_once(
        weights=args.weights,
        image_path=args.image,
        img_size=args.img_size,
        backbone=args.backbone,
        thresholds=args.thresholds,
        make_cam=True,
        cam_label=args.target_label,
        cam_out_dir=args.out_dir
    )
    print(f"Saved {cam_path}")


# -------------------------------- parser -------------------------------------
def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    # train
    tr = sub.add_parser('train')
    tr.add_argument('--data_root', required=True)
    tr.add_argument('--policy', default='ignore', choices=['ignore','zeros','ones'])
    tr.add_argument('--img_size', type=int, default=320)
    tr.add_argument('--batch_size', type=int, default=32)
    tr.add_argument('--epochs', type=int, default=8)
    tr.add_argument('--lr', type=float, default=3e-4)
    tr.add_argument('--backbone', default='efficientnet_b0',
                    choices=['efficientnet_b0','efficientnet_b2','efficientnet_b4'])
    tr.add_argument('--num_workers', type=int, default=4)
    tr.add_argument('--out_dir', default='outputs')
    tr.add_argument('--patience', type=int, default=3)
    tr.add_argument('--no_frontal_only', action='store_true')
    tr.set_defaults(func=train)

    # eval
    ev = sub.add_parser('eval')
    ev.add_argument('--data_root', required=True)
    ev.add_argument('--weights', required=True)
    ev.add_argument('--img_size', type=int, default=320)
    ev.add_argument('--batch_size', type=int, default=64)
    ev.add_argument('--backbone', default='efficientnet_b0',
                    choices=['efficientnet_b0','efficientnet_b2','efficientnet_b4'])
    ev.add_argument('--num_workers', type=int, default=4)
    ev.add_argument('--policy', default='ignore', choices=['ignore','zeros','ones'])
    ev.add_argument('--out_metrics', default='eval_outputs/metrics.json')
    ev.add_argument('--out_thresholds', default='eval_outputs/thresholds.json')
    ev.add_argument('--no_frontal_only', action='store_true')
    ev.set_defaults(func=evaluate_cmd)

    # infer 
    inf = sub.add_parser('infer')
    inf.add_argument('--weights', required=True)
    inf.add_argument('--image', required=True)
    inf.add_argument('--img_size', type=int, default=320)
    inf.add_argument('--backbone', default='efficientnet_b0',
                     choices=['efficientnet_b0','efficientnet_b2','efficientnet_b4'])
    inf.add_argument('--thresholds', default=None)
    inf.add_argument('--cam_top1', action='store_true', help='Also save Grad-CAM for top-1 label')
    inf.add_argument('--cam_out_dir', default='cam_outputs')
    inf.add_argument('--save_card', default=None, help='Optional path to write card JSON')
    inf.set_defaults(func=infer_cmd)

    # cam 
    cam = sub.add_parser('cam')
    cam.add_argument('--weights', required=True)
    cam.add_argument('--image', required=True)
    cam.add_argument('--img_size', type=int, default=320)
    cam.add_argument('--backbone', default='efficientnet_b0',
                     choices=['efficientnet_b0','efficientnet_b2','efficientnet_b4'])
    cam.add_argument('--thresholds', default=None)
    cam.add_argument('--target_label', default=None, help="If omitted, visualizes top-1")
    cam.add_argument('--out_dir', default='cam_outputs')
    cam.set_defaults(func=cam_cmd)

    return ap


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
