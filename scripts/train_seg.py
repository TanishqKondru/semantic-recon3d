"""Train SegFormer instrument segmentation on the patient-split StereoMIS export.

Model selection uses val only: the mean IoU of P1 and a fixed simulated-fluorescence copy of P1
(no training sequence has fluorescence frames, P3 does). Test (P3) is evaluated once, with the best
checkpoint, split by imaging mode. Weights are an EMA of the training weights.

    python scripts/train_seg.py data/seg/stereomis outputs/segmentation/segformer_b2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from sr3d.eval.seg_metrics import SegMeter
from sr3d.segmentation.data import IMAGENET_MEAN, IMAGENET_STD, SPLITS, SegDataset, is_fluorescence, load_index
from sr3d.segmentation.model import build_segformer, ce_dice_loss, logits_full


@torch.inference_mode()
def evaluate(model, loader, by_mode: bool = False) -> dict:
    """Metrics over the loader; with by_mode also split into white-light and fluorescence frames."""
    was_training = model.training
    model.eval()
    meters = {"all": SegMeter(), "white_light": SegMeter(), "fluorescence": SegMeter()}
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = logits_full(model, x.cuda(non_blocking=True)).argmax(1).cpu().numpy().astype(bool)
        rgb = ((x.numpy().transpose(0, 2, 3, 1) * IMAGENET_STD + IMAGENET_MEAN) * 255).clip(0, 255).astype(np.uint8)
        for p, g, im in zip(pred, y.numpy().astype(bool), rgb):
            meters["all"].update(p, g)
            if by_mode:
                meters["fluorescence" if is_fluorescence(im) else "white_light"].update(p, g)
    model.train(was_training)
    if not by_mode:
        return meters["all"].summary()
    return {k: m.summary() for k, m in meters.items() if m.n_frames}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--encoder", default="nvidia/mit-b2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=6, help="8 at 512 px spills past 8 GB VRAM")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-5, help="encoder LR; decoder gets 10x")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--p-fluor", type=float, default=0.3, help="probability of simulated fluorescence per crop")
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay of evaluated/saved weights")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    train = SegDataset(load_index(a.data, SPLITS["train"]), crop=a.crop, p_fluor=a.p_fluor)
    val_pairs = load_index(a.data, SPLITS["val"])
    loaders = {"val": SegDataset(val_pairs), "val_fluor": SegDataset(val_pairs, fluor_eval=True),
               "test": SegDataset(load_index(a.data, SPLITS["test"]))}
    loaders = {k: DataLoader(d, batch_size=4, num_workers=a.workers) for k, d in loaders.items()}
    train_loader = DataLoader(train, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                              worker_init_fn=SegDataset.worker_init, pin_memory=True, persistent_workers=True)

    model = build_segformer(a.encoder).cuda().train()
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)
    enc = [p for n, p in model.named_parameters() if n.startswith("segformer.")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("segformer.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": dec, "lr": a.lr * 10}], weight_decay=0.01)
    total = a.max_steps or a.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * 10], total_steps=total, pct_start=0.05)

    print(f"train {len(train)} frames, val {len(loaders['val'].dataset)}, test {len(loaders['test'].dataset)}, "
          f"{total} steps", flush=True)
    history, best, step, t0 = [], -1.0, 0, time.time()
    for epoch in range(a.epochs):
        losses = []
        for x, y in train_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ce_dice_loss(logits_full(model, x.cuda(non_blocking=True)), y.cuda(non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema.update_parameters(model)
            losses.append(loss.item())
            step += 1
            if step >= total:
                break
        val, val_fl = evaluate(ema.module, loaders["val"]), evaluate(ema.module, loaders["val_fluor"])
        score = (val["iou"] + val_fl["iou"]) / 2
        history.append({"epoch": epoch, "step": step, "loss": float(np.mean(losses)), "val": val,
                        "val_fluor": val_fl, "score": score})
        if score > best:
            best = score
            ema.module.save_pretrained(out / "best")
        print(f"epoch {epoch:2d} step {step:5d} loss {np.mean(losses):.3f}  val IoU {val['iou']:.3f} "
              f"Dice {val['dice']:.3f} FA {val['empty_false_alarm_rate']:.2f} | sim-fluor IoU {val_fl['iou']:.3f} "
              f"| score {score:.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)
        if step >= total:
            break

    model = model.from_pretrained(out / "best").cuda()
    test = evaluate(model, loaders["test"], by_mode=True)
    best_epoch = max(history, key=lambda h: h["score"])
    result = {"encoder": a.encoder, "args": vars(a), "best_epoch": best_epoch["epoch"], "val": best_epoch["val"],
              "val_fluor": best_epoch["val_fluor"], "test": test, "history": history,
              "minutes": (time.time() - t0) / 60}
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    for mode, m in test.items():
        print(f"test (P3) {mode}:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}))


if __name__ == "__main__":
    main()
