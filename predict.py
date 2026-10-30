#!/usr/bin/env python3
"""SignGuard AI - traffic sign classification with an explicit "I don't know".

    python predict.py sign.jpg
    python predict.py photos/*.jpg --topk 3
    python predict.py sign.jpg --threshold 0.95

The model never answers on confidence alone. A prediction is returned only when its
temperature-calibrated confidence clears the threshold that was chosen during the
safety audit (see notebooks/signguard_model.ipynb). Otherwise the answer is UNKNOWN,
which in a real system means "hand this frame to another sensor or to a human".
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "models", "signguard_calibrated.pth")


def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        sys.exit(
            f"checkpoint not found: {ckpt_path}\n"
            "Run notebooks/signguard_model.ipynb in Colab and download "
            "signguard_calibrated.pth into models/."
        )
    ckpt = torch.load(ckpt_path, map_location=device)

    classes = ckpt["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])
    meta = {
        "classes": classes,
        "temperature": ckpt.get("temperature", 1.0),
        "threshold": ckpt.get("threshold", 0.90),
    }
    return model, tf, meta


@torch.no_grad()
def predict(model, tf, meta, path, device, topk=3):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = F.softmax(logits / meta["temperature"], dim=1)[0].cpu()
    energy = (meta["temperature"] *
              torch.logsumexp(logits / meta["temperature"], dim=1)).item()

    conf, idx = probs.max(0)
    conf, idx = conf.item(), idx.item()
    top_p, top_i = probs.topk(min(topk, len(probs)))

    return {
        "file": os.path.basename(path),
        "accepted": conf >= meta["threshold"],
        "label": meta["classes"][idx],
        "confidence": conf,
        "energy": energy,
        "topk": [(meta["classes"][i], p.item()) for p, i in zip(top_p, top_i)],
    }


def main():
    ap = argparse.ArgumentParser(description="SignGuard AI traffic sign classifier")
    ap.add_argument("images", nargs="+", help="image file(s)")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the audited confidence threshold")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, meta = load_model(args.checkpoint, device)
    if args.threshold is not None:
        meta["threshold"] = args.threshold

    results = []
    for path in args.images:
        if not os.path.exists(path):
            print(f"skipped (not found): {path}", file=sys.stderr)
            continue
        results.append(predict(model, tf, meta, path, device, args.topk))

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"\nthreshold {meta['threshold']:.2f}   temperature {meta['temperature']:.2f}\n")
    for r in results:
        if r["accepted"]:
            head = f"\033[92m{r['label']}\033[0m   {r['confidence']*100:.1f}%"
        else:
            head = (f"\033[93mUNKNOWN\033[0m   best guess '{r['label']}' at "
                    f"{r['confidence']*100:.1f}%, below threshold")
        print(f"{r['file']}\n  {head}")
        for name, p in r["topk"][1:]:
            print(f"    {p*100:5.1f}%  {name}")
        print()


if __name__ == "__main__":
    main()
