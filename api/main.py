"""SignGuard API — the gate, served over HTTP.

    uvicorn main:app --reload --port 8000

One endpoint does the work. It never returns a bare label: every response
carries the calibrated confidence, whether the gate accepted it, and the
threshold it was judged against, because the whole point of the project is
that a label without those three numbers cannot be trusted.

The occlusion parameter runs the same sticker attack the audit notebook uses,
so the demo and the reported numbers come from identical code.
"""
from __future__ import annotations

import base64
import io
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

CKPT_PATH = os.environ.get(
    "SIGNGUARD_CKPT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "signguard_calibrated.pth"),
)

app = FastAPI(title="SignGuard API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: dict = {"ready": False, "error": None}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def load_model() -> None:
    if not os.path.exists(CKPT_PATH):
        STATE["error"] = (
            f"checkpoint not found at {CKPT_PATH}. Run a notebook from "
            "notebooks/ and put signguard_calibrated.pth in models/."
        )
        return

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    classes = ckpt["classes"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    STATE.update(
        ready=True,
        error=None,
        model=model,
        classes=classes,
        img_size=ckpt["img_size"],
        mean=ckpt["mean"],
        std=ckpt["std"],
        temperature=float(ckpt.get("temperature", 1.0)),
        threshold=float(ckpt.get("threshold", 0.90)),
        test_acc=ckpt.get("test_acc"),
        selective_acc=ckpt.get("selective_acc"),
        coverage=ckpt.get("coverage"),
    )


@app.on_event("startup")
def _startup() -> None:
    load_model()


# --------------------------------------------------------------------------
# the sticker attack — identical to the audit notebook
# --------------------------------------------------------------------------
PALETTE = np.array(
    [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.90, 0.85, 0.10],
     [0.10, 0.20, 0.80], [0.85, 0.15, 0.15]],
    dtype=np.float32,
)


def apply_stickers(tensor: torch.Tensor, frac: float, seed: int = 0,
                   n_patches: int = 2) -> tuple[torch.Tensor, float]:
    """Cover `frac` of the image with solid rectangles.

    Patches may overlap and clip at the border, so the fraction actually
    covered is measured and returned rather than assumed.
    """
    if frac <= 0:
        return tensor, 0.0

    rng = np.random.default_rng(seed)
    _, h, w = tensor.shape
    out = tensor.clone()
    mask = torch.zeros(h, w, dtype=torch.bool)
    area_each = frac * h * w / n_patches

    for _ in range(n_patches):
        ar = rng.uniform(0.4, 2.5)
        ph = max(1, min(int(round(math.sqrt(area_each * ar))), h))
        pw = max(1, min(int(round(math.sqrt(area_each / ar))), w))
        top = int(rng.integers(0, h - ph + 1))
        left = int(rng.integers(0, w - pw + 1))
        colour = torch.from_numpy(PALETTE[rng.integers(len(PALETTE))]).view(3, 1, 1)
        out[:, top:top + ph, left:left + pw] = colour
        mask[top:top + ph, left:left + pw] = True

    return out, float(mask.float().mean())


def to_data_url(tensor: torch.Tensor) -> str:
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.get("/meta")
def meta():
    if not STATE["ready"]:
        return {"ready": False, "error": STATE["error"]}
    return {
        "ready": True,
        "classes": STATE["classes"],
        "temperature": round(STATE["temperature"], 4),
        "threshold": round(STATE["threshold"], 4),
        "test_acc": STATE.get("test_acc"),
        "selective_acc": STATE.get("selective_acc"),
        "coverage": STATE.get("coverage"),
    }


@app.post("/reload")
def reload_model():
    load_model()
    return meta()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    occlusion: float = Form(0.0),
    seed: int = Form(0),
):
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=STATE["error"] or "model not loaded")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="not a readable image")

    size = STATE["img_size"]
    base = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])(image)

    occluded, covered = apply_stickers(base, max(0.0, min(occlusion, 0.6)), seed)
    normalised = transforms.Normalize(STATE["mean"], STATE["std"])(occluded)

    temperature = STATE["temperature"]
    with torch.no_grad():
        logits = STATE["model"](normalised.unsqueeze(0))
        probs = F.softmax(logits / temperature, dim=1)[0]
        energy = float(temperature * torch.logsumexp(logits / temperature, dim=1))

    conf, idx = probs.max(0)
    conf, idx = float(conf), int(idx)
    thr = float(threshold) if threshold is not None else STATE["threshold"]
    top_p, top_i = probs.topk(min(5, len(probs)))

    return {
        "accepted": conf >= thr,
        "label": STATE["classes"][idx],
        "confidence": conf,
        "threshold": thr,
        "temperature": temperature,
        "energy": energy,
        "occlusion_requested": occlusion,
        "occlusion_actual": covered,
        "preview": to_data_url(occluded),
        "topk": [
            {"label": STATE["classes"][int(i)], "prob": float(p)}
            for p, i in zip(top_p, top_i)
        ],
    }
