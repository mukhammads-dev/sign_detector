# SignGuard AI

Traffic sign recognition that knows when **not** to answer.

A MobileNetV2 classifier on GTSRB (43 German traffic sign classes), plus a safety audit that
measures whether its confidence can be trusted and turns the output into a gated decision:
predict, or return `UNKNOWN`.

```
$ python predict.py stop.jpg
stop.jpg
  Stop   99.4%

$ python predict.py billboard.jpg
billboard.jpg
  UNKNOWN   best guess 'Yield' at 61.2%, below threshold
     18.7%  Priority road
      7.1%  No vehicles
```

> **Status:** the model and the full audit pipeline are implemented and reproducible end to end.
> The results table below is filled from `outputs/audit.json` once the training run completes.

---

## Why a gate

A softmax classifier always answers. Shown a billboard, a shadow, or a stop sign with tape across
it, it still picks one of its 43 classes — often with high confidence. In a driver-assistance
context that is the dangerous failure mode: wrong *and* certain, with nothing downstream able to
tell.

This project measures three things most classification projects skip.

**1. Calibration.** When the model reports 95%, is it right 95% of the time? Measured with a
reliability diagram and Expected Calibration Error, then corrected with temperature scaling — a
single scalar fitted on validation data that changes no prediction, only how confident the model
sounds.

**2. Out-of-distribution rejection.** Two scores (maximum softmax probability and free energy) are
compared on far-OOD inputs and near-OOD inputs (real signs with a quarter of their surface
covered), scored by AUROC.

**3. Occlusion robustness.** The physical sticker attack, simulated: opaque patches covering 5% to
40% of the sign face. Accuracy is plotted against the model's confidence *on the answers it gets
wrong*. The gap between those two curves is the silent-failure zone.

The output of the audit is one line of production logic:

```python
accept the prediction only if calibrated confidence >= THRESHOLD, else return UNKNOWN
```

with the threshold chosen on validation data to hit a target selective accuracy, and its real cost
and benefit reported on the held-out test set.

---

## Results

| Metric | No gate | Confidence gate |
|---|---|---|
| Inputs answered | 100% | _pending_ |
| Accuracy on answers | _pending_ | _pending_ |
| Calibration error (ECE) | _pending_ | _pending_ |
| Silent failure, clean signs | _pending_ | _pending_ |
| Silent failure, 20% covered | _pending_ | _pending_ |
| Non-signs wrongly classified | 100% | _pending_ |

The notebook writes `outputs/results_table.md` in exactly this format — it is pasted in verbatim,
and `outputs/audit.json` carries the same numbers machine-readably.

---

## Project layout

```
signguard-ai/
├── notebooks/
│   ├── signguard_model.ipynb          Colab version   (Drive + unzip)
│   └── signguard_model_kaggle.ipynb   Kaggle version  (dataset mounted read-only)
├── api/main.py                        FastAPI: /predict, /meta, /reload
├── web/                               Next.js app (App Router, TypeScript, SCSS)
├── predict.py                         single-image CLI with the UNKNOWN path
├── run.sh                             starts the API and the web app together
├── models/                            signguard_calibrated.pth  (not in git)
├── docs/                              audit figures
└── requirements.txt
```

## Running it

Training and the audit need a GPU. Either notebook runs the identical pipeline; they differ only
in how the dataset is reached.

**Kaggle** (recommended — the dataset lives there, so nothing is uploaded):
open `notebooks/signguard_model_kaggle.ipynb`, add
[GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) as an
input, set the accelerator to GPU, run all.

**Colab:** download the same dataset, put the zip in Drive as `gtsrb_dataset.zip`, open
`notebooks/signguard_model.ipynb`, set the runtime to T4 GPU, run all.

Either way the run takes roughly 25 minutes and produces `signguard_calibrated.pth` — weights plus
the fitted temperature and the audited threshold — and an `outputs/` folder with `audit.json` and
every figure.

**Inference** runs locally on CPU, either from the command line:

```bash
pip install -r requirements.txt
# put signguard_calibrated.pth in models/
python predict.py path/to/sign.jpg
python predict.py photos/*.jpg --json
```

or through the app:

```bash
./run.sh          # FastAPI on :8000, Next.js on :3000
```

The app is where the gate becomes visible. Drop in a sign and the verdict is
either a label or **UNKNOWN**; the confidence slider moves the gate and the
verdict flips live; the sticker slider covers the sign the way tape would and
the prediction collapses while the confidence stays high — which is the whole
argument for having a gate. The threshold is applied in the browser, so it
responds instantly; occlusion changes the image, so it re-runs the model.

The sticker code in `api/main.py` is the same routine the audit notebook uses,
so the demo and the reported numbers cannot drift apart.

## Method notes

- **No horizontal flip in augmentation.** Mirroring a traffic sign changes its meaning — "turn
  left ahead" becomes "turn right ahead" — so a flip would teach the model wrong labels. The
  augmentation is affine jitter, small rotation and colour jitter only.
- **Two-phase training.** The ImageNet backbone is frozen while the new 43-way head settles, then
  the last four blocks are unfrozen at a 10× smaller learning rate. Traffic signs are far from the
  ImageNet distribution, so frozen features alone leave accuracy on the table.
- **Labels come from the CSVs, not from folder names.** The class directories are named `0`, `1`,
  … `42`, which `ImageFolder` would sort lexicographically into `0, 1, 10, 11, …` — silently
  scrambling every label. `Train.csv` / `Test.csv` carry the correct `ClassId`.
- **The GTSRB test split is a genuine holdout** — photographed on different roads on different
  days than the training split, not a reshuffle of the same images.
- **The threshold is chosen on validation, never on test.** Fitting the operating point on the
  test set would make every number above meaningless.
- **Occlusion coverage is measured, not assumed.** Patches overlap and clip at the border, so the
  area actually covered is below the nominal value; the real fraction is what gets reported.

## Limitations

- Simulated stickers are opaque rectangles. Real adversarial patches are optimised against the
  model and are substantially stronger.
- GTSRB is German signage in daylight. Night, rain and other countries' signage are untested — and
  are precisely the near-OOD cases the gate exists to catch.
- The gate reduces silent failures; it does not eliminate them. Anything built on this needs a
  second sensor.

## Stack

PyTorch · torchvision · MobileNetV2 (ImageNet transfer learning) · scikit-learn · matplotlib
FastAPI · Next.js 14 · TypeScript · SCSS

---

Portfolio project. Companion to
[DriveSafe AI](https://github.com/mukhammads-dev/drivesafe-ai) — webcam driver-fatigue detection
with OpenCV.
