# SignGuard AI

Traffic sign recognition that knows when **not** to answer.

A MobileNetV2 classifier on GTSRB (43 German traffic sign classes), plus a safety audit that
measures whether its confidence can be trusted and turns the answer into a gated decision:
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
compared on far-OOD inputs (CIFAR-10 photos) and near-OOD inputs (real signs with a quarter of
their surface covered), scored by AUROC.

**3. Occlusion robustness.** The physical sticker attack, simulated: opaque patches covering 5% to
40% of the sign face. Accuracy is plotted against the model's confidence *on the answers it gets
wrong*. The gap between those two curves is the silent-failure zone.

---

## Results

<!-- paste outputs/results_table.md from the notebook here after running it -->

| Metric | No gate | Confidence gate |
|---|---|---|
| Inputs answered | 100% | _tbd_ |
| Accuracy on answers | _tbd_ | _tbd_ |
| Calibration error (ECE) | _tbd_ | _tbd_ |
| Silent failure, clean signs | _tbd_ | _tbd_ |
| Silent failure, 20% covered | _tbd_ | _tbd_ |
| Non-signs wrongly classified | 100% | _tbd_ |

OOD detection AUROC — far (CIFAR-10) / near (25% covered signs): _tbd_

![calibration](docs/calibration.png)
![occlusion](docs/occlusion_curve.png)

---

## Project layout

```
signguard-ai/
├── notebooks/signguard_model.ipynb   training (Part A) + safety audit (Part B)
├── predict.py                        single-image CLI with the UNKNOWN path
├── models/                           signguard_calibrated.pth  (not in git)
├── docs/                             audit figures
└── requirements.txt
```

## Running it

**Training and the audit** run in Google Colab on a free T4 GPU. Open
`notebooks/signguard_model.ipynb`, set `Runtime -> Change runtime type -> T4 GPU`, and run every
cell. The dataset downloads itself through `torchvision`; no Kaggle token is needed. Roughly 25
minutes end to end.

The notebook writes `signguard_calibrated.pth` — weights plus the fitted temperature and the
audited threshold — and an `outputs/` folder with `audit.json` and every figure.

**Inference** runs locally on CPU:

```bash
pip install -r requirements.txt
# download signguard_calibrated.pth from Colab into models/
python predict.py path/to/sign.jpg
python predict.py photos/*.jpg --json
```

## Method notes

- **No horizontal flip in augmentation.** Mirroring a traffic sign changes its meaning — "turn
  left ahead" becomes "turn right ahead" — so a flip would teach the model wrong labels. The
  augmentation is affine jitter, small rotation and colour jitter only.
- **Two-phase training.** The ImageNet backbone is frozen while the new 43-way head settles, then
  the last four blocks are unfrozen at a 10× smaller learning rate. Traffic signs are far from the
  ImageNet distribution, so frozen features alone leave accuracy on the table.
- **The GTSRB test split is a genuine holdout** — photographed on different roads on different
  days than the training split, not a reshuffle of the same images.
- **The threshold is chosen on validation, never on test.** Fitting the operating point on the
  test set would make every number below meaningless.

## Limitations

- Simulated stickers are opaque rectangles. Real adversarial patches are optimised against the
  model and are substantially stronger.
- GTSRB is German signage in daylight. Night, rain and other countries' signage are untested — and
  are precisely the near-OOD cases the gate exists to catch.
- The gate reduces silent failures; it does not eliminate them. Anything built on this needs a
  second sensor.

## Stack

PyTorch · torchvision · MobileNetV2 (ImageNet transfer learning) · scikit-learn · matplotlib

---

Built as a portfolio project. Companion to
[DriveSafe AI](https://github.com/mukhammads-dev/drivesafe-ai) — webcam driver-fatigue detection
with OpenCV.
