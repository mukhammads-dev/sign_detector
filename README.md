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

MobileNetV2, 8 epochs (4 frozen + 4 fine-tuned), 12,630 held-out test images.

| Metric | No gate | Confidence gate (T=0.86, thr=0.73) |
|---|---|---|
| Inputs answered | 100% | 91.7% |
| Accuracy on answers | 91.62% | **95.62%** |
| Calibration error (ECE) | 2.06% | 3.08% |
| Silent failure, clean signs | 8.38% | **4.02%** |
| Silent failure, 20% covered | 61.4% | **23.7%** |
| Non-signs wrongly classified | 100% | **35.7%** |

OOD detection, AUROC — far (Food101 photos) / near (signs 25% covered):

| Score | far-OOD | near-OOD |
|---|---|---|
| Max softmax probability | 0.923 | 0.869 |
| **Energy** | **0.994** | **0.936** |

Refusing 8.3% of inputs cuts the silent-failure rate in half on clean signs and by a factor of
2.6 under the sticker attack. That is the trade the gate makes, measured rather than asserted.

### Three things the audit found that the accuracy number hides

**Validation accuracy overstated the model by six points.** 97.74% on the validation split,
91.62% on the official test split. The validation images come from the same recording sessions as
the training images; the test set was shot on different roads on different days. The second
number is the honest one.

**Temperature scaling made calibration worse, and that is a finding, not a bug.** The fitted
temperature was 0.86 — below 1, meaning the model looked *under*-confident on validation, where it
is 97.7% accurate. Sharpening it there makes it overconfident on test, where it is 91.6% accurate:
ECE moved from 2.06% to 3.08%. Temperature scaling assumes the calibration set and the deployment
set are drawn from the same distribution. Here they are not, and the method inherits the gap.
The same mechanism explains the threshold: 0.73 was chosen on validation to reach 99% selective
accuracy, and delivered 95.62% on test.

**Energy beats max-softmax by a wide margin on out-of-distribution input** — 0.994 against 0.923
on far-OOD, 0.936 against 0.869 on near-OOD. The gate currently thresholds softmax confidence, so
35.7% of non-sign images still get an answer. Gating on energy instead is the single clearest
improvement available, and the audit is what makes that visible.

Raw numbers: `outputs/audit.json`. Figures: `docs/`.

### Fine-tuning was necessary, and the log proves it

```
            val accuracy
[1/8]  0.6387   backbone frozen
[2/8]  0.6965
[3/8]  0.7216
[4/8]  0.7114   ← plateau, and a step backwards
>>> fine-tuning starts
[5/8]  0.9305   ← +22 points in one epoch
[6/8]  0.9488
[7/8]  0.9774
[8/8]  0.9770
```

Frozen ImageNet features stall at 72% on traffic signs. Unfreezing the last four blocks at a 10×
smaller learning rate recovers 22 points in a single epoch — a measurement of how far this domain
sits from ImageNet, not a guess.

---

## Project layout

```
signguard-ai/
├── notebooks/
│   ├── signguard_model.ipynb          Colab version   (Drive + unzip)
│   └── signguard_model_kaggle.ipynb   Kaggle version  (dataset mounted read-only)
├── predict.py                         single-image CLI with the UNKNOWN path
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

**Inference** runs locally on CPU:

```bash
pip install -r requirements.txt
# put signguard_calibrated.pth in models/
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
- The gate reduces silent failures; it does not eliminate them. More than a third of non-sign
  images still get an answer. Anything built on this needs a second sensor.
- Calibration is fitted on a validation split that is easier than the test set, so both the
  temperature and the threshold under-deliver on deployment data. The honest fix is a calibration
  set drawn from the deployment distribution, not from the training recordings.
- 91.6% is well short of the 99%+ published on GTSRB. Eight epochs at 224×224 with a partly frozen
  backbone is a modest recipe; the project spends its effort on the audit, not on the leaderboard.

## Stack

PyTorch · torchvision · MobileNetV2 (ImageNet transfer learning) · scikit-learn · matplotlib

---

Portfolio project. Companion to
[DriveSafe AI](https://github.com/mukhammads-dev/drivesafe-ai) — webcam driver-fatigue detection
with OpenCV.
