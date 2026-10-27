# End-to-End Guide (IEEE paper + accessibility demo)

This repo now has **two connected tracks**:

| Track | Matches | Goal |
|-------|---------|------|
| **A. GTSRB compact CNN** | IEEE docx (classification paper) | Train/eval a ~0.63M CNN on GTSRB with track-wise split |
| **B. Dashcam accessibility demo** | Your driver-assist aim + IEEE “future work: detection” | 5s video → find sign → classify (VGG16) → voice/screen alert |

---

## MANUAL INTERVENTION CHECKLIST (do these yourself)

Steps marked **MANUAL** cannot be fully automated (data rights, hardware, or personal choice).

| # | Step | Who | What you put where |
|---|------|-----|--------------------|
| **M1** | Get GTSRB zips (Track A only) | **MANUAL** if download fails | Place into `data/raw/gtsrb_zips/`: `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip`, `GTSRB_Final_Test_GT.zip` |
| **M2** | Add dashcam video (Track B) | **MANUAL** | Put your clip at `data/raw/sample_dashcam.mp4` (~5–15 s with visible signs) |
| **M3** | Choose driver profile | **MANUAL** | `--driver-profile hearing` \| `deaf` \| `both` |
| **M4** | Optional: real TTS voice | **MANUAL** | `pip install pyttsx3` (Linux may need `espeak`) |
| **M5** | Optional: better detector | **MANUAL** | Label boxes + train YOLO, then pass `--yolo-weights models/runs/sign_detector/weights/best.pt` |
| **M6** | Fill IEEE paper numbers | **MANUAL** | Copy `reports/gtsrb_eval/metrics.json` into Table 4 placeholders `[xx.xx]` |
| **M7** | Author names / GPU details | **MANUAL** | Edit IEEE docx front matter and Section 5 |

Everything else below is **automated** once M1/M2 are done.

---

## Track A — IEEE GTSRB pipeline (automated after M1)

```bash
cd traffic-sign-alert
source .venv_linux/bin/activate   # or use .venv_linux/bin/python

# 1) Download + track-wise split + 32x32 ROI crops
python -m src.gtsrb.prepare_data \
  --raw data/raw/gtsrb \
  --zips data/raw/gtsrb_zips \
  --out data/processed/gtsrb_split \
  --val-frac 0.2 --seed 42

# If download fails → do M1, then:
python -m src.gtsrb.prepare_data --skip-download ...

# 2) Train compact CNN (Adam, early stopping, no horizontal flip)
python -m src.gtsrb.train \
  --data data/processed/gtsrb_split \
  --out models/gtsrb_compact.pt \
  --epochs 50 --batch-size 64 --lr 1e-3 --patience 10

# 3) Evaluate → accuracy, macro-F1, confusion matrix
python -m src.gtsrb.evaluate \
  --data data/processed/gtsrb_split \
  --weights models/gtsrb_compact.pt \
  --split test \
  --out reports/gtsrb_eval
```

**Outputs for the paper**

| File | Use in IEEE doc |
|------|-----------------|
| `models/gtsrb_compact.pt` | Trained model |
| `models/gtsrb_compact.history.json` | Fig. 2 training curves |
| `reports/gtsrb_eval/metrics.json` | Table 4 accuracy / macro-F1 |
| `reports/gtsrb_eval/confusion_matrix.png` | Fig. 3 |

---

## Track B — Dashcam accessibility demo (automated after M2)

You already have VGG16 weights (`models/vgg16_signs_general.pt`).

```bash
# Optional: extract frames for inspection / offline runs
python src/preprocessing/extract_frames.py \
  --video data/raw/sample_dashcam.mp4 \
  --out data/processed/frames \
  --fps 5

# Live-style 5 second end-to-end demo
python -m src.pipeline.demo_e2e \
  --source data/raw/sample_dashcam.mp4 \
  --weights models/vgg16_signs_general.pt \
  --driver-profile both \
  --max-seconds 5

# Deaf-driver mode (screen flash, no reliance on audio)
python -m src.pipeline.demo_e2e \
  --source data/raw/sample_dashcam.mp4 \
  --driver-profile deaf \
  --max-seconds 5

# Webcam for 5 seconds
python -m src.pipeline.demo_e2e --source webcam --max-seconds 5
```

**Pipeline flow**

```
video/webcam/frames
   → detect boxes (YOLO if provided, else color proposals)
   → crop → VGG16 classify (52 classes)
   → action_map (sign name + required action + urgency)
   → alert: screen flash banner + optional TTS + vibrate log
```

**Directories**

| Path | Contents | Who creates it |
|------|----------|----------------|
| `data/raw/sample_dashcam.mp4` | Your dashcam clip | **MANUAL (M2)** |
| `data/processed/frames/` | Sampled frames | `extract_frames.py` |
| `models/vgg16_signs_general.pt` | Classifier | Already trained |
| `data/processed/alert_log.tsv` | Fired alerts log | `demo_e2e.py` |
| `src/action_inference/action_map.py` | 52-class → action text | Code (done) |
| `src/detection/sign_proposals.py` | Demo detector | Code (done) |
| `src/pipeline/demo_e2e.py` | Full demo entrypoint | Code (done) |

---

## What is already done vs still MANUAL

| Capability | Status |
|------------|--------|
| VGG16 on signs_general (52 classes) | Done |
| Unfrozen fine-tune + high confidence | Done |
| 52-class → action / urgency / modality | Done |
| Screen flash overlay (deaf mode) | Done |
| TTS voice (hearing mode) | Done if `pyttsx3` installed (**M4**) |
| Color-based sign proposals | Done (demo quality) |
| GTSRB compact CNN code | Done |
| GTSRB dataset on disk | **MANUAL M1** |
| Dashcam video on disk | **MANUAL M2** |
| Production YOLO detector | **MANUAL M5** (optional upgrade) |
| IEEE table numbers filled | **MANUAL M6** after Track A eval |

---

## Recommended order this week

1. **M2** — add `data/raw/sample_dashcam.mp4`
2. Run Track B `demo_e2e` with `--driver-profile deaf` and `both`
3. If you need the IEEE GTSRB results: **M1** → Track A train/eval → **M6** fill tables
4. Later: **M5** train YOLO for better detection than color heuristics
