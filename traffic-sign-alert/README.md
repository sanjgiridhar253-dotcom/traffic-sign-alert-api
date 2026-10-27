# Traffic Sign Alert — End-to-End

Two tracks live in this repo:

1. **IEEE / GTSRB** — compact CNN (~0.63M params), track-wise split, metrics for the paper  
2. **Accessibility demo** — dashcam → detect → **VGG16** (52-class `signs_general`) → voice / screen flash for hearing or deaf drivers

Full step list + **manual intervention points**: see [`END_TO_END.md`](END_TO_END.md).

## Quick start (accessibility demo)

**MANUAL:** place a short dashcam clip at `data/raw/sample_dashcam.mp4`, then:

```bash
.venv_linux/bin/python -m src.pipeline.demo_e2e \
  --source data/raw/sample_dashcam.mp4 \
  --weights models/vgg16_signs_general.pt \
  --driver-profile both \
  --max-seconds 5
```

## Layout

```
data/raw/signs_general/              # cropped 52-class set (already present)
data/raw/sample_dashcam.mp4          # YOU add (dashcam video)
data/raw/gtsrb_zips/                 # YOU add if GTSRB download fails
data/processed/signs_general_split/  # VGG16 train/val/test
data/processed/gtsrb_split/          # created by prepare_data
data/processed/frames/               # extracted video frames
models/vgg16_signs_general.pt        # trained VGG16 (present)
models/gtsrb_compact.pt              # after GTSRB training
src/gtsrb/                           # IEEE paper pipeline
src/pipeline/demo_e2e.py             # accessibility end-to-end demo
```

## Retrain VGG16 (optional)

```bash
python src/models/train_vgg16.py \
  --data data/processed/signs_general_split \
  --out models/vgg16_signs_general.pt \
  --epochs 20
```
