"""
train_yolo.py

Fine-tunes a pretrained YOLOv8 checkpoint on your unified traffic sign
dataset. This is your primary "DL CNN model" from the architecture slide -
it does detection AND classification in one pass, which is why it's the
faster path to a working end-to-end demo than a separate detector+VGG16
two-stage setup.

Usage:
    python src/models/train_yolo.py --data configs/data.yaml --epochs 50 --model yolov8n.pt

Start with yolov8n.pt (nano) to get a fast baseline running end-to-end,
then move to yolov8s.pt / yolov8m.pt once the pipeline works and you're
optimizing for accuracy over iteration speed.
"""
import argparse

from ultralytics import YOLO


def train(data_yaml: str, base_model: str, epochs: int, imgsz: int, batch: int, out_dir: str):
    model = YOLO(base_model)  # loads pretrained COCO weights, swaps head for your classes

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=out_dir,
        name="sign_detector",
        patience=15,          # early stop if val mAP plateaus
        val=True,
        plots=True,
    )
    print("Training complete. Best weights at:", model.trainer.best)

    # run once on the held-out test split reported in your data.yaml
    metrics = model.val(data=data_yaml, split="test")
    print(f"Test mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    return model.trainer.best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default="models/runs")
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.out)
