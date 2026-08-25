"""Fine-tune YOLO-World-M on Cloud-to-BIM wall-token histograms.

The histogram channels carry physical meaning, so the training recipe deliberately
disables colour jitter, mosaic and geometric deformations. A horizontal flip is
safe because it only reverses the wall direction.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml
from ultralytics import YOLOWorld
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.world.train import WorldTrainer
from ultralytics.utils.torch_utils import unwrap_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "artifacts" / "cloud2bim_yoloworld_m_dataset_v1"
DEFAULT_WEIGHTS = ROOT / "yolov8m-worldv2.pt"
DEFAULT_PROJECT = ROOT / "artifacts" / "cloud2bim_yoloworld_m_training"


class RectWorldTrainer(WorldTrainer):
    """YOLO-World trainer that keeps the native 2:1 wall-histogram aspect ratio."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=True,
            stride=gs,
            multi_modal=mode == "train",
        )
        if mode == "train":
            self.set_text_embeddings([dataset], batch)
        return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--square-padding",
        action="store_true",
        help="Use the stock square-padded trainer instead of native 2:1 rectangular batches.",
    )
    return parser.parse_args()


def balanced_smoke_records(records: list[dict], split: str, limit: int, seed: int) -> list[dict]:
    candidates = [record for record in records if record["split"] == split]
    positive = [record for record in candidates if record.get("objects")]
    negative = [record for record in candidates if not record.get("objects")]
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    pos_count = min(len(positive), limit // 2)
    selected = positive[:pos_count] + negative[: limit - pos_count]
    rng.shuffle(selected)
    return selected


def prepare_smoke_yaml(dataset: Path) -> Path:
    records = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    smoke_dir = dataset / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    train_records = balanced_smoke_records(records, "train", 64, 42)
    val_records = balanced_smoke_records(records, "val", 32, 43)

    train_list = smoke_dir / "train.txt"
    val_list = smoke_dir / "val.txt"
    train_list.write_text("\n".join(record["image"] for record in train_records) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(record["image"] for record in val_records) + "\n", encoding="utf-8")

    config = {
        "path": dataset.as_posix(),
        "train": train_list.as_posix(),
        "val": val_list.as_posix(),
        "test": (dataset / "images" / "test").as_posix(),
        "names": {0: "door", 1: "window"},
    }
    output = smoke_dir / "dataset_smoke.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    weights = args.weights.resolve()
    project = args.project.resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if not (dataset / "dataset.yaml").exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    smoke = args.mode == "smoke"
    data_yaml = prepare_smoke_yaml(dataset) if smoke else dataset / "dataset.yaml"
    epochs = args.epochs if args.epochs is not None else (1 if smoke else 30)
    name = args.name or (f"smoke_m_{args.imgsz}" if smoke else f"wall_tokens_m_{args.imgsz}_v1")

    print(
        json.dumps(
            {
                "mode": args.mode,
                "model": "YOLO-World-M V2 (Ultralytics)",
                "weights": str(weights),
                "data": str(data_yaml),
                "imgsz": args.imgsz,
                "batch": args.batch,
                "epochs": epochs,
                "device": args.device,
                "rectangular_batches": not args.square_padding,
                "output": str(project / name),
            },
            indent=2,
        ),
        flush=True,
    )

    model = YOLOWorld(str(weights))
    model.train(
        trainer=WorldTrainer if args.square_padding else RectWorldTrainer,
        data=str(data_yaml),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=name,
        exist_ok=True,
        resume=args.resume,
        pretrained=True,
        amp=True,
        cache=False,
        seed=42,
        deterministic=True,
        patience=8,
        cos_lr=True,
        plots=True,
        save=True,
        save_period=5 if not smoke else -1,
        verbose=True,
        # Preserve histogram-channel and metric geometry semantics.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
    )


if __name__ == "__main__":
    main()
