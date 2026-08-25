"""Train a compact CNN to review geometric wall candidates."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class WallCandidateNet(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()

        def block(source: int, target: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(source, target, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(target),
                nn.SiLU(inplace=True),
                nn.Conv2d(target, target, 3, padding=1, bias=False),
                nn.BatchNorm2d(target),
                nn.SiLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-checkpoint", type=Path)
    return parser.parse_args()


def make_loaders(args):
    common = [
        transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        *common,
    ])
    eval_transform = transforms.Compose(common)
    train = datasets.ImageFolder(args.dataset / "train", transform=train_transform)
    val = datasets.ImageFolder(args.dataset / "val", transform=eval_transform)
    test = datasets.ImageFolder(args.dataset / "test", transform=eval_transform)
    if train.class_to_idx != val.class_to_idx or train.class_to_idx != test.class_to_idx:
        raise ValueError("Class mappings differ between dataset splits")
    pin = torch.cuda.is_available()
    return (
        train,
        val,
        test,
        DataLoader(train, args.batch, shuffle=True, num_workers=args.workers, pin_memory=pin),
        DataLoader(val, args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin),
        DataLoader(test, args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin),
    )


def confusion_metrics(confusion: np.ndarray, classes: list[str]) -> dict:
    result = {}
    for index, name in enumerate(classes):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        result[name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(2.0 * precision * recall / max(precision + recall, 1e-12), 6),
            "support": int(confusion[index, :].sum()),
        }
    result["accuracy"] = round(float(np.trace(confusion) / max(confusion.sum(), 1)), 6)
    return result


@torch.inference_mode()
def evaluate(model, loader, device, criterion, class_count: int):
    model.eval()
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    loss_sum = 0.0
    examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += float(criterion(logits, targets)) * len(targets)
        examples += len(targets)
        predictions = torch.argmax(logits, dim=1)
        for truth, prediction in zip(targets.cpu().numpy(), predictions.cpu().numpy()):
            confusion[int(truth), int(prediction)] += 1
    return loss_sum / max(examples, 1), confusion


def render_confusion(confusion: np.ndarray, classes: list[str], output: Path) -> None:
    cell = 180
    canvas = np.full(((len(classes) + 1) * cell, (len(classes) + 1) * cell, 3), 245, np.uint8)
    maximum = max(int(confusion.max()), 1)
    for row, truth in enumerate(classes):
        cv2.putText(canvas, truth, (8, (row + 1) * cell + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        cv2.putText(canvas, truth, ((row + 1) * cell + 8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        for column in range(len(classes)):
            value = int(confusion[row, column])
            intensity = int(round(220 * value / maximum))
            top = (row + 1) * cell
            left = (column + 1) * cell
            cv2.rectangle(canvas, (left, top), (left + cell, top + cell), (255 - intensity, 245 - intensity // 2, 235), -1)
            cv2.putText(canvas, str(value), (left + 55, top + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (15, 15, 15), 2)
    cv2.putText(canvas, "truth / prediction", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    cv2.imwrite(str(output), canvas)


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_set, val_set, test_set, train_loader, val_loader, test_loader = make_loaders(args)
    classes = train_set.classes
    model = WallCandidateNet(len(classes)).to(device)
    if args.initial_checkpoint is not None:
        initial = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        if initial.get("classes") != classes:
            raise ValueError("Initial checkpoint classes do not match this dataset")
        model.load_state_dict(initial["model_state"])
    class_counts = np.bincount([target for _, target in train_set.samples], minlength=len(classes))
    class_weights = np.sqrt(class_counts.sum() / np.maximum(class_counts, 1))
    class_weights /= class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_f1 = -1.0
    stale = 0
    best_path = args.output / "best.pt"
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        examples = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss) * len(targets)
            examples += len(targets)
        scheduler.step()
        val_loss, val_confusion = evaluate(model, val_loader, device, criterion, len(classes))
        metrics = confusion_metrics(val_confusion, classes)
        wall_f1 = metrics.get("wall", {}).get("f1", 0.0)
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss / max(examples, 1), 6),
            "val_loss": round(val_loss, 6),
            "wall_f1": wall_f1,
            "val_accuracy": metrics["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if wall_f1 > best_f1 + 1e-5:
            best_f1 = wall_f1
            stale = 0
            torch.save({
                "schema": "cloud2bim.wall-candidate-classifier.v1",
                "model_state": model.state_dict(),
                "classes": classes,
                "class_to_idx": train_set.class_to_idx,
                "input_size": [args.width, args.height],
                "epoch": epoch,
                "wall_f1": wall_f1,
            }, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_loss, val_confusion = evaluate(model, val_loader, device, criterion, len(classes))
    test_loss, test_confusion = evaluate(model, test_loader, device, criterion, len(classes))
    summary = {
        "schema": "cloud2bim.wall-candidate-training.v1",
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "classes": classes,
        "dataset_counts": {
            "train": len(train_set), "val": len(val_set), "test": len(test_set)
        },
        "class_counts_train": {
            classes[index]: int(value) for index, value in enumerate(class_counts)
        },
        "best_epoch": checkpoint["epoch"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "validation": {
            "loss": round(val_loss, 6),
            "confusion": val_confusion.tolist(),
            "metrics": confusion_metrics(val_confusion, classes),
        },
        "test": {
            "loss": round(test_loss, 6),
            "confusion": test_confusion.tolist(),
            "metrics": confusion_metrics(test_confusion, classes),
        },
        "history": history,
        "checkpoint": str(best_path.resolve()),
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_confusion(test_confusion, classes, args.output / "test_confusion.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
