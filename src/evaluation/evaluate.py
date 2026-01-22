import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

import wandb

from src.models.simple_cnn import SimpleCNN
from src.utils.dataset import XRayDataset
from src.utils.transforms import get_val_transforms
from src.utils.paths import PROCESSED_DIR, CHECKPOINTS_DIR


CLASS_NAMES = ["COVID", "Pneumonia", "Normal"]


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = SimpleCNN(num_classes=3).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    model.eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    y_true, y_pred = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        y_true.append(targets.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "best_simplecnn.pth"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--wandb_project", type=str, default="lung-xray-classification-cnn")
    parser.add_argument("--wandb_run_name", type=str, default="cnn_test_eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_csv = PROCESSED_DIR / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing: {test_csv}. Run split first.")

    test_ds = XRayDataset(str(test_csv), transform=get_val_transforms(args.img_size))
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model(checkpoint_path, device)

    y_true, y_pred = predict(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

    print("\n=== TEST RESULTS ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {f1_macro:.4f}\n")
    print(report)
    print("Confusion Matrix:\n", cm)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    wandb.log({
        "test/accuracy": acc,
        "test/f1_macro": f1_macro,
    })

    wandb.log({
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=CLASS_NAMES
        )
    })

    report_path = Path("test_classification_report.txt")
    report_path.write_text(report)
    wandb.save(str(report_path))

    wandb.finish()
    print("✅ Logged test metrics to W&B.")


if __name__ == "__main__":
    main()
