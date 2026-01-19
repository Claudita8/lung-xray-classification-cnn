import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.dataloaders import create_dataloaders
from src.models.simple_cnn import SimpleCNN
from src.utils.metrics import compute_metrics
from src.utils.checkpoints import save_checkpoint


def get_device(config):
    # Auto device selection (works even if config says cuda but you don't have it)
    if config["training"]["device"].lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    val_loss = running_loss / len(loader.dataset)
    return val_loss, metrics


def main():
    config = load_config("configs/cnn_baseline.yaml")
    set_seed(config["split"]["seed"])

    device = get_device(config)
    print("Using device:", device)

    # W&B init
    wandb.init(
        project=config["project"]["name"],
        name=config["project"]["experiment"],
        config=config,
    )

    train_loader, val_loader, _ = create_dataloaders(
        img_size=config["data"]["img_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    model = SimpleCNN(num_classes=config["data"]["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_acc = 0.0

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        val_acc = val_metrics["accuracy"]

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1_macro": val_metrics["f1_macro"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
        })

        print(
            f"Epoch {epoch+1}/{config['training']['epochs']} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = "checkpoints/best_simplecnn.pth"
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_acc, config)
            wandb.save(ckpt_path)

    wandb.finish()
    print("Training done. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
