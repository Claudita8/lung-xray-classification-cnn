from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config,
        },
        path,
    )
