import torch
from src.utils.dataloaders import create_dataloaders

if __name__ == "__main__":
    train_loader, val_loader, _ = create_dataloaders(img_size=224, batch_size=8, num_workers=0)

    images, labels = next(iter(train_loader))
    print("Batch images:", images.shape)
    print("Batch labels:", labels.shape)
    print("Labels:", labels.tolist())

    print("CUDA available:", torch.cuda.is_available())
