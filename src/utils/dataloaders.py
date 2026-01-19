from torch.utils.data import DataLoader
from src.utils.dataset import XRayDataset
from src.utils.transforms import get_train_transforms, get_val_transforms
from src.utils.paths import PROCESSED_DIR


def create_dataloaders(img_size: int, batch_size: int, num_workers: int = 4):
    train_ds = XRayDataset(str(PROCESSED_DIR / "train.csv"), transform=get_train_transforms(img_size))
    val_ds = XRayDataset(str(PROCESSED_DIR / "val.csv"), transform=get_val_transforms(img_size))
    test_ds = XRayDataset(str(PROCESSED_DIR / "test.csv"), transform=get_val_transforms(img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
