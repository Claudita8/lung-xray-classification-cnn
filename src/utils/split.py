import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.utils.paths import RAW_DIR, PROCESSED_DIR as OUT_DIR
from src.utils.paths import PROJECT_ROOT

OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def find_images(root: Path):
    if not root.exists():
        return []

    exts = (".png",)
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if "mask" in str(p).lower() or "masks" in str(p).lower():
            continue
        files.append(p)
    return files


def collect_images() -> pd.DataFrame:
    samples = []

    class_map = {
        "COVID": "COVID",
        "Normal": "Normal",
        "Lung_Opacity": "Pneumonia",
        "Viral_Pneumonia": "Pneumonia",
    }

    for folder, label in class_map.items():
        folder_path = RAW_DIR / folder
        img_files = find_images(folder_path)

        for img_path in img_files:
            samples.append({
                "path": str(img_path.relative_to(PROJECT_ROOT)),
                "label": label
            })

    df_local = pd.DataFrame(samples)

    print("RAW_DIR:", RAW_DIR, "exists:", RAW_DIR.exists())
    print("Folders in RAW_DIR:", [p.name for p in RAW_DIR.iterdir()] if RAW_DIR.exists() else "N/A")

    if df_local.empty:
        raise RuntimeError(f"No images found under {RAW_DIR}. Check dataset location and extensions.")

    return df_local


def split_dataset(df_in: pd.DataFrame):
    if "label" not in df_in.columns:
        raise ValueError(f"Expected columns ['path','label'], got: {list(df_in.columns)}")

    train_df, temp_df = train_test_split(
        df_in,
        test_size=0.30,
        stratify=df_in["label"],
        random_state=SEED,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED,
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    df = collect_images()
    print("\nFound images per class:")
    print(df["label"].value_counts())

    train_df, val_df, test_df = split_dataset(df)

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    print("\nSplit completed:")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")
    print("\nSaved to:", OUT_DIR)
