import os
import shutil
import random
from pathlib import Path

# Paths
ROOT = Path(r"c:\Users\lipib\Downloads\hackathon\NEU-DET")
ORIG_TRAIN_DIR = ROOT / "train" / "images"
ORIG_VAL_DIR = ROOT / "validation" / "images"

# New paths
DATASET_DIR = ROOT / "inspectai_pipeline" / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# Fix seed
random.seed(42)

def split_data():
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASSES:
            (split / cls).mkdir(parents=True, exist_ok=True)
            
    for cls in CLASSES:
        # Collect all images
        images = []
        
        train_cls_dir = ORIG_TRAIN_DIR / cls
        if train_cls_dir.exists():
            images.extend(list(train_cls_dir.glob("*.jpg")))
            
        val_cls_dir = ORIG_VAL_DIR / cls
        if val_cls_dir.exists():
            images.extend(list(val_cls_dir.glob("*.jpg")))
            
        # Shuffle deterministically
        images = sorted(images) # Sort first for true determinism regardless of filesystem
        random.shuffle(images)
        
        # Split 70/15/15
        total = len(images) # Should be 300
        n_train = int(total * 0.70) # 210
        n_val = int(total * 0.15) # 45
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        # Copy
        for img in train_imgs:
            shutil.copy(img, TRAIN_DIR / cls / img.name)
        for img in val_imgs:
            shutil.copy(img, VAL_DIR / cls / img.name)
        for img in test_imgs:
            shutil.copy(img, TEST_DIR / cls / img.name)
            
        print(f"Class '{cls}': {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
        
if __name__ == "__main__":
    split_data()
    print("Dataset split complete.")
