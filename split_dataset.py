import os
print("✅ Script started, current directory:", os.getcwd())

import random
import shutil
from pathlib import Path

import shutil
from pathlib import Path

# 👇 Change this to your real dataset folder
SOURCE_DIR = 'data/dataset-resized'   # folder that has 'cardboard', 'glass', etc.
DEST_DIR = 'dataset'         # output folder
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test

print(f"Looking for dataset in: {Path(SOURCE_DIR).resolve()}")

if not os.path.exists(SOURCE_DIR):
    print(f"❌ ERROR: Folder '{SOURCE_DIR}' not found! Please check the name.")
    exit()

# Check class folders
class_folders = [f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]
if not class_folders:
    print(f"❌ ERROR: No class folders found in '{SOURCE_DIR}'.")
    exit()

print(f"✅ Found class folders: {class_folders}")

# Create destination folders
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

for class_name in class_folders:
    class_path = os.path.join(SOURCE_DIR, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(images) == 0:
        print(f"⚠️ WARNING: No images found in {class_name}")
        continue

    random.shuffle(images)
    n_total = len(images)
    n_train = int(n_total * SPLIT_RATIOS[0])
    n_val = int(n_total * SPLIT_RATIOS[1])

    print(f"\n📁 Splitting '{class_name}' ({n_total} images)")
    print(f" - Train: {n_train}, Val: {n_val}, Test: {n_total - n_train - n_val}")

    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }

    for split, files in splits.items():
        split_dir = os.path.join(DEST_DIR, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for file in files:
            shutil.copy2(os.path.join(class_path, file), os.path.join(split_dir, file))

print("\n✅ Splitting complete! Check the 'dataset' folder.")
