import os

print("=" * 50)
print("MASTER DATASET")
print("=" * 50)
for split in ["train", "valid", "test"]:
    for cls in ["damaged", "intact"]:
        path = f"master-dataset/{split}/{cls}"
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f"{split}/{cls}: {count} images")

print()
print("=" * 50)
print("AUGMENTED DATASET")
print("=" * 50)
for split in ["train", "valid", "test"]:
    for cls in ["damaged", "intact"]:
        path = f"augmented-dataset/{split}/{cls}"
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f"{split}/{cls}: {count} images")

print()
print("=" * 50)
print("DIFFERENCE (augmented - master)")
print("=" * 50)
for split in ["train", "valid", "test"]:
    for cls in ["damaged", "intact"]:
        m = len(os.listdir(f"master-dataset/{split}/{cls}"))
        a = len(os.listdir(f"augmented-dataset/{split}/{cls}"))
        diff = a - m
        print(f"{split}/{cls}: +{diff} new images added")