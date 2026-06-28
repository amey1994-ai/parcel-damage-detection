import os
import shutil

datasets = [
    "Damaged-Package-Detection-2",
    "Damaged-Intact-Box-Dataset-1"
]

output = "master-dataset"

for split in ["train", "valid", "test"]:
    for cls in ["damaged", "intact"]:
        os.makedirs(f"{output}/{split}/{cls}", exist_ok=True)

for ds in datasets:
    for split in ["train", "valid", "test"]:
        for cls in ["damaged", "intact"]:
            src = f"{ds}/{split}/{cls}"
            dst = f"{output}/{split}/{cls}"
            if not os.path.exists(src):
                continue
            for img in os.listdir(src):
                src_file = f"{src}/{img}"
                dst_file = f"{dst}/{ds[:3]}_{img}"
                shutil.copy2(src_file, dst_file)

for split in ["train", "valid", "test"]:
    damaged = len(os.listdir(f"{output}/{split}/damaged"))
    intact = len(os.listdir(f"{output}/{split}/intact"))
    print(f"{split}: {damaged+intact} total | damaged={damaged} | intact={intact}")