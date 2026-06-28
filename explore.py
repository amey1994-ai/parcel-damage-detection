import os
import glob
from collections import Counter

base = "Damaged-Package-Detection-2"

for root, dirs, files in os.walk(base):
    level = root.replace(base, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 3:
        for f in files[:3]:
            print(f"{indent}  {f}")


import os

base = "Damaged-Package-Detection-2"

for split in ["train", "valid", "test"]:
    damaged = len(os.listdir(f"{base}/{split}/damaged"))
    intact = len(os.listdir(f"{base}/{split}/intact"))
    total = damaged + intact
    print(f"{split}: {total} total | damaged={damaged} | intact={intact}")