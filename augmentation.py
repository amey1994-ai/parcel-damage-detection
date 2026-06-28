import os
import shutil
from PIL import Image, ImageEnhance, ImageFilter

src = "master-dataset/train"
dst = "augmented-dataset/train"


for split in ["valid", "test"]:
    for cls in ["damaged", "intact"]:
        os.makedirs(f"augmented-dataset/{split}/{cls}", exist_ok=True)
        s = f"master-dataset/{split}/{cls}"  # ← fixed
        for img in os.listdir(s):
            shutil.copy2(f"{s}/{img}", f"augmented-dataset/{split}/{cls}/{img}")

for cls in ["damaged", "intact"]:
    os.makedirs(f"{dst}/{cls}", exist_ok=True)
    src_cls = f"{src}/{cls}"
    
    for img_name in os.listdir(src_cls):
        img_path = f"{src_cls}/{img_name}"
        img = Image.open(img_path).convert("RGB")
        base = img_name.replace(".jpg","").replace(".png","").replace(".jpeg","")
        
        img.save(f"{dst}/{cls}/{base}_orig.jpg")
        img.transpose(Image.FLIP_LEFT_RIGHT).save(f"{dst}/{cls}/{base}_flip.jpg")
        img.rotate(15).save(f"{dst}/{cls}/{base}_rot15.jpg")
        img.rotate(-15).save(f"{dst}/{cls}/{base}_rot-15.jpg")
        ImageEnhance.Brightness(img).enhance(1.4).save(f"{dst}/{cls}/{base}_bright.jpg")
        ImageEnhance.Brightness(img).enhance(0.6).save(f"{dst}/{cls}/{base}_dark.jpg")
        img.filter(ImageFilter.GaussianBlur(2)).save(f"{dst}/{cls}/{base}_blur.jpg")
        ImageEnhance.Contrast(img).enhance(1.5).save(f"{dst}/{cls}/{base}_contrast.jpg")

for split in ["train", "valid", "test"]:
    for cls in ["damaged", "intact"]:
        count = len(os.listdir(f"augmented-dataset/{split}/{cls}"))
        print(f"{split}/{cls}: {count} images")