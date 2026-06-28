import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import random
import numpy as np

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True  # ← change to True
torch.backends.cudnn.benchmark = False     # ← change to False

DATA_DIR    = "augmented-dataset"
BATCH_SIZE  = 16
EPOCHS      = 50
LR          = 0.0001
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH   = "best_model.pth"
PATIENCE    = 7

print(f"Using device: {DEVICE}")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/valid", transform=val_transform)
test_ds  = datasets.ImageFolder(f"{DATA_DIR}/test",  transform=val_transform)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=0, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                      num_workers=0, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                      num_workers=0, pin_memory=True)

print(f"Classes : {train_ds.classes}")
print(f"Train   : {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# Model
model = models.efficientnet_b5(weights='DEFAULT')

# Freeze first 5 blocks only (unfreeze one more vs train5)
for param in model.features[:5].parameters():
    param.requires_grad = False

# Dropout
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.classifier[1].in_features, 2)
)
model = model.to(DEVICE)

torch.cuda.empty_cache()

# Class weights — penalize missing damaged 1.5x more
class_weights = torch.tensor([1.5, 1.0]).to(DEVICE)
criterion     = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

best_val_acc = 0.0
no_improve   = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss, train_correct = 0, 0
    for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    torch.cuda.empty_cache()

    # Validate
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs    = model(imgs)
            val_loss  += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    torch.cuda.empty_cache()

    train_acc = train_correct / len(train_ds) * 100
    val_acc   = val_correct   / len(val_ds)   * 100
    gap       = train_acc - val_acc

    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_loss:.3f} Acc={train_acc:.1f}% | "
          f"Val Loss={val_loss:.3f} Acc={val_acc:.1f}% | "
          f"Gap={gap:.1f}%")

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve   = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ Best model saved ({val_acc:.1f}%)")
    else:
        no_improve += 1
        print(f"  ⏳ No improvement for {no_improve}/{PATIENCE} epochs")
        if no_improve >= PATIENCE:
            print(f"\n🛑 Early stopping — best val acc: {best_val_acc:.1f}%")
            break

print("\n" + "="*50)
print("FINAL TEST EVALUATION")
print("="*50)

model.load_state_dict(torch.load(SAVE_PATH))
model.eval()

test_correct = 0
damaged_correct, damaged_total = 0, 0
intact_correct,  intact_total  = 0, 0

with torch.no_grad():
    for imgs, labels in test_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(1)
        test_correct += (preds == labels).sum().item()

        # Per class
        damaged_mask = labels == 0
        intact_mask  = labels == 1
        damaged_correct += (preds[damaged_mask] == 0).sum().item()
        damaged_total   += damaged_mask.sum().item()
        intact_correct  += (preds[intact_mask]  == 1).sum().item()
        intact_total    += intact_mask.sum().item()

print(f"Overall  Test Accuracy : {test_correct/len(test_ds)*100:.1f}%")
print(f"Damaged  Accuracy      : {damaged_correct/damaged_total*100:.1f}% ({damaged_correct}/{damaged_total})")
print(f"Intact   Accuracy      : {intact_correct/intact_total*100:.1f}% ({intact_correct}/{intact_total})")
print(f"Best Val Accuracy      : {best_val_acc:.1f}%")


