import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# ─── CONFIG ───────────────────────────────────────
MODEL_PATH = "best_model.pth"
CLASSES    = ["damaged", "intact"]
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────

# Load model once at startup
def load_model():
    model = models.efficientnet_b5(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

model = load_model()
print(f"✅ Model loaded on {DEVICE}")

# Transform for inference
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# FastAPI app
app = FastAPI(
    title="Parcel Damage Detection API",
    description="Upload a box image → get damaged/intact prediction",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {
        "status"  : "running",
        "model"   : "EfficientNet-B5",
        "device"  : str(DEVICE),
        "accuracy": "88.1%"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)
        confidence  = probs.max().item() * 100
        class_idx   = probs.argmax().item()
        prediction  = CLASSES[class_idx]

    # Business logic
    status = "REJECT ❌" if prediction == "damaged" else "ACCEPT ✅"

    return JSONResponse({
        "filename"  : file.filename,
        "prediction": prediction,
        "confidence": f"{confidence:.1f}%",
        "status"    : status,
        "damaged_prob": f"{probs[0][0].item()*100:.1f}%",
        "intact_prob" : f"{probs[0][1].item()*100:.1f}%"
    })

@app.get("/metrics")
def metrics():
    return {
        "model"         : "EfficientNet-B5",
        "dataset"       : "1,581 real images",
        "augmented"     : "9,648 training images",
        "overall_acc"   : "88.1%",
        "damaged_acc"   : "82.3%",
        "intact_acc"    : "91.8%",
        "epochs"        : 9,
        "best_val_acc"  : "85.6%"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)