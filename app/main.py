# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, torch, torchvision.transforms as T, json

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# load model
checkpoint = torch.load("model.pt", map_location="cpu")
classes = checkpoint['classes']
model = ... # load architecture then state_dict (see train.py) 
model.eval()

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...), metadata: str = ""):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, k=3)
    result = {
        "top1_label": classes[topk.indices[0].item()],
        "top3": [{"label": classes[i.item()], "score": float(probs[i].item())} for i in topk.indices],
        "short_description": f"A {classes[topk.indices[0].item()]} (auto-generated)."
    }
    return result

# Mock inventory DB (in-memory)
INVENTORY = {
    "SKU123": {"title":"Stainless Travel Mug", "available": 12, "warehouse":"Toronto - WH1"},
    "SKU456": {"title":"Wireless Earbuds", "available": 0, "warehouse":"Vancouver - WH2"}
}

@app.get("/api/inventory/{sku}")
def inventory_lookup(sku: str):
    return INVENTORY.get(sku, {"error":"not found"})
