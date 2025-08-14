# agent_tools.py
import requests

BASE = "http://localhost:8000"

def tool_inventory_lookup(sku):
    r = requests.get(f"{BASE}/api/inventory/{sku}")
    return r.json()

def tool_predict_from_image(image_bytes):
    files = {"file": ("img.jpg", image_bytes, "image/jpeg")}
    r = requests.post(f"{BASE}/predict", files=files)
    return r.json()