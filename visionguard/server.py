"""VisionGuard AI — Flask backend. Serves the website and runs your trained model + OCR."""
import io, os, re, time, json
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from rapidfuzz import fuzz
from pathlib import Path

IMG_SIZE = (224, 224)
MODEL_PATH = os.environ.get("MODEL_PATH", "models/image_detector.keras")
BASE_DIR = Path(__file__).resolve().parent
PRODUCTS = BASE_DIR / "products.csv"

import tensorflow as tf, easyocr
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print(f"\n*** Model not found at {os.path.abspath(MODEL_PATH)}")
    print("*** The website will start in demo mode. Add image_detector.keras to the models folder and restart.\n")
reader = easyocr.Reader(["en"], gpu=bool(tf.config.list_physical_devices("GPU")))
products = pd.read_csv(PRODUCTS)
app = Flask(__name__, static_folder="static", static_url_path="")

norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def match(text, df):
    q, out = norm(text), []
    for _, r in df.iterrows():
        n, b, qt = (fuzz.partial_ratio(norm(r[k]), q) if q else 0 for k in ("product_name", "brand", "net_quantity"))
        out.append(dict(product_id=str(r["product_id"]), product_name=r["product_name"], brand=r["brand"],
                        net_quantity=r["net_quantity"], name=n, brand_s=b, qty=qt,
                        overall=round(.60 * n + .25 * b + .15 * qt, 2)))
    return sorted(out, key=lambda m: -m["overall"])

@app.get("/")
def index(): return send_from_directory("static", "index.html")

@app.get("/api/health")
def health(): return jsonify(ok=True, model=model is not None)

@app.post("/api/analyze")
def analyze():
    t0 = time.time()
    if model is None: return jsonify(error="Model file missing. Add models/image_detector.keras and restart the server."), 503
    try:
        f = request.files.get("image")
        if not f: return jsonify(error="No image uploaded"), 400
        img = Image.open(f.stream).convert("RGB")
        arr = np.expand_dims(np.asarray(img.resize(IMG_SIZE), dtype=np.float32), 0)
        p_real = float(model.predict(arr, verbose=0)[0][0])
        ocr = [dict(text=str(t), conf=float(c) * 100, box=[[int(x), int(y)] for x, y in b])
               for b, t, c in reader.readtext(np.asarray(img))]
        ref = products
        if request.files.get("csv"):
            ref = pd.read_csv(request.files["csv"])
            need = {"product_id", "product_name", "brand", "net_quantity"}
            if not need.issubset(ref.columns): return jsonify(error=f"CSV needs columns: {sorted(need)}"), 400
        text = " ".join(o["text"] for o in ocr)
        return jsonify(p_real=p_real, width=img.width, height=img.height, ocr=ocr,
                       matches=match(text, ref) if text.strip() else [], ms=int((time.time() - t0) * 1000))
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
