# VisionGuard AI — Streamlit

Classifies an image as camera-captured vs AI-generated (EfficientNetB0 on CIFAKE),
reads label text with EasyOCR, and fuzzy-matches it against a product CSV.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10–3.12
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder's contents to a GitHub repo (the 19 MB `models/image_detector.keras` is included, under GitHub's 100 MB limit).
2. On https://share.streamlit.io choose **New app**, pick the repo, main file `app.py`.
3. In Advanced settings select Python 3.11 or 3.12. Deploy.

Notes: the first analysis downloads EasyOCR weights (~100 MB) and can be slow. If the
free tier runs out of memory, the app still shows the image score (OCR is optional);
for more headroom use Hugging Face Spaces or a small VM.

## Files
- `app.py` — Streamlit app
- `models/image_detector.keras`, `models/metadata.json` — trained model (Keras 3.13)
- `products.csv` — fictional sample products (required columns: product_id, product_name, brand, net_quantity)
- `static/evaluation/` — saved evaluation charts and metrics (shown in the Performance tab)
- `evaluate_model.py` — optional re-evaluation script
- `legacy_flask/` — the original Flask server and web UI (not used by Streamlit)

Limitations: the score is an experimental signal, not proof of authenticity; CIFAKE is not a product dataset.
