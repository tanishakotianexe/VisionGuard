"""VisionGuard AI - Streamlit app.
Scores an image as camera-captured vs AI-generated, reads label text (OCR),
and fuzzy-matches the text against a product reference CSV.
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from rapidfuzz import fuzz

BASE = Path(__file__).parent
IMG_SIZE = (224, 224)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE / "models" / "image_detector.keras"))
PRODUCTS_PATH = Path(os.environ.get("PRODUCTS_CSV", BASE / "products.csv"))
EVAL_DIR = BASE / "static" / "evaluation"
REQUIRED_COLS = {"product_id", "product_name", "brand", "net_quantity"}
MAX_SIDE = 1600  # downscale big uploads so OCR stays fast and memory-safe

st.set_page_config(page_title="VisionGuard AI", page_icon="🔍", layout="wide")

st.markdown(
    """
<style>
.block-container{padding-top:2rem;max-width:1180px}
.vg-tag{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;font-weight:700}
.vg-card{border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:14px 16px;margin-bottom:10px}
.vg-sub{font-size:13px;opacity:.7}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- cached resources ----------
@st.cache_resource(show_spinner="Loading image model…")
def load_model():
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf

    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


@st.cache_resource(show_spinner="Loading OCR engine (first run downloads models)…")
def load_ocr():
    try:
        import easyocr

        return easyocr.Reader(["en"], gpu=False)
    except Exception as e:  # keep the app usable if OCR can't load
        return e


@st.cache_data
def load_default_products():
    return pd.read_csv(PRODUCTS_PATH)


@st.cache_data
def load_metrics():
    p = EVAL_DIR / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------- logic ----------
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def match_products(text, df):
    q, out = norm(text), []
    for _, r in df.iterrows():
        n, b, qt = (
            fuzz.partial_ratio(norm(r[k]), q) if q else 0
            for k in ("product_name", "brand", "net_quantity")
        )
        out.append(
            dict(
                product_name=r["product_name"],
                brand=r["brand"],
                net_quantity=r["net_quantity"],
                name=n,
                brand_s=b,
                qty=qt,
                overall=round(0.60 * n + 0.25 * b + 0.15 * qt, 2),
            )
        )
    return sorted(out, key=lambda m: -m["overall"])


def predict_real(model, img):
    arr = np.expand_dims(np.asarray(img.resize(IMG_SIZE), dtype=np.float32), 0)
    return float(model.predict(arr, verbose=0)[0][0])  # P(real); label 1 = real


def run_ocr(reader, img):
    results = reader.readtext(np.asarray(img))
    return [
        dict(text=str(t), conf=float(c) * 100, box=[(int(x), int(y)) for x, y in b])
        for b, t, c in results
    ]


def draw_boxes(img, ocr):
    out = img.copy()
    d = ImageDraw.Draw(out, "RGBA")
    w = max(2, img.width // 260)
    for o in ocr:
        pts = o["box"]
        d.polygon(pts, fill=(120, 220, 190, 40))
        d.line(pts + [pts[0]], fill=(120, 220, 190, 255), width=w)
    return out


# ---------- UI ----------
model = load_model()

head_l, head_r = st.columns([4, 1])
head_l.title("🔍 VisionGuard")
head_r.markdown(
    f"<div style='text-align:right;padding-top:1.2rem'>{'🟢 Live model loaded' if model is not None else '🟠 Model missing'}</div>",
    unsafe_allow_html=True,
)

tab_check, tab_perf, tab_next = st.tabs(["Analyze image", "Model performance", "Enhancements & conclusion"])

# ===== Analyze =====
with tab_check:
    st.subheader("Is this photo real? Then read what's on it.")
    st.write(
        "Upload a product or listing image. VisionGuard scores it as camera-captured or "
        "AI-generated, reads the label text, and compares it with your product list."
    )
    if model is None:
        st.error(
            f"Model file not found at `{MODEL_PATH}`. Add `models/image_detector.keras` to the repo."
        )

    col_up, col_csv = st.columns([2, 1])
    up = col_up.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
    csv_up = col_csv.file_uploader("Optional product list (CSV)", type=["csv"],
                                   help="Columns: product_id, product_name, brand, net_quantity")

    if up is not None:
        try:
            img = Image.open(up).convert("RGB")
        except Exception:
            st.error("Could not read that file as an image.")
            st.stop()
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE))

        ref = None
        try:
            ref = pd.read_csv(csv_up) if csv_up else load_default_products()
        except Exception as e:
            st.error(f"Could not read product CSV: {e}")
        if ref is not None and not REQUIRED_COLS.issubset(ref.columns):
            st.error(f"CSV needs columns: {sorted(REQUIRED_COLS)}")
            ref = None

        if st.button("Analyze image", type="primary", disabled=(model is None or ref is None)):
            t0 = time.time()
            with st.spinner("Analyzing…"):
                p_real = predict_real(model, img)
                reader = load_ocr()
                ocr = [] if isinstance(reader, Exception) else run_ocr(reader, img)
            ms = int((time.time() - t0) * 1000)
            st.session_state["result"] = dict(
                p_real=p_real, ocr=ocr, ms=ms, size=img.size, kb=len(up.getvalue()) / 1024,
                ocr_error=str(reader) if isinstance(reader, Exception) else None,
                matches=match_products(" ".join(o["text"] for o in ocr), ref) if ocr else [],
                image=img,
            )

        r = st.session_state.get("result")
        if r:
            if r["ocr_error"]:
                st.warning(f"OCR unavailable on this server ({r['ocr_error']}). Showing image score only.")
            left, right = st.columns([1.1, 1])
            with left:
                show = st.checkbox("Show detected text regions", value=True)
                st.image(draw_boxes(r["image"], r["ocr"]) if show else r["image"], use_container_width=True)
                st.caption(f"{r['size'][0]}×{r['size'][1]} · {r['kb']:.0f} KB · {r['ms']/1000:.1f} s")
            with right:
                p = r["p_real"]
                real = p >= 0.5
                conf = max(p, 1 - p) * 100
                color = "#2b7a66" if real else "#b3402e"
                st.markdown(
                    f"<h2 style='color:{color};margin-bottom:0'>"
                    f"{'Looks like a real photo' if real else 'Looks AI-generated'}</h2>"
                    f"<div class='vg-sub'>A model estimate, not proof that a product or seller is genuine.</div>",
                    unsafe_allow_html=True,
                )
                st.metric("Confidence", f"{conf:.0f}%")
                st.write(f"Camera-captured — {p*100:.1f}%")
                st.progress(min(max(p, 0.0), 1.0))
                st.write(f"AI-generated — {(1-p)*100:.1f}%")
                st.progress(min(max(1 - p, 0.0), 1.0))

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Text found")
                txt = "\n".join(o["text"] for o in r["ocr"]) or "No readable text detected."
                st.code(txt, language=None)
                for o in r["ocr"][:8]:
                    st.caption(f"{o['text'][:36]} — {o['conf']:.0f}%")
                    st.progress(min(o["conf"] / 100, 1.0))
            with c2:
                st.markdown("#### Closest products")
                if not r["matches"]:
                    st.caption("No text to match.")
                for m in r["matches"][:3]:
                    s = m["overall"]
                    c = "#2b7a66" if s >= 75 else "#b57d22" if s >= 50 else "#b3402e"
                    t = "Strong" if s >= 75 else "Partial" if s >= 50 else "Weak"
                    st.markdown(
                        f"<div class='vg-card'><b>{m['product_name']}</b> "
                        f"<span class='vg-tag' style='color:{c};background:{c}22'>{t} · {s:.0f}%</span>"
                        f"<div class='vg-sub'>{m['brand']} · {m['net_quantity']}</div>"
                        f"<div class='vg-sub'>Name {m['name']:.0f}% · Brand {m['brand_s']:.0f}% · Quantity {m['qty']:.0f}%</div></div>",
                        unsafe_allow_html=True,
                    )
                st.caption("Similarity of printed text only. It does not verify authenticity.")

    with st.expander("How a check works"):
        st.markdown(
            "1. **Classify** — an EfficientNetB0 model trained on CIFAKE scores the image as real or synthetic.\n"
            "2. **Read** — EasyOCR finds every line of text and marks where it sits in the image.\n"
            "3. **Match** — fuzzy matching compares the text with product name, brand and quantity.\n\n"
            "CIFAKE is a general image dataset, not a product dataset. Treat scores as one signal among several."
        )

# ===== Performance =====
with tab_perf:
    st.subheader("Performance & statistics")
    st.caption("Measured on the held-out CIFAKE test set. These are model evaluation results, not per-image confidence scores.")
    m = load_metrics()
    if m:
        pct = lambda v: f"{v*100:.1f}%"
        cols = st.columns(3)
        cols[0].metric("Test accuracy", pct(m["accuracy"]))
        cols[1].metric("Precision · REAL", pct(m["classes"]["REAL"]["precision"]))
        cols[2].metric("Recall · REAL", pct(m["classes"]["REAL"]["recall"]))
        cols = st.columns(3)
        cols[0].metric("F1-score · REAL", pct(m["classes"]["REAL"]["f1"]))
        cols[1].metric("ROC-AUC", f"{m['roc_auc']:.4f}")
        cols[2].metric("Test images", f"{m['test_count']:,}")
    else:
        st.info("Saved evaluation metrics not found.")
    g1, g2 = st.columns(2)
    for col, (title, fn) in zip(
        [g1, g2, g1, g2],
        [("Confusion matrix", "confusion_matrix.png"), ("ROC curve", "roc_curve.png"),
         ("Precision-recall curve", "precision_recall_curve.png")] + [(None, None)],
    ):
        if title and (EVAL_DIR / fn).exists():
            col.markdown(f"**{title}**")
            col.image(str(EVAL_DIR / fn), use_container_width=True)
    rep = EVAL_DIR / "evaluation_report.txt"
    if rep.exists():
        g2.markdown("**Classification report**")
        g2.code(rep.read_text(encoding="utf-8"), language=None)
    st.caption("CIFAKE is a general real-vs-synthetic image dataset, not a dedicated product-authenticity benchmark.")

# ===== Roadmap =====
with tab_next:
    st.subheader("Further enhancements")
    items = [
        ("01 · Product-specific training", "Build a labelled dataset of genuine and manipulated product photographs, then fine-tune and validate the detector on those images."),
        ("02 · Explainable predictions", "Add visual explanations such as Grad-CAM heatmaps so users can see which regions influenced a prediction."),
        ("03 · Stronger evaluation", "Test across cameras, lighting, compression levels and AI generators; report class-wise metrics and calibration."),
        ("04 · Safer product verification", "Combine image signals with OCR, barcode/QR checks, trusted product records and provenance data."),
        ("05 · Production readiness", "Add secure uploads, size limits, privacy-aware retention, monitoring, model versioning and a stable hosted API."),
        ("06 · Broader model coverage", "Evaluate newer synthetic-image detectors and ensembles, checking performance on unseen datasets."),
    ]
    a, b = st.columns(2)
    for i, (t, d) in enumerate(items):
        (a if i % 2 == 0 else b).markdown(f"**{t}**\n\n{d}")
    st.subheader("Conclusion")
    st.write(
        "VisionGuard brings together image classification, text extraction and product-catalog matching in one workflow. "
        "Its model provides an experimental signal for distinguishing real from AI-generated images, while OCR and fuzzy "
        "matching compare visible label text with reference product data. It is a decision-support prototype, not a "
        "guarantee of product authenticity."
    )

st.caption("VisionGuard AI · Experimental research tool")
