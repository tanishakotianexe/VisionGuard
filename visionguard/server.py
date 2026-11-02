"""VisionGuard AI — Streamlit application.

Run from this folder with:
    streamlit run app.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from rapidfuzz import fuzz

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "image_detector.keras"
PRODUCTS_PATH = BASE_DIR / "products.csv"
EVALUATION_DIR = BASE_DIR / "static" / "evaluation"
IMAGE_SIZE = (224, 224)
REQUIRED_COLUMNS = {"product_id", "product_name", "brand", "net_quantity"}

st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the VisionGuard model and OCR engine...")
def load_engines(model_path: str):
    """Load heavyweight resources once per Streamlit process."""
    import tensorflow as tf
    import easyocr

    model = tf.keras.models.load_model(model_path)
    reader = easyocr.Reader(["en"], gpu=bool(tf.config.list_physical_devices("GPU")))
    return model, reader


def normalize(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def match_products(text: str, frame: pd.DataFrame) -> list[dict]:
    query = normalize(text)
    results = []
    for _, row in frame.iterrows():
        name_score = fuzz.partial_ratio(normalize(row["product_name"]), query) if query else 0
        brand_score = fuzz.partial_ratio(normalize(row["brand"]), query) if query else 0
        qty_score = fuzz.partial_ratio(normalize(row["net_quantity"]), query) if query else 0
        results.append({
            "Product ID": str(row["product_id"]),
            "Product": str(row["product_name"]),
            "Brand": str(row["brand"]),
            "Net quantity": str(row["net_quantity"]),
            "Name": round(name_score, 1),
            "Brand score": round(brand_score, 1),
            "Quantity": round(qty_score, 1),
            "Overall match": round(.60 * name_score + .25 * brand_score + .15 * qty_score, 1),
        })
    return sorted(results, key=lambda item: item["Overall match"], reverse=True)


def load_products(uploaded_csv) -> pd.DataFrame:
    if uploaded_csv is not None:
        frame = pd.read_csv(uploaded_csv)
    elif PRODUCTS_PATH.exists():
        frame = pd.read_csv(PRODUCTS_PATH)
    else:
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("Product CSV is missing required columns: " + ", ".join(sorted(missing)))
    return frame.fillna("")


def draw_ocr_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    line_width = max(2, round(min(result.size) / 250))
    for item in detections:
        points = [tuple(map(int, point)) for point in item["box"]]
        if points:
            draw.polygon(points, outline=(35, 190, 145), width=line_width)
    return result


def show_evaluation():
    st.header("Model performance")
    metrics_path = EVALUATION_DIR / "metrics.json"
    if not metrics_path.exists():
        st.info("Saved evaluation metrics were not found. The image analysis feature can still run.")
        return
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f'{metrics["accuracy"] * 100:.2f}%')
        c2.metric("ROC-AUC", f'{metrics["roc_auc"]:.4f}')
        c3.metric("Test images", f'{metrics["test_count"]:,}')
        c4.metric("Correct predictions", f'{metrics["correct"]:,}')

        class_metrics = metrics.get("classes", {})
        rows = []
        for class_name, values in class_metrics.items():
            rows.append({
                "Class": class_name,
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "F1-score": values.get("f1"),
                "Support": values.get("support"),
            })
        if rows:
            st.subheader("Classification report")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Confusion matrix")
        cm = metrics.get("confusion_matrix")
        if cm:
            st.dataframe(
                pd.DataFrame(cm, index=["Actual FAKE", "Actual REAL"],
                             columns=["Predicted FAKE", "Predicted REAL"]),
                use_container_width=True,
            )
        for filename, title in [
            ("confusion_matrix.png", "Confusion matrix plot"),
            ("roc_curve.png", "ROC curve"),
            ("precision_recall_curve.png", "Precision–recall curve"),
        ]:
            path = EVALUATION_DIR / filename
            if path.exists():
                st.subheader(title)
                st.image(str(path), use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not read the saved evaluation data: {exc}")
    st.caption(
        "These are held-out CIFAKE evaluation results, not a guarantee of performance "
        "on product photographs. A model score is not proof of authenticity."
    )


def analyze_image(image: Image.Image, model, reader, products: pd.DataFrame):
    start = time.time()
    resized = image.resize(IMAGE_SIZE)
    array = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    prediction = np.asarray(model.predict(array, verbose=0)).reshape(-1)
    if prediction.size == 0:
        raise ValueError("The model returned an empty prediction.")
    # The supplied model metadata maps output 1 to REAL.
    p_real = float(np.clip(prediction[0], 0.0, 1.0))
    ocr_results = reader.readtext(np.asarray(image.convert("RGB")))
    detections = []
    for box, detected_text, confidence in ocr_results:
        detections.append({
            "text": str(detected_text),
            "confidence": round(float(confidence) * 100, 1),
            "box": [[int(x), int(y)] for x, y in box],
        })
    extracted_text = " ".join(item["text"] for item in detections)
    matches = match_products(extracted_text, products) if extracted_text.strip() else []
    return {
        "p_real": p_real,
        "detections": detections,
        "matches": matches,
        "elapsed": time.time() - start,
    }


st.title("🔍 VisionGuard AI")
st.write(
    "Analyze an image for real-vs-AI-generated signals, extract visible label text, "
    "and compare that text with a product catalogue."
)
st.warning(
    "Experimental decision-support tool only. The CIFAKE model is not trained to "
    "prove product authenticity or detect every form of image manipulation."
)

with st.sidebar:
    st.header("Image and product data")
    uploaded_image = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "webp"]
    )
    uploaded_csv = st.file_uploader(
        "Optional product catalogue (CSV)", type=["csv"],
        help="Required columns: product_id, product_name, brand, net_quantity",
    )
    show_boxes = st.checkbox("Show OCR detection boxes", value=True)
    st.caption("The default catalogue is products.csv in the project folder.")

tab_analyze, tab_metrics, tab_about = st.tabs(
    ["Image analysis", "Model performance", "About"]
)

with tab_analyze:
    if uploaded_image is None:
        st.info("Upload a PNG, JPG, JPEG or WEBP image using the sidebar to begin.")
    else:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            left, right = st.columns([1, 1])
            with left:
                st.subheader("Uploaded image")
                st.image(image, use_container_width=True)
            with right:
                st.subheader("Analysis")
                if st.button("Analyze image", type="primary", use_container_width=True):
                    if not MODEL_PATH.exists():
                        st.error(
                            f"Model file not found: {MODEL_PATH}. "
                            "Make sure models/image_detector.keras is present."
                        )
                    else:
                        try:
                            catalogue = load_products(uploaded_csv)
                            model, reader = load_engines(str(MODEL_PATH))
                            with st.spinner("Classifying image and reading text..."):
                                result = analyze_image(image, model, reader, catalogue)
                            st.session_state["visionguard_result"] = result
                            st.session_state["visionguard_image"] = image
                        except Exception as exc:
                            st.error(f"Analysis failed: {exc}")

            result = st.session_state.get("visionguard_result")
            if result is not None:
                st.divider()
                p_real = result["p_real"]
                is_real = p_real >= 0.5
                confidence = max(p_real, 1 - p_real)
                a, b, c = st.columns(3)
                a.metric("Prediction", "Camera-captured / REAL" if is_real else "AI-generated / FAKE")
                b.metric("Confidence", f"{confidence * 100:.1f}%")
                c.metric("Estimated REAL probability", f"{p_real * 100:.1f}%")
                st.progress(confidence, text="Confidence in the predicted class")

                if show_boxes and result["detections"]:
                    st.subheader("Detected text regions")
                    st.image(draw_ocr_boxes(image, result["detections"]), use_container_width=True)

                st.subheader("Text found (OCR)")
                if result["detections"]:
                    ocr_frame = pd.DataFrame([
                        {"Text": d["text"], "OCR confidence": f'{d["confidence"]:.1f}%'}
                        for d in result["detections"]
                    ])
                    st.dataframe(ocr_frame, use_container_width=True, hide_index=True)
                    st.text_area(
                        "Extracted text",
                        value="\n".join(d["text"] for d in result["detections"]),
                        height=120,
                    )
                else:
                    st.info("No readable text was detected.")

                st.subheader("Closest product matches")
                if result["matches"]:
                    st.dataframe(
                        pd.DataFrame(result["matches"]).head(10),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No product matches were found because no readable text was detected.")
                st.caption(f'Analysis completed in {result["elapsed"]:.2f} seconds.')
                st.caption("Product similarity compares printed text only; it does not verify genuineness.")
        except Exception as exc:
            st.error(f"Could not open the uploaded image: {exc}")

with tab_metrics:
    show_evaluation()

with tab_about:
    st.header("How VisionGuard works")
    st.markdown(
        """
        1. **Classify:** the saved TensorFlow model estimates whether the image resembles
           the REAL class or the AI-generated class in its training data.
        2. **Read:** EasyOCR extracts visible text and its location in the image.
        3. **Match:** RapidFuzz compares extracted text with product name, brand and
           net quantity in the reference CSV.

        **Important:** CIFAKE is a general real-versus-synthetic image dataset, not a
        product-authenticity dataset. Lighting, compression, camera type and unseen
        generators can affect predictions. Validate with representative product images
        before using this system in a real-world workflow.
        """
    )
