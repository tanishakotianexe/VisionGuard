"""VisionGuard AI - Streamlit app (styled to match the original web UI).
Scores an image as camera-captured vs AI-generated, reads label text (OCR),
and fuzzy-matches the text against a product reference CSV.
"""
import base64
import html
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from rapidfuzz import fuzz

BASE = Path(__file__).parent
IMG_SIZE = (224, 224)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE / "models" / "image_detector.keras"))
MODEL_URL = os.environ.get("MODEL_URL", "")
try:  # optional: set MODEL_URL in Streamlit secrets to download the model at startup
    MODEL_URL = MODEL_URL or st.secrets.get("MODEL_URL", "")
except Exception:
    pass
PRODUCTS_PATH = Path(os.environ.get("PRODUCTS_CSV", BASE / "products.csv"))
EVAL_DIR = BASE / "static" / "evaluation"
REQUIRED_COLS = {"product_id", "product_name", "brand", "net_quantity"}
MAX_SIDE = 1600
RING = 364.4

st.set_page_config(page_title="VisionGuard AI — Image verification & label reading",
                   page_icon="🔍", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');
:root{--paper:#ece6db;--card:#f6f2ea;--ink:#1a1511;--muted:#6b6056;--line:#d2c8b8;--real:#2b7a66;--ai:#b3402e;--warn:#b57d22;--r:14px;
--f1:'Bricolage Grotesque',system-ui,sans-serif;--f2:'Figtree',system-ui,sans-serif}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:var(--paper)!important;color:var(--ink);font-family:var(--f2)}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],footer,#MainMenu{display:none!important}
.block-container{max-width:1180px;padding:0 clamp(18px,4vw,40px) 40px!important}
html{scroll-behavior:smooth}
p,li,label,span,div{font-family:var(--f2)}
h1,h2,h3{font-family:var(--f1)!important;color:var(--ink)}
.vg-sub{font-size:13px;color:var(--muted);margin:0}
.vg-top{display:flex;justify-content:space-between;align-items:center;padding:22px 0}
.vg-logo{display:flex;align-items:center;gap:10px;font:800 21px var(--f1);color:var(--ink);letter-spacing:-.4px}
.vg-pill{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600;padding:6px 14px;border:1px solid var(--line);border-radius:99px;background:var(--card)}
.vg-pill i{width:8px;height:8px;border-radius:50%;background:var(--warn);display:inline-block}.vg-pill.live i{background:var(--real)}
.vg-nav{display:flex;gap:18px;font-size:13px;padding-bottom:6px}
.vg-nav a{color:var(--muted)!important;text-decoration:none;border-bottom:1px solid var(--line);padding-bottom:4px}
.vg-nav a:hover{color:var(--ink)!important;border-color:var(--ink)}
.vg-h1{font:800 clamp(38px,5.4vw,66px)/1.02 var(--f1);letter-spacing:-.035em;margin:clamp(20px,4vw,56px) 0 20px;color:var(--ink)}
.vg-lead{font-size:18px;color:var(--muted);max-width:46ch;margin:0 0 28px}

/* buttons */
.st-key-go button,.st-key-clear button{border-radius:99px;font:700 16px var(--f2);padding:14px 28px;min-height:0;transition:transform .15s,background .2s}
.st-key-go button{background:var(--ink)!important;color:var(--paper)!important;border:0}
.st-key-go button:hover:not(:disabled){transform:translateY(-2px);background:#3a2e25!important}
.st-key-go button:disabled{opacity:.35}
.st-key-go button p{color:var(--paper)!important;font-weight:700}
.st-key-clear button{background:none!important;border:1px solid var(--line)!important;color:var(--ink)!important;font-weight:600;font-size:15px}
.st-key-clear button:hover{border-color:var(--ink)!important}
[data-testid="stExpander"]{border:0!important;background:transparent!important}
[data-testid="stExpander"] summary{color:var(--muted);font-size:14px;font-weight:600;padding-left:0}

/* film-strip upload frame */
.st-key-film{position:relative;background:var(--ink);border-radius:10px;padding:34px 16px!important;box-shadow:0 30px 50px -28px rgba(26,21,17,.55);transform:rotate(1.2deg);margin-top:18px}
.st-key-film::before,.st-key-film::after{content:"";position:absolute;left:10px;right:10px;height:10px;background:repeating-linear-gradient(90deg,var(--paper) 0 14px,transparent 14px 30px);border-radius:3px;opacity:.9}
.st-key-film::before{top:12px}.st-key-film::after{bottom:12px}
.st-key-film [data-testid="stFileUploaderDropzone"]{background:#2a211b;border:0;border-radius:4px;color:#cdbfae;min-height:230px;justify-content:center;transition:background .2s}
.st-key-film [data-testid="stFileUploaderDropzone"]:hover{background:#352a22}
.st-key-film [data-testid="stFileUploaderDropzone"] *{color:#cdbfae!important}
.st-key-film [data-testid="stFileUploaderDropzone"] button{background:#f1e8d9;color:var(--ink)!important;border:0;border-radius:99px}
.st-key-film [data-testid="stFileUploaderDropzone"] button *{color:var(--ink)!important}
.st-key-film [data-testid="stFileUploaderFile"] *{color:#cdbfae!important}
.st-key-film [data-testid="stFileUploader"] label{display:none}
.st-key-film img{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:4px;display:block}
.st-key-film [data-testid="stImageContainer"]{margin-bottom:10px}

/* report */
.vg-banner{background:#f3e3bf;border:1px solid #d9bd7c;color:#5a4311;border-radius:var(--r);padding:12px 18px;margin-bottom:20px;font-size:14px}
.st-key-shot,.st-key-panel_text,.st-key-panel_match{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:16px!important}
.st-key-panel_text,.st-key-panel_match{padding:22px!important}
.st-key-shot img{border-radius:8px;max-height:460px;object-fit:contain;background:#111;width:100%}
.vg-verdict{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:26px;display:grid;grid-template-columns:auto 1fr;gap:22px;align-content:start}
.vg-gauge{position:relative;width:128px;height:128px}
.vg-gauge>div{position:absolute;inset:0;display:grid;place-content:center;text-align:center}
.vg-gauge b{font:800 30px var(--f1);line-height:1}.vg-gauge small{font-size:11px;color:var(--muted)}
.vg-ring{animation:ring 1.3s cubic-bezier(.2,.8,.2,1)}
@keyframes ring{from{stroke-dashoffset:364.4}}
@keyframes grow{from{width:0}}
.vg-verdict h2{font:800 28px/1.1 var(--f1);letter-spacing:-.02em;margin:0 0 6px;padding:0}
.vg-bars{grid-column:1/-1}
.vg-bar{display:grid;grid-template-columns:1fr auto;gap:4px 10px;font-size:14px;margin-bottom:14px}
.vg-bar i{grid-column:1/-1;height:10px;background:var(--line);border-radius:99px;overflow:hidden;display:block}
.vg-bar u{display:block;height:100%;border-radius:99px;animation:grow 1.1s cubic-bezier(.2,.8,.2,1);text-decoration:none}
.vg-ph{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.vg-ph h3{font:700 18px var(--f1);margin:0;padding:0}
.vg-pre{margin:0 0 14px;white-space:pre-wrap;font:15px/1.6 var(--f2);background:var(--paper);border-radius:10px;padding:14px;max-height:200px;overflow:auto}
.vg-m{padding:14px 0;border-top:1px solid var(--line)}.vg-m:first-of-type{border-top:0;padding-top:0}
.vg-m .t{display:flex;justify-content:space-between;gap:10px;align-items:baseline}.vg-m .t b{font-weight:600}
.vg-tag{font-size:12px;font-weight:700;padding:2px 10px;border-radius:99px;white-space:nowrap}
.vg-chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:var(--muted)}
.vg-chips span{background:var(--paper);padding:2px 9px;border-radius:99px}

/* sections */
.vg-sec{padding-top:48px;border-top:1px solid var(--line);margin-top:32px}
.vg-eyebrow{font:700 12px var(--f2);letter-spacing:.14em;color:var(--real);margin:0 0 6px}
.vg-sec h2{font:800 clamp(26px,3.4vw,36px)/1.1 var(--f1);letter-spacing:-.02em;margin:0 0 8px;padding:0}
.vg-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0}
.vg-stat{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:18px 20px}
.vg-stat span{font-size:13px;color:var(--muted);display:block}
.vg-stat strong{font:800 34px var(--f1);display:block;line-height:1.15}
.vg-stat small{font-size:12px;color:var(--muted)}
.vg-charts{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
.vg-chart{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:20px}
.vg-chart h3{font:700 18px var(--f1);margin:0 0 2px;padding:0}.vg-chart img{width:100%;margin-top:10px;border-radius:8px}
.vg-chart p{margin:6px 0 0;font-size:14px}
.vg-foot{font-size:13px;color:var(--muted);margin-top:14px}
.vg-enh{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0}
.vg-enh article{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:20px}
.vg-enh .no{font:800 14px var(--f1);color:var(--real)}.vg-enh h3{font:700 18px var(--f1);margin:6px 0;padding:0}.vg-enh p{font-size:14px;color:var(--muted);margin:0}
.vg-concl{background:var(--ink);color:var(--paper);border-radius:var(--r);padding:30px}
.vg-concl h2,.vg-concl p{color:var(--paper)}.vg-concl .vg-eyebrow{color:#78dcbe}.vg-concl p{opacity:.85;max-width:80ch}
.vg-how ol{list-style:none;counter-reset:s;display:grid;grid-template-columns:repeat(3,1fr);gap:28px;padding:0;margin:24px 0}
.vg-how li{counter-increment:s}
.vg-how li::before{content:counter(s);display:grid;place-items:center;width:34px;height:34px;border-radius:50%;background:var(--ink);color:var(--paper);font:700 15px var(--f1);margin-bottom:12px}
.vg-how li b{display:block;font:700 19px var(--f1)}.vg-how li span{color:var(--muted);font-size:15px}
.vg-note{font-size:14px;color:var(--muted);max-width:70ch}
.vg-footer{text-align:center;color:var(--muted);font-size:13px;padding:30px 0 0}
@media(max-width:800px){.vg-stats,.vg-charts,.vg-enh,.vg-how ol{grid-template-columns:1fr}.vg-verdict{grid-template-columns:1fr}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- cached resources ----------
def find_model():
    """Look for the model in the expected place, then anywhere in the repo."""
    cands = [MODEL_PATH, Path.cwd() / "models" / "image_detector.keras"]
    for root in (BASE, BASE.parent, Path.cwd()):
        cands += sorted(root.rglob("*.keras"))[:5]
    for c in cands:
        if c.is_file() and c.stat().st_size > 1_000_000:  # ignore LFS pointer stubs
            return c
    if MODEL_URL:
        import urllib.request
        dest = BASE / "models" / "image_detector.keras"
        dest.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, dest)
        return dest
    return None


@st.cache_resource(show_spinner="Loading image model…")
def load_model():
    path = find_model()
    if path is None:
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(str(path), compile=False)


@st.cache_resource(show_spinner="Loading OCR engine (first run downloads models)…")
def load_ocr():
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False)
    except Exception as e:
        return e


@st.cache_data
def load_default_products():
    return pd.read_csv(PRODUCTS_PATH)


@st.cache_data
def load_metrics():
    p = EVAL_DIR / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def data_uri(name):
    p = EVAL_DIR / name
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


# ---------- logic ----------
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
esc = lambda s: html.escape(str(s))


def match_products(text, df):
    q, out = norm(text), []
    for _, r in df.iterrows():
        n, b, qt = (fuzz.partial_ratio(norm(r[k]), q) if q else 0
                    for k in ("product_name", "brand", "net_quantity"))
        out.append(dict(product_name=r["product_name"], brand=r["brand"], net_quantity=r["net_quantity"],
                        name=n, brand_s=b, qty=qt, overall=round(.60 * n + .25 * b + .15 * qt, 2)))
    return sorted(out, key=lambda m: -m["overall"])


def predict_real(model, img):
    arr = np.expand_dims(np.asarray(img.resize(IMG_SIZE), dtype=np.float32), 0)
    return float(model.predict(arr, verbose=0)[0][0])  # P(real)


def run_ocr(reader, img):
    return [dict(text=str(t), conf=float(c) * 100, box=[(int(x), int(y)) for x, y in b])
            for b, t, c in reader.readtext(np.asarray(img))]


def draw_boxes(img, ocr):
    out = img.copy()
    d = ImageDraw.Draw(out, "RGBA")
    w = max(2, img.width // 260)
    for o in ocr:
        pts = o["box"]
        d.polygon(pts, fill=(120, 220, 190, 30))
        d.line(pts + [pts[0]], fill=(120, 220, 190, 255), width=w)
    return out


def copy_button(text):
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(
        """<button id="b" style="font:600 13px Figtree,system-ui,sans-serif;background:none;border:1px solid #d2c8b8;
border-radius:99px;padding:6px 16px;cursor:pointer;color:#1a1511">Copy text</button>
<script>const t=__T__;const b=document.getElementById('b');
b.onclick=()=>{const done=m=>{b.textContent=m;setTimeout(()=>b.textContent='Copy text',1500)};
const fb=()=>{const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();
try{document.execCommand('copy');done('Copied')}catch(e){done('Copy failed')}a.remove()};
if(navigator.clipboard){navigator.clipboard.writeText(t).then(()=>done('Copied'),fb)}else fb()};</script>""".replace("__T__", payload),
        height=44)


# ---------- header ----------
model = load_model()
pill = ('<span class="vg-pill live"><i></i><span>Live model connected</span></span>' if model is not None
        else '<span class="vg-pill"><i></i><span>Model missing</span></span>')
st.markdown(f"""<div class="vg-top"><div class="vg-logo"><svg width="26" height="26" viewBox="0 0 26 26"><circle cx="11" cy="11" r="8" fill="none" stroke="currentColor" stroke-width="2.4"/><path d="M17 17l6 6" stroke="currentColor" stroke-width="2.8" stroke-linecap="round"/><circle cx="11" cy="11" r="3" fill="#2b7a66"/></svg>VisionGuard</div>{pill}</div>
<nav class="vg-nav"><a href="#performance">Model performance</a><a href="#roadmap">Enhancements &amp; conclusion</a></nav>""",
            unsafe_allow_html=True)

# ---------- hero ----------
if "up_n" not in st.session_state:
    st.session_state.up_n = 0

left, right = st.columns([1, 1.05], gap="large")
with right:
    with st.container(key="film"):
        up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"],
                              key=f"up_{st.session_state.up_n}", label_visibility="collapsed")
        if up is not None:
            st.image(up, use_container_width=True)

with left:
    st.markdown("""<div class="vg-h1">Is this photo real? Then read what’s on it.</div>
<p class="vg-lead">Upload a product or listing image. VisionGuard scores it as camera-captured or AI-generated, reads the label text, and compares it with your product list.</p>""",
                unsafe_allow_html=True)
    b1, b2, _ = st.columns([0.5, 0.3, 0.6])
    go = b1.button("Analyze image", key="go", disabled=(up is None or model is None))
    clear = b2.button("Clear", key="clear")
    with st.expander("Use your own product list (CSV)"):
        csv_up = st.file_uploader("Columns: product_id, product_name, brand, net_quantity", type=["csv"], key="csv")
    if model is None:
        st.error(f"Model file not found (looked for `{MODEL_PATH}`). Add `models/image_detector.keras` "
                 "to your GitHub repo, or set a `MODEL_URL` secret pointing to the file.")
        with st.expander("What the server can see"):
            seen = sorted(str(p.relative_to(BASE.parent)) for p in BASE.rglob("*")
                          if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts)
            st.code("\n".join(seen[:80]) or "(nothing)")

if clear:
    st.session_state.pop("result", None)
    st.session_state.up_n += 1
    st.rerun()

if go and up is not None:
    try:
        img = Image.open(up).convert("RGB")
    except Exception:
        st.error("Could not read that file as an image.")
        st.stop()
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    try:
        ref = pd.read_csv(csv_up) if csv_up else load_default_products()
    except Exception as e:
        st.error(f"Could not read product CSV: {e}")
        st.stop()
    if not REQUIRED_COLS.issubset(ref.columns):
        st.error(f"CSV needs columns: {sorted(REQUIRED_COLS)}")
        st.stop()
    t0 = time.time()
    with st.spinner("Analyzing…"):
        p_real = predict_real(model, img)
        reader = load_ocr()
        ocr = [] if isinstance(reader, Exception) else run_ocr(reader, img)
    st.session_state["result"] = dict(
        p_real=p_real, ocr=ocr, ms=int((time.time() - t0) * 1000), size=img.size,
        kb=len(up.getvalue()) / 1024, image=img,
        ocr_error=str(reader) if isinstance(reader, Exception) else None,
        matches=match_products(" ".join(o["text"] for o in ocr), ref) if ocr else [])

# ---------- report ----------
r = st.session_state.get("result")
if r and up is not None:
    if r["ocr_error"]:
        st.markdown(f'<div class="vg-banner">OCR is unavailable on this server ({esc(r["ocr_error"])}). Showing the image score only.</div>',
                    unsafe_allow_html=True)
    p = r["p_real"]
    real = p >= .5
    conf = max(p, 1 - p) * 100
    col = "#2b7a66" if real else "#b3402e"
    off = RING * (1 - conf / 100)

    c1, c2 = st.columns([1.1, 1], gap="medium")
    with c1:
        with st.container(key="shot"):
            show = st.checkbox("Show detected text regions", value=True)
            st.image(draw_boxes(r["image"], r["ocr"]) if show else r["image"], use_container_width=True)
            st.caption(f"{r['size'][0]}×{r['size'][1]} · {r['kb']:.0f} KB · {r['ms']/1000:.1f} s")
    with c2:
        st.markdown(f"""<div class="vg-verdict">
<div class="vg-gauge"><svg viewBox="0 0 140 140"><circle cx="70" cy="70" r="58" fill="none" stroke="#d2c8b8" stroke-width="10"/><circle class="vg-ring" cx="70" cy="70" r="58" fill="none" stroke="{col}" stroke-width="10" stroke-linecap="round" stroke-dasharray="{RING}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 70 70)"/></svg><div><b>{conf:.0f}%</b><small>confidence</small></div></div>
<div><h2 style="color:{col}">{'Looks like a real photo' if real else 'Looks AI-generated'}</h2><p class="vg-sub">A model estimate, not proof that a product or seller is genuine.</p></div>
<div class="vg-bars">
<div class="vg-bar"><span>Camera-captured</span><b>{p*100:.1f}%</b><i><u style="width:{p*100:.1f}%;background:#2b7a66"></u></i></div>
<div class="vg-bar"><span>AI-generated</span><b>{(1-p)*100:.1f}%</b><i><u style="width:{(1-p)*100:.1f}%;background:#b3402e"></u></i></div>
</div></div>""", unsafe_allow_html=True)

    t1, t2 = st.columns(2, gap="medium")
    with t1:
        with st.container(key="panel_text"):
            txt = "\n".join(o["text"] for o in r["ocr"])
            bars = "".join(
                f'<div class="vg-bar"><span>{esc(o["text"][:36])}</span><b>{o["conf"]:.0f}%</b><i><u style="width:{min(o["conf"],100):.0f}%;background:#1a1511"></u></i></div>'
                for o in r["ocr"][:8])
            st.markdown(f'<div class="vg-ph"><h3>Text found</h3></div><div class="vg-pre">{esc(txt) or "No readable text detected."}</div>{bars}',
                        unsafe_allow_html=True)
            if txt:
                copy_button(txt)
    with t2:
        with st.container(key="panel_match"):
            ms = ""
            for m in r["matches"][:3]:
                s = m["overall"]
                c = "#2b7a66" if s >= 75 else "#b57d22" if s >= 50 else "#b3402e"
                tg = "Strong" if s >= 75 else "Partial" if s >= 50 else "Weak"
                ms += (f'<div class="vg-m"><div class="t"><b>{esc(m["product_name"])}</b>'
                       f'<span class="vg-tag" style="color:{c};background:{c}24">{tg} · {s:.0f}%</span></div>'
                       f'<div class="vg-sub">{esc(m["brand"])} · {esc(m["net_quantity"])}</div>'
                       f'<div class="vg-chips"><span>Name {m["name"]:.0f}%</span><span>Brand {m["brand_s"]:.0f}%</span><span>Quantity {m["qty"]:.0f}%</span></div></div>')
            st.markdown(f'<div class="vg-ph"><h3>Closest products</h3></div>{ms or "<p class=vg-sub>No text to match.</p>"}'
                        '<p class="vg-sub" style="margin-top:10px">Similarity of printed text only. It does not verify authenticity.</p>',
                        unsafe_allow_html=True)

# ---------- performance ----------
m = load_metrics()
if m:
    R = m["classes"]["REAL"]
    pct = lambda v: f"{v*100:.1f}%"
    stats = [("Test accuracy", pct(m["accuracy"]), "Correct predictions / all test images"),
             ("Precision · REAL", pct(R["precision"]), "How often REAL predictions are correct"),
             ("Recall · REAL", pct(R["recall"]), "REAL images correctly identified"),
             ("F1-score · REAL", pct(R["f1"]), "Balance of precision and recall"),
             ("ROC-AUC", f"{m['roc_auc']:.4f}", "Ability to separate the two classes"),
             ("Test images", f"{m['test_count']:,}", "Images used for evaluation")]
    stats_html = '<div class="vg-stats">' + "".join(
        f'<article class="vg-stat"><span>{a}</span><strong>{b}</strong><small>{c}</small></article>' for a, b, c in stats) + "</div>"
    status = "Saved evaluation results loaded"
else:
    stats_html, status = "", "Saved evaluation data unavailable"

def chart(title, sub, fn):
    uri = data_uri(fn)
    return f'<article class="vg-chart"><h3>{title}</h3><p class="vg-sub">{sub}</p>' + (f'<img src="{uri}" alt="{title}">' if uri else "") + "</article>"

st.markdown(f"""<section class="vg-sec" id="performance"><p class="vg-eyebrow">MODEL EVALUATION</p><h2>Performance &amp; statistics</h2>
<p class="vg-sub">Measured on the held-out CIFAKE test set. These are model evaluation results, not per-image confidence scores. · {status}</p>
{stats_html}
<div class="vg-charts">{chart("Confusion matrix", "Actual class vs. predicted class", "confusion_matrix.png")}{chart("ROC curve", "ROC-AUC: 0.9914", "roc_curve.png")}{chart("Precision-recall curve", "Precision remains high across most recall values", "precision_recall_curve.png")}<article class="vg-chart"><h3>Classification summary</h3><p class="vg-sub">Test set: 10,000 FAKE and 10,000 REAL images</p><p>FAKE — precision 0.96, recall 0.95, F1-score 0.95.</p><p>REAL — precision 0.95, recall 0.96, F1-score 0.95.</p><p>Correct predictions: 19,093 of 20,000.</p></article></div>
<p class="vg-foot">Evaluation is only as representative as its test data. CIFAKE is a general real-vs-synthetic image dataset, not a dedicated product-authenticity benchmark.</p></section>""",
            unsafe_allow_html=True)

# ---------- roadmap ----------
items = [("01", "Product-specific training", "Build a carefully labelled dataset of genuine and manipulated product photographs, then fine-tune and validate the detector on those images."),
         ("02", "Explainable predictions", "Add visual explanations such as Grad-CAM heatmaps so users can see which image regions influenced a model prediction."),
         ("03", "Stronger evaluation", "Test across different cameras, lighting, compression levels and AI generators; report class-wise metrics and confidence calibration."),
         ("04", "Safer product verification", "Combine image signals with OCR, barcode or QR checks, trusted product records and provenance data. Do not treat one model score as proof of authenticity."),
         ("05", "Production readiness", "Add secure uploads, file-size limits, privacy-aware retention, monitoring, model versioning and a stable hosted API."),
         ("06", "Broader model coverage", "Evaluate newer synthetic-image detectors and ensemble approaches, while checking performance on unseen datasets to reduce overfitting.")]
enh = "".join(f'<article><span class="no">{a}</span><h3>{b}</h3><p>{c}</p></article>' for a, b, c in items)
st.markdown(f"""<section class="vg-sec" id="roadmap"><p class="vg-eyebrow">NEXT STEPS</p><h2>Further enhancements</h2>
<div class="vg-enh">{enh}</div>
<div class="vg-concl"><p class="vg-eyebrow">CONCLUSION</p><h2>One tool, multiple verification signals.</h2><p>VisionGuard brings together image classification, text extraction and product-catalog matching in a single workflow. Its current model provides an experimental signal for distinguishing real from AI-generated images, while OCR and fuzzy matching help compare visible label text with reference product data. The system is a decision-support prototype—not a guarantee of product authenticity. Product-specific validation, transparent evaluation and trusted external evidence are essential before real-world use.</p></div></section>
<section class="vg-sec vg-how"><h2>How a check works</h2><ol>
<li><b>Classify</b><span>An EfficientNetB0 model trained on CIFAKE scores the image as real or synthetic.</span></li>
<li><b>Read</b><span>EasyOCR finds every line of text and marks where it sits in the image.</span></li>
<li><b>Match</b><span>Fuzzy matching compares the text with product name, brand and quantity.</span></li></ol>
<p class="vg-note">CIFAKE is a general image dataset, not a product dataset. Treat scores as one signal among several, and test on your own product photos before relying on them.</p></section>
<div class="vg-footer">VisionGuard AI · Experimental research tool</div>""", unsafe_allow_html=True)
