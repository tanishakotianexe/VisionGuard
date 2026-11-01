# VisionGuard AI — Deployment Package

This package contains the VisionGuard Flask application, static web interface,
dependency list, sample product reference CSV, model metadata, and the original
Google Colab notebook.

## Package contents

- `server.py` — Flask API and web server
- `static/` — browser interface, JavaScript, and styles
- `requirements.txt` — Python dependencies
- `products.csv` — fictional sample reference data
- `models/metadata.json` — model/dataset notes
- `models/PUT_MODEL_HERE.txt` — placeholder; the trained model is **not included**
- `colab_export_cell.py` — corrected Colab cell to download the trained model
- `VisionGuard_run_in_colab.ipynb` — original training/Colab workflow (at package root)

## 1. Install

Use Python 3.10 or 3.11 in a virtual environment.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow and EasyOCR are substantial dependencies. Installation and first-run
OCR model downloads can take time. A compatible Python environment and sufficient
disk/RAM are required.

## 2. Add the trained image model (for live analysis)

The ZIP intentionally does not contain a trained model. To create it:

1. Open `VisionGuard_run_in_colab.ipynb` in Google Colab.
2. Select a GPU runtime if available, then run the notebook cells in order.
3. Wait for the training cell to finish. It saves
   `/content/visionguard/models/image_detector.keras`.
4. Run `colab_export_cell.py` in a new Colab cell (or paste its contents into the
   notebook) and download the model.
5. Place the downloaded file at:

   `visionguard/models/image_detector.keras`

Without this file, the website starts in demo mode and `/api/analyze` returns
HTTP 503. Demo results are samples, not model predictions.

## 3. Start the application

From this `visionguard` directory:

```bash
python server.py
```

Then open <http://127.0.0.1:5000>.

To use a different port:

```bash
PORT=8000 python server.py
```

On Windows PowerShell:

```powershell
$env:PORT=8000
python server.py
```

The health endpoint is <http://127.0.0.1:5000/api/health>. Its `model` field
indicates whether the trained model was loaded.

## 4. Product reference data

`products.csv` is fictional demonstration data. Replace it with a trusted,
appropriately licensed reference list using these required columns:

- `product_id`
- `product_name`
- `brand`
- `net_quantity`

The UI also supports uploading a CSV during analysis.

## Important limitations

- The notebook trains an experimental EfficientNetB0 classifier on CIFAKE.
  Its score is not proof that an image is authentic or AI-generated.
- CIFAKE is not a product dataset; product matching uses the separate CSV.
- The sample product records are fictional and must not be treated as real listings.
- The Flask development server is suitable for local testing, not production
  internet exposure. For production, use a production WSGI server, HTTPS,
  authentication, upload limits, and appropriate monitoring.
- Do not expose the development server directly to the public internet.


## Model performance dashboard

The dashboard displays the saved evaluation artifacts in `static/evaluation/`: metrics, confusion matrix, ROC curve, precision-recall curve and classification report. These are the results already generated for the 20,000-image CIFAKE test set. **You do not need to run `evaluate_model.py` to view the dashboard.** The script remains available only if you intentionally want a fresh evaluation later.
