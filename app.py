import os
import io
import json
import sqlite3
import urllib.request
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import onnxruntime as ort

app = Flask(__name__, template_folder="templates")
CORS(app)

# --- CONFIGURATION ---
MODEL_URL = "https://huggingface.co/Samson123Ade/maize-infection-detection/resolve/main/best.onnx"
MODEL_LOCAL_PATH = "best.onnx"
DB_PATH = "database.db"
CONFIDENCE_THRESHOLD = 5.0
IMAGE_SIZE = (224, 224)  # Standard ONNX input shape

# Label used by the new 6-class model to mean "no disease/pest present"
HEALTHY_LABEL = "healthy"

# Same normalization used during training (ImageNet mean/std) — must match
# the other maize-infection-detection deployment, since both load the same
# best.onnx model.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

session = None
CATEGORY_MAP = {}

# --- 1. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source TEXT,
            problem TEXT,
            confidence REAL,
            cultural_biological TEXT,
            chemical_direct TEXT
        )
    ''')
    conn.commit()
    conn.close()

# --- 2. CATEGORIES MAPPING ---
def load_categories():
    global CATEGORY_MAP
    base_dir = os.path.dirname(os.path.abspath(__file__))
    category_file = os.path.join(base_dir, "categories.json")
    if os.path.exists(category_file):
        try:
            with open(category_file, "r", encoding="utf-8") as f:
                CATEGORY_MAP = json.load(f)
            print(f"Successfully loaded {len(CATEGORY_MAP)} categories from categories.json")
        except Exception as e:
            print(f"Warning: Failed to load categories.json: {e}")
    else:
        print(f"Warning: categories.json not found at {category_file}")

# --- 3. ONNX MODEL LOADER FROM HUGGINGFACE ---
def _get_remote_content_length(url):
    """HEAD request to check the size of the file currently hosted at `url`.
    Returns None if it can't be determined (network issue, no Content-Length, etc.)."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            size = resp.headers.get("Content-Length")
            return int(size) if size is not None else None
    except Exception as e:
        print(f"Could not check remote size for {url}: {e}")
        return None

def load_onnx_model():
    global session
    if session is not None:
        return session

    local_exists = os.path.exists(MODEL_LOCAL_PATH)
    local_size = os.path.getsize(MODEL_LOCAL_PATH) if local_exists else None
    remote_size = _get_remote_content_length(MODEL_URL)

    # Re-download if there's no local copy, OR if the file hosted at MODEL_URL
    # has a different size than what's cached (i.e. a new model was pushed to
    # Hugging Face under the same filename).
    needs_download = (
        not local_exists
        or (remote_size is not None and remote_size != local_size)
    )

    if needs_download:
        print(f"Downloading ONNX model from Hugging Face: {MODEL_URL} "
              f"(local_size={local_size}, remote_size={remote_size}) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_LOCAL_PATH)
        print("Model download complete.")
    else:
        print(f"Using cached model at {MODEL_LOCAL_PATH} "
              f"(size={local_size}, remote reports same size)")

    print("Loading ONNX Runtime Session...")
    session = ort.InferenceSession(MODEL_LOCAL_PATH, providers=['CPUExecutionProvider'])
    print("ONNX Session successfully initialized.")
    return session

# --- 4. IMAGE PREPROCESSING FOR ONNX ---
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)

    img_data = np.array(img, dtype=np.float32) / 255.0
    img_data = (img_data - IMAGENET_MEAN) / IMAGENET_STD
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    return img_data

# --- Softmax Helper ---
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Startup tasks
init_db()
load_categories()

# --- ROUTES ---

@app.route("/")
def index():
    """Serves the frontend directly"""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": session is not None,
        "database": os.path.exists(DB_PATH),
        "categories_count": len(CATEGORY_MAP)   # add this
    })


@app.route("/api/categories", methods=["GET"])
def get_categories():
    """Serves category mappings to the frontend"""
    return jsonify({
        "success": True,
        "count": len(CATEGORY_MAP),
        "categories": CATEGORY_MAP
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Main endpoint for web app and ESP32 uploads"""
    try:
        ort_sess = load_onnx_model()

        if "file" in request.files:
            file_bytes = request.files["file"].read()
            source = request.form.get("source", "Web Client")
        else:
            file_bytes = request.data
            source = "ESP32"

        if not file_bytes:
            return jsonify({"success": False, "error": "No image data received"}), 400

        input_tensor = preprocess_image(file_bytes)

        input_name = ort_sess.get_inputs()[0].name
        output_name = ort_sess.get_outputs()[0].name

        outputs = ort_sess.run([output_name], {input_name: input_tensor})
        raw_output = outputs[0][0]

        probabilities = softmax(raw_output)
        top1_idx = int(np.argmax(probabilities))
        top1_conf = float(probabilities[top1_idx]) * 100

        category_entry = CATEGORY_MAP.get(str(top1_idx), {})
        problem_name = category_entry.get("problem", f"Disease Class {top1_idx}")
        cultural = category_entry.get("cultural_biological", "Maintain proper crop spacing and weed control.")
        chemical = category_entry.get("chemical_direct", "Apply targeted bio-pesticide if threshold exceeded.")
        is_healthy = problem_name.strip().lower() == HEALTHY_LABEL

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (source, problem, confidence, cultural_biological, chemical_direct)
            VALUES (?, ?, ?, ?, ?)
        ''', (source, problem_name, round(top1_conf, 2), cultural, chemical))
        conn.commit()
        conn.close()

        # inside /predict, replace the final return with:
        mapped = str(top1_idx) in CATEGORY_MAP
        prediction_obj = {
            "class_id": top1_idx,
            "problem": problem_name,
            "confidence": round(top1_conf, 2),
            "is_healthy": is_healthy
        }
        return jsonify({
            "success": True,
            "source": source,
            "class_id": top1_idx,
            "confidence": round(top1_conf, 2),
            "problem": problem_name,
            "is_healthy": is_healthy,
            "solutions": {"cultural_biological": cultural, "chemical_direct": chemical},
            "prediction": prediction_obj,
            "detected_faults": [{**prediction_obj, "mapped": mapped,
                                  "cultural_biological": cultural, "chemical_direct": chemical}],
            "categories_mapped": 1 if mapped else 0
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/latest", methods=["GET"])
def get_latest():
    """Fetches the latest reading from DB"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, source, problem, confidence, cultural_biological, chemical_direct FROM predictions ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"success": False, "message": "No database records found"}), 404

    return jsonify({
        "success": True,
        "id": row[0],
        "timestamp": row[1],
        "source": row[2],
        "problem": row[3],
        "confidence": row[4],
        "solutions": {
            "cultural_biological": row[5],
            "chemical_direct": row[6]
        }
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns historical logs"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, source, problem, confidence FROM predictions ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()

    history = [
        {"id": r[0], "timestamp": r[1], "source": r[2], "problem": r[3], "confidence": r[4]}
        for r in rows
    ]
    return jsonify({"success": True, "history": history})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
