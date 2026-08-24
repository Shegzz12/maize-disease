import os
import io
import json
import sqlite3
import urllib.request
import uuid
import time
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import onnxruntime as ort

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# =====================================================================
# CONFIGURATION
# =====================================================================

# --- Model sources (Hugging Face) ---
MODEL_URLS = {
    "gate":    "https://huggingface.co/Samson123Ade/maize-infection-detection/resolve/main/gate_model.onnx",
    "disease": "https://huggingface.co/Samson123Ade/maize-infection-detection/resolve/main/disease_model.onnx",
    "pest":    "https://huggingface.co/Samson123Ade/maize-infection-detection/resolve/main/best.onnx",
}

MODEL_LOCAL_PATHS = {
    "gate": "gate_model.onnx",
    "disease": "disease_model.onnx",
    "pest": "best.onnx",
}

DB_PATH = "database.db"
IMAGE_SIZE = (224, 224)  # same input shape for all three models
UPLOAD_FOLDER = "static/images"

CONFIDENCE_THRESHOLD = 5.0  # kept from original code (currently informational only)

# Label used by the disease model to mean "no disease present"
HEALTHY_LABEL = "healthy"

# Disease model class order — MUST match the order used during training
# (see the provided training script: classes=CLASS_NAMES, in that exact order)
DISEASE_CLASS_NAMES = ["HEALTHY", "leaf blight", "leaf spot", "streak virus"]

# --- Gate model config ---------------------------------------------------
# ASSUMPTION (please verify against how you trained gate_model.onnx):
#   - 2 output logits/probabilities, softmax over ["maize", "not_maize"]
#   - index 0 = "maize", index 1 = "not_maize"
#   - same preprocessing family as the disease model (MobileNetV2-style,
#     scaled to [-1, 1]) since no training script was provided for it.
# If your gate model actually outputs a single sigmoid value instead of
# two softmax classes, or uses ImageNet mean/std normalization, adjust
# GATE_OUTPUT_MODE and preprocess_mobilenet/preprocess_imagenet usage below.
GATE_MAIZE_INDEX = 0
GATE_OUTPUT_MODE = "softmax_2class"  # or "sigmoid_1class"
GATE_PREPROCESS = "mobilenet"        # "mobilenet" ([-1,1]) or "imagenet" (mean/std)
GATE_CONFIDENCE_THRESHOLD = 50.0     # % confidence required to trust "maize" verdict

# --- Pest model normalization (unchanged from original deployment) ------
# Same normalization used during training for best.onnx (ImageNet mean/std)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Global session/category holders ---
sessions = {"gate": None, "disease": None, "pest": None}
PEST_CATEGORY_MAP = {}      # keyed by str(class_id) -> {"problem", "cultural_biological", "chemical_direct"}
DISEASE_CATEGORY_MAP = {}   # keyed by str(class_id) -> {"problem", "cultural_biological", "chemical_direct"}


# =====================================================================
# 1. DATABASE INITIALIZATION
# =====================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ========== FIX: create predictions table unconditionally ==========
    # This table is written to by log_prediction() (gate reject case) and
    # read by /api/latest. It must always exist, even on a fresh DB.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source TEXT,
            detection_type TEXT,
            problem TEXT,
            confidence REAL,
            cultural_biological TEXT,
            chemical_direct TEXT,
            image_path TEXT
        )
    ''')

    # Create new table for combined disease + pest results per image
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source TEXT,
            image_path TEXT,
            disease_problem TEXT,
            disease_confidence REAL,
            disease_is_healthy INTEGER,
            disease_cultural_biological TEXT,
            disease_chemical_direct TEXT,
            pest_problem TEXT,
            pest_confidence REAL,
            pest_is_healthy INTEGER,
            pest_cultural_biological TEXT,
            pest_chemical_direct TEXT,
            gate_confidence REAL
        )
    ''')

    # Migrate from old predictions table if it exists (and has data)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    old_table_exists = cursor.fetchone() is not None

    if old_table_exists:
        # Check if we've already migrated
        cursor.execute("SELECT COUNT(*) FROM assessments")
        if cursor.fetchone()[0] == 0:
            print("Migrating data from old predictions table to new assessments table...")
            cursor.execute('''
                SELECT timestamp, source, image_path,
                       MAX(CASE WHEN detection_type = 'disease' THEN problem END) as disease_problem,
                       MAX(CASE WHEN detection_type = 'disease' THEN confidence END) as disease_confidence,
                       MAX(CASE WHEN detection_type = 'disease' THEN cultural_biological END) as disease_cultural_biological,
                       MAX(CASE WHEN detection_type = 'disease' THEN chemical_direct END) as disease_chemical_direct,
                       MAX(CASE WHEN detection_type = 'pest' THEN problem END) as pest_problem,
                       MAX(CASE WHEN detection_type = 'pest' THEN confidence END) as pest_confidence,
                       MAX(CASE WHEN detection_type = 'pest' THEN cultural_biological END) as pest_cultural_biological,
                       MAX(CASE WHEN detection_type = 'pest' THEN chemical_direct END) as pest_chemical_direct
                FROM predictions
                WHERE image_path IS NOT NULL AND image_path != ''
                GROUP BY image_path, timestamp, source
            ''')
            old_data = cursor.fetchall()

            for row in old_data:
                cursor.execute('''
                    INSERT INTO assessments (timestamp, source, image_path,
                                            disease_problem, disease_confidence, disease_cultural_biological, disease_chemical_direct,
                                            pest_problem, pest_confidence, pest_cultural_biological, pest_chemical_direct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', row)

            conn.commit()
            print(f"Migrated {len(old_data)} records to new assessments table")

    conn.close()


# =====================================================================
# 2. CATEGORY MAPPINGS
# =====================================================================
def _load_json_map(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} entries from {filename}")
            return data
        except Exception as e:
            print(f"Warning: Failed to load {filename}: {e}")
            return {}
    else:
        print(f"Warning: {filename} not found at {path}")
        return {}


def load_categories():
    global PEST_CATEGORY_MAP, DISEASE_CATEGORY_MAP
    PEST_CATEGORY_MAP = _load_json_map("categories.json")
    DISEASE_CATEGORY_MAP = _load_json_map("disease_categories.json")


# =====================================================================
# 3. ONNX MODEL LOADER (generic, reused for all three models)
# =====================================================================
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


def load_onnx_model(model_key):
    """Loads (downloading/re-downloading if needed) and caches an ONNX session
    for one of "gate", "disease", "pest"."""
    global sessions
    if sessions.get(model_key) is not None:
        return sessions[model_key]

    url = MODEL_URLS[model_key]
    local_path = MODEL_LOCAL_PATHS[model_key]

    local_exists = os.path.exists(local_path)
    local_size = os.path.getsize(local_path) if local_exists else None
    remote_size = _get_remote_content_length(url)

    needs_download = (
        not local_exists
        or (remote_size is not None and remote_size != local_size)
    )

    if needs_download:
        print(f"Downloading '{model_key}' ONNX model from Hugging Face: {url} "
              f"(local_size={local_size}, remote_size={remote_size}) ...")
        urllib.request.urlretrieve(url, local_path)
        print(f"'{model_key}' model download complete.")
    else:
        print(f"Using cached '{model_key}' model at {local_path} "
              f"(size={local_size}, remote reports same size)")

    print(f"Loading ONNX Runtime Session for '{model_key}'...")
    session = ort.InferenceSession(local_path, providers=['CPUExecutionProvider'])
    sessions[model_key] = session
    print(f"'{model_key}' ONNX Session successfully initialized.")
    return session


def load_all_models():
    """Eagerly load all three models at startup so the first request isn't slow."""
    for key in ("gate", "disease", "pest"):
        try:
            load_onnx_model(key)
        except Exception as e:
            print(f"Warning: failed to preload '{key}' model at startup: {e}")


# =====================================================================
# 4. IMAGE PREPROCESSING
# =====================================================================
def _load_resized_rgb(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    return np.array(img, dtype=np.float32)


def preprocess_imagenet(image_bytes):
    """ImageNet mean/std normalization — used by the pest model (best.onnx).
    Returns an unbatched HWC array; `format_for_model` handles the final
    layout (NCHW vs NHWC) and batch dimension per the target model."""
    img_data = _load_resized_rgb(image_bytes) / 255.0
    return (img_data - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_mobilenet(image_bytes):
    """MobileNetV2 `preprocess_input`-equivalent normalization: scales to
    [-1, 1]. Used by the disease model (and, by default, the gate model —
    see GATE_PREPROCESS note above). Matches keras.applications.mobilenet_v2
    .preprocess_input, which does: x / 127.5 - 1, channel order RGB.
    Returns an unbatched HWC array; `format_for_model` handles the final
    layout and batch dimension per the target model."""
    img_data = _load_resized_rgb(image_bytes)
    return (img_data / 127.5) - 1.0


def format_for_model(sess, hwc_image):
    """Adapt a normalized HWC (height, width, channels) image to the exact
    input layout the ONNX model expects, then add the batch dimension.

    Models exported from Keras/TF are channels-last (N, H, W, C) while models
    exported from PyTorch/torchvision are channels-first (N, C, H, W). We read
    the target layout from the model's own input signature so each model gets
    the tensor it expects, instead of hard-coding one convention.
    """
    shape = sess.get_inputs()[0].shape  # e.g. [1, 3, 224, 224] or [N, 224, 224, 3]
    # Channels-last when the last static dim is the 3 colour channels.
    channels_last = len(shape) == 4 and shape[-1] == 3
    arr = hwc_image if channels_last else np.transpose(hwc_image, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =====================================================================
# 5. IMAGE SAVING
# =====================================================================
def save_image(image_bytes, source="ESP32"):
    """Save image bytes to static/images directory and return the relative path."""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Generate unique filename
    filename = f"{source}_{uuid.uuid4().hex[:8]}_{int(time.time())}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Save the image
    with open(filepath, 'wb') as f:
        f.write(image_bytes)

    # Return relative path for serving via Flask
    return f"/static/images/{filename}"


# =====================================================================
# 6. INFERENCE HELPERS
# =====================================================================
def run_gate_model(image_bytes):
    """Runs the 'is maize detected?' gate model.
    Returns (is_maize: bool, confidence_pct: float)."""
    sess = load_onnx_model("gate")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    hwc = preprocess_mobilenet(image_bytes) if GATE_PREPROCESS == "mobilenet" \
        else preprocess_imagenet(image_bytes)
    tensor = format_for_model(sess, hwc)

    raw = sess.run([output_name], {input_name: tensor})[0][0]

    if GATE_OUTPUT_MODE == "sigmoid_1class":
        maize_prob = float(sigmoid(raw)[0]) if hasattr(raw, "__len__") else float(sigmoid(raw))
        is_maize = maize_prob * 100 >= GATE_CONFIDENCE_THRESHOLD
        return is_maize, round(maize_prob * 100, 2)

    # default: softmax_2class
    probs = softmax(raw)
    maize_prob = float(probs[GATE_MAIZE_INDEX]) * 100
    is_maize = maize_prob >= GATE_CONFIDENCE_THRESHOLD
    return is_maize, round(maize_prob, 2)


def run_classifier(model_key, image_bytes, class_name_lookup, category_map, preprocess_fn):
    """Generic single-label classifier runner shared by disease + pest models.
    class_name_lookup: optional list of class names (used for disease model,
                        since we know DISEASE_CLASS_NAMES from training).
                        Pass None to rely purely on category_map's "problem" field
                        (used for the pest model, matching original behavior).
    """
    sess = load_onnx_model(model_key)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    tensor = format_for_model(sess, preprocess_fn(image_bytes))
    raw_output = sess.run([output_name], {input_name: tensor})[0][0]
    probabilities = softmax(raw_output)

    top1_idx = int(np.argmax(probabilities))
    top1_conf = float(probabilities[top1_idx]) * 100

    category_entry = category_map.get(str(top1_idx), {})

    if class_name_lookup and 0 <= top1_idx < len(class_name_lookup):
        default_name = class_name_lookup[top1_idx]
    else:
        default_name = f"Class {top1_idx}"

    problem_name = category_entry.get("problem", default_name)
    cultural = category_entry.get("cultural_biological", "Maintain proper crop spacing and weed control.")
    chemical = category_entry.get("chemical_direct", "Apply targeted bio-pesticide if threshold exceeded.")
    is_healthy = problem_name.strip().lower() == HEALTHY_LABEL

    return {
        "class_id": top1_idx,
        "problem": problem_name,
        "confidence": round(top1_conf, 2),
        "is_healthy": is_healthy,
        "cultural_biological": cultural,
        "chemical_direct": chemical,
    }


def log_assessment(source, disease_result, pest_result, image_path=None, gate_confidence=None):
    """Log combined disease + pest assessment as a single record."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO assessments (source, image_path,
                                disease_problem, disease_confidence, disease_is_healthy, disease_cultural_biological, disease_chemical_direct,
                                pest_problem, pest_confidence, pest_is_healthy, pest_cultural_biological, pest_chemical_direct,
                                gate_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (source, image_path,
          disease_result["problem"], disease_result["confidence"], int(disease_result["is_healthy"]),
          disease_result["cultural_biological"], disease_result["chemical_direct"],
          pest_result["problem"], pest_result["confidence"], int(pest_result["is_healthy"]),
          pest_result["cultural_biological"], pest_result["chemical_direct"],
          gate_confidence))
    conn.commit()
    conn.close()

def log_prediction(source, detection_type, result, image_path=None):
    """Legacy function for backward compatibility - now logs to old table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (source, detection_type, problem, confidence, cultural_biological, chemical_direct, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (source, detection_type, result["problem"], result["confidence"],
          result["cultural_biological"], result["chemical_direct"], image_path))
    conn.commit()
    conn.close()


# Startup tasks
init_db()
load_categories()
load_all_models()

# =====================================================================
# ROUTES
# =====================================================================

@app.route("/")
def index():
    """Serves the frontend directly"""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "gate_model_loaded": sessions["gate"] is not None,
        "disease_model_loaded": sessions["disease"] is not None,
        "pest_model_loaded": sessions["pest"] is not None,
        "database": os.path.exists(DB_PATH),
        "pest_categories_count": len(PEST_CATEGORY_MAP),
        "disease_categories_count": len(DISEASE_CATEGORY_MAP),
    })


@app.route("/api/categories", methods=["GET"])
def get_categories():
    """Serves category mappings to the frontend"""
    return jsonify({
        "success": True,
        "pest": {"count": len(PEST_CATEGORY_MAP), "categories": PEST_CATEGORY_MAP},
        "disease": {"count": len(DISEASE_CATEGORY_MAP), "categories": DISEASE_CATEGORY_MAP},
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Main endpoint for web app and ESP32 uploads.

    Flow (matches the block diagram):
      1. Save image if from ESP32
      2. Run gate model -> is maize detected?
         - No  -> return early with a "no maize / healthy" style response.
         - Yes -> run disease model AND pest model, return both results.
    """
    try:
        image_path = None

        if "file" in request.files:
            file_bytes = request.files["file"].read()
            source = request.form.get("source", "Web Client")
            # Also save web client images
            image_path = save_image(file_bytes, "WebClient")
        else:
            file_bytes = request.data
            source = "ESP32"
            # Save ESP32 images
            image_path = save_image(file_bytes, "ESP32")

        if not file_bytes:
            return jsonify({"success": False, "error": "No image data received"}), 400

        # --- Step 1: gate check -------------------------------------------------
        is_maize, gate_confidence = run_gate_model(file_bytes)

        if not is_maize:
            not_maize_result = {
                "problem": "No maize detected",
                "confidence": gate_confidence,
                "is_healthy": True,
                "cultural_biological": "No action needed — no maize plant detected in image.",
                "chemical_direct": "No action needed — no maize plant detected in image.",
            }
            log_prediction(source, "gate_reject", not_maize_result, image_path)
            return jsonify({
                "success": True,
                "source": source,
                "maize_detected": False,
                "gate_confidence": gate_confidence,
                "message": "No maize detected in the image.",
                "image_path": image_path,
            })

        # --- Step 2: maize detected -> run disease model + pest model ----------
        disease_result = run_classifier(
            "disease", file_bytes,
            class_name_lookup=DISEASE_CLASS_NAMES,
            category_map=DISEASE_CATEGORY_MAP,
            preprocess_fn=preprocess_mobilenet,
        )
        pest_result = run_classifier(
            "pest", file_bytes,
            class_name_lookup=None,
            category_map=PEST_CATEGORY_MAP,
            preprocess_fn=preprocess_imagenet,
        )

        # Log combined assessment
        log_assessment(source, disease_result, pest_result, image_path, gate_confidence)

        detected_faults = []
        for kind, result in (("disease", disease_result), ("pest", pest_result)):
            detected_faults.append({
                "type": kind,
                "class_id": result["class_id"],
                "problem": result["problem"],
                "confidence": result["confidence"],
                "is_healthy": result["is_healthy"],
                "cultural_biological": result["cultural_biological"],
                "chemical_direct": result["chemical_direct"],
            })

        return jsonify({
            "success": True,
            "source": source,
            "maize_detected": True,
            "gate_confidence": gate_confidence,
            "image_path": image_path,
            "disease": {
                "problem": disease_result["problem"],
                "confidence": disease_result["confidence"],
                "is_healthy": disease_result["is_healthy"],
                "solutions": {
                    "cultural_biological": disease_result["cultural_biological"],
                    "chemical_direct": disease_result["chemical_direct"],
                },
            },
            "pest": {
                "problem": pest_result["problem"],
                "confidence": pest_result["confidence"],
                "is_healthy": pest_result["is_healthy"],
                "solutions": {
                    "cultural_biological": pest_result["cultural_biological"],
                    "chemical_direct": pest_result["chemical_direct"],
                },
            },
            "detected_faults": detected_faults,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/latest", methods=["GET"])
def get_latest():
    """Fetches the latest reading from DB"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, source, detection_type, problem, confidence, "
        "cultural_biological, chemical_direct, image_path FROM predictions ORDER BY id DESC LIMIT 1"
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"success": False, "message": "No database records found"}), 404

    return jsonify({
        "success": True,
        "id": row[0],
        "timestamp": row[1],
        "source": row[2],
        "detection_type": row[3],
        "problem": row[4],
        "confidence": row[5],
        "solutions": {
            "cultural_biological": row[6],
            "chemical_direct": row[7]
        },
        "image_path": row[8]
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns historical logs with combined disease + pest data per image"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, source, image_path,
               disease_problem, disease_confidence, disease_is_healthy,
               disease_cultural_biological, disease_chemical_direct,
               pest_problem, pest_confidence, pest_is_healthy,
               pest_cultural_biological, pest_chemical_direct,
               gate_confidence
        FROM assessments
        ORDER BY id DESC LIMIT 20
    ''')
    rows = cursor.fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "id": r[0],
            "timestamp": r[1],
            "source": r[2],
            "image_path": r[3],
            "disease": {
                "problem": r[4],
                "confidence": r[5],
                "is_healthy": bool(r[6]),
                "cultural_biological": r[7],
                "chemical_direct": r[8]
            },
            "pest": {
                "problem": r[9],
                "confidence": r[10],
                "is_healthy": bool(r[11]),
                "cultural_biological": r[12],
                "chemical_direct": r[13]
            },
            "gate_confidence": r[14]
        })

    return jsonify({"success": True, "history": history})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
