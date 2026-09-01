import os
import io
import json
import sqlite3
import urllib.request
import uuid
import time
import logging
import traceback
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory, g
from flask_cors import CORS
import onnxruntime as ort

# =====================================================================
# LOGGING SETUP
# =====================================================================
# Everything of interest goes to stdout so it shows up in Render's log
# stream. LOG_LEVEL can be overridden via env var without touching code
# (e.g. set LOG_LEVEL=DEBUG on Render if you need even more detail).
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("maize-app")

# Quiet down noisy third-party loggers a little so our own lines aren't
# buried, but still let warnings/errors from them through.
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


# =====================================================================
# REQUEST / RESPONSE LOGGING (applies to every route automatically)
# =====================================================================
@app.before_request
def _log_request_start():
    g._start_time = time.time()
    try:
        content_length = request.content_length or 0
    except Exception:
        content_length = "?"
    logger.info(
        "--> %s %s | remote=%s | content-type=%s | content-length=%s",
        request.method, request.path, request.remote_addr,
        request.content_type, content_length,
    )


@app.after_request
def _log_request_end(response):
    duration_ms = (time.time() - getattr(g, "_start_time", time.time())) * 1000
    logger.info(
        "<-- %s %s | status=%s | %.1fms",
        request.method, request.path, response.status_code, duration_ms,
    )
    return response


@app.errorhandler(Exception)
def _log_unhandled_exception(e):
    # Catches anything that slips past a route's own try/except so nothing
    # fails silently or with just a bare 500 and no trace in the logs.
    logger.error("UNHANDLED EXCEPTION on %s %s", request.method, request.path)
    logger.error(traceback.format_exc())
    return jsonify({"success": False, "error": f"Internal server error: {e}"}), 500

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
# UPDATED: gate model is now a YOLOv8n-cls (Ultralytics) classifier exported
# to ONNX. Verified directly from the exported ONNX graph metadata/nodes:
#   - metadata "names" = {0: 'maize', 1: 'out_of_bounds'}  -> maize is index 0
#   - the graph's FINAL node is a Softmax op, so output0 is already
#     probabilities (NOT raw logits) — do not re-apply softmax on it.
#   - input "images": shape [1, 3, 224, 224] (NCHW), float32
#   - output "output0": shape [1, 2]
#   - Ultralytics classify preprocessing: RGB, resized to imgsz, scaled to
#     [0, 1] (divide by 255) — NO ImageNet mean/std, NO MobileNet [-1,1]
#     scaling. This is different from the disease/pest models below.
GATE_MAIZE_INDEX = 0
GATE_OUTPUT_MODE = "presoftmax_2class"  # output0 is already softmax'd by the graph
GATE_PREPROCESS = "yolo_cls"            # RGB, resized, scaled to [0,1], no mean/std
GATE_CONFIDENCE_THRESHOLD = 50.0        # % confidence required to trust "maize" verdict

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
    logger.info("Initializing database at '%s' (exists=%s)", DB_PATH, os.path.exists(DB_PATH))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ========== FIX: create predictions table unconditionally ==========
    # This table is written to by log_prediction() (gate reject case) and
    # read by /api/latest. It must always exist on a fresh DB, not only
    # when an old DB happens to already have it — the previous code only
    # ever created it inside the migration branch, so on a brand-new
    # database (e.g. every redeploy on Render's ephemeral disk) the very
    # first "no maize detected" prediction raised:
    #   sqlite3.OperationalError: no such table: predictions
    # which the outer try/except in /predict turned into a 500, and which
    # also broke /api/latest since it queries this table directly.
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
    logger.info("'predictions' table ready")

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
    logger.info("'assessments' table ready")

    # Migrate from old predictions table if it exists (and has data)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    old_table_exists = cursor.fetchone() is not None
    logger.debug("Migration check: old predictions table present=%s", old_table_exists)

    if old_table_exists:
        # Check if we've already migrated
        cursor.execute("SELECT COUNT(*) FROM assessments")
        if cursor.fetchone()[0] == 0:
            # Check the predictions table actually has rows before attempting
            # a migration — on a brand-new DB it was just created above and
            # is empty, so this is a fast no-op rather than an error.
            cursor.execute("SELECT COUNT(*) FROM predictions")
            old_row_count = cursor.fetchone()[0]
            logger.info("Legacy predictions table has %d row(s)", old_row_count)
            if old_row_count > 0:
                logger.info("Migrating data from old predictions table to new assessments table...")
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
                logger.info("Migrated %d record(s) to new assessments table", len(old_data))

    conn.close()
    logger.info("Database initialization complete")


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
            logger.info("Loaded %d entries from %s", len(data), filename)
            return data
        except Exception as e:
            logger.error("Failed to load %s: %s", filename, e)
            logger.error(traceback.format_exc())
            return {}
    else:
        logger.warning("%s not found at %s (category names/solutions will fall back to defaults)", filename, path)
        return {}


def load_categories():
    global PEST_CATEGORY_MAP, DISEASE_CATEGORY_MAP
    logger.info("Loading category mapping files...")
    PEST_CATEGORY_MAP = _load_json_map("categories.json")
    DISEASE_CATEGORY_MAP = _load_json_map("disease_categories.json")
    logger.info(
        "Category maps loaded: pest=%d entries, disease=%d entries",
        len(PEST_CATEGORY_MAP), len(DISEASE_CATEGORY_MAP),
    )


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
            size_int = int(size) if size is not None else None
            logger.debug("HEAD %s -> Content-Length=%s", url, size_int)
            return size_int
    except Exception as e:
        logger.warning("Could not check remote size for %s: %s", url, e)
        return None


def load_onnx_model(model_key):
    """Loads (downloading/re-downloading if needed) and caches an ONNX session
    for one of "gate", "disease", "pest"."""
    global sessions
    if sessions.get(model_key) is not None:
        logger.debug("'%s' model already loaded, reusing cached session", model_key)
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
    logger.info(
        "'%s' model check: local_exists=%s local_size=%s remote_size=%s -> needs_download=%s",
        model_key, local_exists, local_size, remote_size, needs_download,
    )

    if needs_download:
        logger.info("Downloading '%s' ONNX model from %s ...", model_key, url)
        dl_start = time.time()
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            logger.error("Download FAILED for '%s' model from %s: %s", model_key, url, e)
            logger.error(traceback.format_exc())
            raise
        dl_seconds = time.time() - dl_start
        downloaded_size = os.path.getsize(local_path) if os.path.exists(local_path) else None
        logger.info(
            "'%s' model download complete in %.1fs (size=%s bytes)",
            model_key, dl_seconds, downloaded_size,
        )
    else:
        logger.info(
            "Using cached '%s' model at %s (size=%s, remote reports same size)",
            model_key, local_path, local_size,
        )

    logger.info("Loading ONNX Runtime Session for '%s' from %s ...", model_key, local_path)
    load_start = time.time()
    try:
        session = ort.InferenceSession(local_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        logger.error("Failed to create ONNX InferenceSession for '%s': %s", model_key, e)
        logger.error(traceback.format_exc())
        raise
    load_seconds = time.time() - load_start

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    logger.info(
        "'%s' ONNX session ready in %.2fs | input='%s' shape=%s dtype=%s | output='%s' shape=%s",
        model_key, load_seconds, input_meta.name, input_meta.shape, input_meta.type,
        output_meta.name, output_meta.shape,
    )

    sessions[model_key] = session
    return session


def load_all_models():
    """Eagerly load all three models at startup so the first request isn't slow."""
    logger.info("Preloading all models at startup...")
    for key in ("gate", "disease", "pest"):
        try:
            load_onnx_model(key)
        except Exception as e:
            logger.error(
                "Failed to preload '%s' model at startup: %s "
                "(requests needing this model will fail until it loads successfully)",
                key, e,
            )
    loaded = {k: (v is not None) for k, v in sessions.items()}
    logger.info("Startup model preload finished. Loaded status: %s", loaded)


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
    [-1, 1]. Used by the disease model. Matches keras.applications.mobilenet_v2
    .preprocess_input, which does: x / 127.5 - 1, channel order RGB.
    Returns an unbatched HWC array; `format_for_model` handles the final
    layout and batch dimension per the target model."""
    img_data = _load_resized_rgb(image_bytes)
    return (img_data / 127.5) - 1.0


def preprocess_yolo_cls(image_bytes):
    """Ultralytics classification (YOLOv8n-cls) preprocessing — used by the
    gate model. RGB, resized, scaled to [0, 1]. NO ImageNet mean/std and NO
    MobileNet [-1, 1] scaling — those are different normalizations used by
    the pest/disease models respectively and would be wrong here.
    Returns an unbatched HWC array; `format_for_model` handles the final
    layout (this model is NCHW) and batch dimension."""
    img_data = _load_resized_rgb(image_bytes)
    return img_data / 255.0


def format_for_model(sess, hwc_image):
    """Adapt a normalized HWC (height, width, channels) image to the exact
    input layout the ONNX model expects, then add the batch dimension.

    Models exported from Keras/TF are channels-last (N, H, W, C) while models
    exported from PyTorch/torchvision (including Ultralytics YOLO) are
    channels-first (N, C, H, W). We read the target layout from the model's
    own input signature so each model gets the tensor it expects, instead of
    hard-coding one convention.
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


def boost_disease_confidence(raw_confidence_pct):
    """Cosmetic confidence boost applied ONLY to the disease model's
    displayed confidence (per explicit request) — NOT the model's actual
    certainty. Never returns below ~55%, and leaves anything already >=70%
    untouched. Piecewise-linear through these anchor points:
        raw:      0    10    20    30    40    70
        shown:   55    60    60    80    85    70   (unchanged for raw>=70)
    """
    anchors = [
        (0.0, 55.0),
        (10.0, 60.0),
        (20.0, 60.0),
        (30.0, 80.0),
        (40.0, 85.0),
        (70.0, 70.0),
    ]
    if raw_confidence_pct >= 70.0:
        return round(raw_confidence_pct, 2)
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= raw_confidence_pct <= x1:
            frac = (raw_confidence_pct - x0) / (x1 - x0) if x1 > x0 else 0.0
            return round(y0 + frac * (y1 - y0), 2)
    return round(max(raw_confidence_pct, 55.0), 2)


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
    try:
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
    except Exception as e:
        logger.error("Failed to save image to %s: %s", filepath, e)
        logger.error(traceback.format_exc())
        raise

    logger.info("Saved image: %s (%d bytes, source=%s)", filepath, len(image_bytes), source)

    # Return relative path for serving via Flask
    return f"/static/images/{filename}"


# =====================================================================
# 6. INFERENCE HELPERS
# =====================================================================
def run_gate_model(image_bytes):
    """Runs the 'is maize detected?' gate model (YOLOv8n-cls, exported ONNX).
    Returns (is_maize: bool, confidence_pct: float).

    NOTE on this model specifically (verified from the exported ONNX graph):
      - class names: {0: 'maize', 1: 'out_of_bounds'} -> GATE_MAIZE_INDEX = 0
      - the graph's final op is Softmax, so `raw` below is ALREADY a
        probability distribution over the 2 classes — it must NOT be passed
        through softmax() again (that would flatten/distort the confidence).
      - preprocessing is Ultralytics-style: RGB, resized, scaled to [0, 1].
    """
    sess = load_onnx_model("gate")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    hwc = preprocess_yolo_cls(image_bytes) if GATE_PREPROCESS == "yolo_cls" \
        else (preprocess_mobilenet(image_bytes) if GATE_PREPROCESS == "mobilenet"
              else preprocess_imagenet(image_bytes))
    tensor = format_for_model(sess, hwc)

    infer_start = time.time()
    raw = sess.run([output_name], {input_name: tensor})[0][0]
    infer_ms = (time.time() - infer_start) * 1000
    logger.debug("Gate model raw output: %s (inference %.1fms)", raw, infer_ms)

    if GATE_OUTPUT_MODE == "presoftmax_2class":
        # Output already softmax'd by the ONNX graph itself — use as-is.
        probs = raw
        maize_prob = float(probs[GATE_MAIZE_INDEX]) * 100
        is_maize = maize_prob >= GATE_CONFIDENCE_THRESHOLD
        logger.info(
            "Gate result (pre-softmaxed graph output): is_maize=%s confidence=%.2f%% (all probs=%s, threshold=%s%%)",
            is_maize, maize_prob, np.round(probs, 4).tolist(), GATE_CONFIDENCE_THRESHOLD,
        )
        return is_maize, round(maize_prob, 2)

    if GATE_OUTPUT_MODE == "sigmoid_1class":
        maize_prob = float(sigmoid(raw)[0]) if hasattr(raw, "__len__") else float(sigmoid(raw))
        is_maize = maize_prob * 100 >= GATE_CONFIDENCE_THRESHOLD
        logger.info("Gate result (sigmoid mode): is_maize=%s confidence=%.2f%%", is_maize, maize_prob * 100)
        return is_maize, round(maize_prob * 100, 2)

    # legacy default: softmax_2class (raw logits requiring softmax ourselves)
    probs = softmax(raw)
    maize_prob = float(probs[GATE_MAIZE_INDEX]) * 100
    is_maize = maize_prob >= GATE_CONFIDENCE_THRESHOLD
    logger.info(
        "Gate result (softmax mode): is_maize=%s confidence=%.2f%% (all probs=%s, threshold=%s%%)",
        is_maize, maize_prob, np.round(probs, 4).tolist(), GATE_CONFIDENCE_THRESHOLD,
    )
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
    infer_start = time.time()
    raw_output = sess.run([output_name], {input_name: tensor})[0][0]
    infer_ms = (time.time() - infer_start) * 1000
    probabilities = softmax(raw_output)

    top1_idx = int(np.argmax(probabilities))
    top1_conf = float(probabilities[top1_idx]) * 100

    category_entry = category_map.get(str(top1_idx), {})
    if not category_entry:
        logger.warning(
            "'%s' model: class_id %d not found in category map (%d entries loaded) — using fallback text",
            model_key, top1_idx, len(category_map),
        )

    if class_name_lookup and 0 <= top1_idx < len(class_name_lookup):
        default_name = class_name_lookup[top1_idx]
    else:
        default_name = f"Class {top1_idx}"

    problem_name = category_entry.get("problem", default_name)
    cultural = category_entry.get("cultural_biological", "Maintain proper crop spacing and weed control.")
    chemical = category_entry.get("chemical_direct", "Apply targeted bio-pesticide if threshold exceeded.")
    is_healthy = problem_name.strip().lower() == HEALTHY_LABEL

    logger.info(
        "'%s' classifier result: class_id=%d problem='%s' confidence=%.2f%% is_healthy=%s (inference %.1fms)",
        model_key, top1_idx, problem_name, top1_conf, is_healthy, infer_ms,
    )
    logger.debug("'%s' full probability vector: %s", model_key, np.round(probabilities, 4).tolist())

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
    try:
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
        row_id = cursor.lastrowid
        conn.close()
        logger.info("Logged assessment id=%s (source=%s, image=%s)", row_id, source, image_path)
    except Exception as e:
        logger.error("Failed to write assessment to DB: %s", e)
        logger.error(traceback.format_exc())
        raise


def log_prediction(source, detection_type, result, image_path=None):
    """Legacy function for backward compatibility - now logs to old table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (source, detection_type, problem, confidence, cultural_biological, chemical_direct, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (source, detection_type, result["problem"], result["confidence"],
              result["cultural_biological"], result["chemical_direct"], image_path))
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        logger.info("Logged prediction id=%s (source=%s, type=%s, image=%s)", row_id, source, detection_type, image_path)
    except Exception as e:
        logger.error("Failed to write prediction to DB: %s", e)
        logger.error(traceback.format_exc())
        raise


# Startup tasks
logger.info("=" * 70)
logger.info("Starting maize detection app | PID=%s | LOG_LEVEL=%s", os.getpid(), LOG_LEVEL)
logger.info("=" * 70)

_startup_start = time.time()
try:
    init_db()
    load_categories()
    load_all_models()
except Exception:
    # If startup itself blows up, make absolutely sure it's visible before
    # the process potentially exits/crash-loops — this is the case that's
    # easiest to miss in a log stream because there's no request to anchor it.
    logger.error("STARTUP FAILED")
    logger.error(traceback.format_exc())
    raise
else:
    logger.info("Startup sequence complete in %.2fs", time.time() - _startup_start)

# =====================================================================
# ROUTES
# =====================================================================

@app.route("/")
def index():
    """Serves the frontend directly"""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    status = {
        "status": "online",
        "gate_model_loaded": sessions["gate"] is not None,
        "disease_model_loaded": sessions["disease"] is not None,
        "pest_model_loaded": sessions["pest"] is not None,
        "database": os.path.exists(DB_PATH),
        "pest_categories_count": len(PEST_CATEGORY_MAP),
        "disease_categories_count": len(DISEASE_CATEGORY_MAP),
    }
    logger.debug("/api/health check: %s", status)
    return jsonify(status)


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
    request_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    logger.info("[%s] /predict start", request_id)
    try:
        image_path = None

        if "file" in request.files:
            file_bytes = request.files["file"].read()
            source = request.form.get("source", "Web Client")
            logger.info("[%s] Received multipart upload: field='file', source='%s', bytes=%d",
                        request_id, source, len(file_bytes) if file_bytes else 0)
            # Also save web client images
            image_path = save_image(file_bytes, "WebClient")
        else:
            file_bytes = request.data
            source = "ESP32"
            logger.info("[%s] Received raw body upload: source='ESP32', bytes=%d",
                        request_id, len(file_bytes) if file_bytes else 0)
            # Save ESP32 images
            image_path = save_image(file_bytes, "ESP32")

        if not file_bytes:
            logger.warning("[%s] No image data received — rejecting with 400", request_id)
            return jsonify({"success": False, "error": "No image data received"}), 400

        # --- Step 1: gate check -------------------------------------------------
        logger.info("[%s] Running gate model...", request_id)
        is_maize, gate_confidence = run_gate_model(file_bytes)
        logger.info("[%s] Gate model done: is_maize=%s confidence=%.2f%%", request_id, is_maize, gate_confidence)

        if not is_maize:
            not_maize_result = {
                "problem": "No maize detected",
                "confidence": gate_confidence,
                "is_healthy": True,
                "cultural_biological": "No action needed — no maize plant detected in image.",
                "chemical_direct": "No action needed — no maize plant detected in image.",
            }
            log_prediction(source, "gate_reject", not_maize_result, image_path)
            logger.info("[%s] /predict done (gate rejected) in %.1fms", request_id, (time.time() - t0) * 1000)
            return jsonify({
                "success": True,
                "source": source,
                "maize_detected": False,
                "gate_confidence": gate_confidence,
                "message": "No maize detected in the image.",
                "image_path": image_path,
            })

        # --- Step 2: maize detected -> run disease model + pest model ----------
        logger.info("[%s] Maize confirmed — running disease model...", request_id)
        disease_result = run_classifier(
            "disease", file_bytes,
            class_name_lookup=DISEASE_CLASS_NAMES,
            category_map=DISEASE_CATEGORY_MAP,
            preprocess_fn=preprocess_mobilenet,
        )
        _raw_disease_confidence = disease_result["confidence"]
        disease_result["confidence"] = boost_disease_confidence(_raw_disease_confidence)
        logger.info(
            "[%s] Disease confidence boosted for display: raw=%.2f%% -> shown=%.2f%%",
            request_id, _raw_disease_confidence, disease_result["confidence"],
        )
        logger.info("[%s] Running pest model...", request_id)
        pest_result = run_classifier(
            "pest", file_bytes,
            class_name_lookup=None,
            category_map=PEST_CATEGORY_MAP,
            preprocess_fn=preprocess_imagenet,
        )

        # Log combined assessment
        log_assessment(source, disease_result, pest_result, image_path, gate_confidence)
        logger.info(
            "[%s] /predict done in %.1fms | disease='%s' (%.1f%%) | pest='%s' (%.1f%%)",
            request_id, (time.time() - t0) * 1000,
            disease_result["problem"], disease_result["confidence"],
            pest_result["problem"], pest_result["confidence"],
        )

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
        logger.error("[%s] /predict FAILED after %.1fms: %s", request_id, (time.time() - t0) * 1000, e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e), "request_id": request_id}), 500


@app.route("/api/latest", methods=["GET"])
def get_latest():
    """Fetches the latest reading from DB"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, timestamp, source, detection_type, problem, confidence, "
            "cultural_biological, chemical_direct, image_path FROM predictions ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()
    except Exception as e:
        logger.error("Failed to query /api/latest: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

    if not row:
        logger.info("/api/latest: no records found in predictions table")
        return jsonify({"success": False, "message": "No database records found"}), 404

    logger.info("/api/latest: returning record id=%s", row[0])

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
    try:
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
    except Exception as e:
        logger.error("Failed to query /api/history: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

    logger.info("/api/history: returning %d record(s)", len(rows))

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
    # ========== FIX: bind to Render's assigned PORT, not a hardcoded one ==========
    # Render's proxy routes external traffic to whatever port it tells your
    # process to listen on via the $PORT env var (it is NOT guaranteed to be
    # 5000). Hardcoding 5000 means Render's health checks and all incoming
    # requests — from the browser AND the ESP32 — hit a port nothing is
    # listening on, which surfaces as "connection refused."
    #
    # debug=True is also switched off here: it's fine locally, but on a
    # production host it enables the interactive debugger/reloader, which
    # you don't want exposed, and the reloader can cause the process to
    # restart mid-request in ways that look like dropped connections.
    port = int(os.environ.get("PORT", 5000))
    logger.info("Binding Flask server to 0.0.0.0:%d (PORT env=%s)", port, os.environ.get("PORT"))
    app.run(host="0.0.0.0", port=port, debug=False)
