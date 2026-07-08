import os
import json
import re
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
CONFIDENCE_THRESHOLD = 5.0  # minimum % to list as a detected fault
TOP_K = 5

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
model = None
MODEL_LOAD_ERROR = None


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def _load_categories():
    path = os.path.join(os.path.dirname(__file__), "categories.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


CATEGORY_MAP = _load_categories()
NAME_INDEX: dict[str, dict] = {}

for key, entry in CATEGORY_MAP.items():
    if not isinstance(entry, dict):
        continue
    problem = entry.get("problem")
    if not problem:
        continue
    normalized = _normalize_name(problem)
    NAME_INDEX[normalized] = {
        "class_id": int(key),
        "problem": problem,
        "cultural_biological": entry.get("cultural_biological"),
        "chemical_direct": entry.get("chemical_direct"),
    }


def _load_model_if_needed():
    global model, MODEL_LOAD_ERROR
    if model is not None or MODEL_LOAD_ERROR is not None:
        return
    if YOLO is None:
        MODEL_LOAD_ERROR = (
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        )
        return
    if not os.path.exists(MODEL_PATH):
        MODEL_LOAD_ERROR = (
            f"Model file not found at '{MODEL_PATH}'. "
            "Place your trained best.pt in the project folder and restart."
        )
        return
    try:
        model = YOLO(MODEL_PATH)
    except Exception as exc:
        MODEL_LOAD_ERROR = f"Failed to load model '{MODEL_PATH}': {exc}"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def category_for_class_id(class_id: int, fallback_name: str | None = None) -> dict:
    entry = CATEGORY_MAP.get(str(class_id))
    if isinstance(entry, dict):
        return {
            "class_id": class_id,
            "problem": entry.get("problem") or fallback_name or f"Class {class_id}",
            "cultural_biological": entry.get("cultural_biological"),
            "chemical_direct": entry.get("chemical_direct"),
            "mapped": True,
        }

    if fallback_name:
        normalized = _normalize_name(fallback_name)
        by_name = NAME_INDEX.get(normalized)
        if by_name:
            return {**by_name, "mapped": True}

        for name_key, mapped in NAME_INDEX.items():
            if normalized in name_key or name_key in normalized:
                return {**mapped, "mapped": True}

    return {
        "class_id": class_id,
        "problem": fallback_name or f"Class {class_id}",
        "cultural_biological": None,
        "chemical_direct": None,
        "mapped": False,
    }


def build_prediction(class_id: int, confidence: float, fallback_name: str | None = None) -> dict:
    info = category_for_class_id(class_id, fallback_name=fallback_name)
    return {
        "class_id": info["class_id"],
        "confidence": round(confidence, 2),
        "problem": info["problem"],
        "cultural_biological": info["cultural_biological"],
        "chemical_direct": info["chemical_direct"],
        "mapped": info["mapped"],
        "solutions": {
            "cultural_biological": info["cultural_biological"]
            or "No cultural/biological recommendation found in categories.json.",
            "chemical_direct": info["chemical_direct"]
            or "No chemical/direct recommendation found in categories.json.",
        },
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    _load_model_if_needed()
    return jsonify(
        {
            "status": "ok" if MODEL_LOAD_ERROR is None else "degraded",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
            "categories_count": len(CATEGORY_MAP),
            "error": MODEL_LOAD_ERROR,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    _load_model_if_needed()
    if MODEL_LOAD_ERROR is not None:
        return jsonify({"success": False, "error": MODEL_LOAD_ERROR}), 503

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected for uploading"}), 400

    if not file or not allowed_file(file.filename):
        return jsonify(
            {"success": False, "error": "Allowed file types are png, jpg, jpeg, webp"}
        ), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)

        with Image.open(filepath) as im:
            im.verify()

        results = model(filepath)
        result = results[0]
        probs = result.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf) * 100

        names = getattr(result, "names", {}) or {}

        topk: list[dict] = []
        top5 = getattr(probs, "top5", None)
        top5conf = getattr(probs, "top5conf", None)

        if top5 is not None and top5conf is not None:
            for idx, conf in zip(list(top5), list(top5conf)):
                idx_i = int(idx)
                conf_pct = float(conf) * 100
                fallback = names.get(idx_i) if isinstance(names, dict) else None
                topk.append(build_prediction(idx_i, conf_pct, fallback_name=fallback))
        else:
            fallback = names.get(top1_idx) if isinstance(names, dict) else None
            topk.append(build_prediction(top1_idx, top1_conf, fallback_name=fallback))

        detected_faults = [
            item for item in topk if item["confidence"] >= CONFIDENCE_THRESHOLD
        ]
        if not detected_faults and topk:
            detected_faults = [topk[0]]

        primary = topk[0] if topk else build_prediction(top1_idx, top1_conf)

        return jsonify(
            {
                "success": True,
                "class_id": primary["class_id"],
                "confidence": primary["confidence"],
                "prediction": primary,
                "top_predictions": topk,
                "detected_faults": detected_faults,
                "categories_mapped": sum(1 for item in topk if item["mapped"]),
            }
        )

    except Exception as exc:
        return jsonify({"success": False, "error": f"Inference failed: {exc}"}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
