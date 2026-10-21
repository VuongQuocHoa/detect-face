"""Application configuration and filesystem initialization."""

import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = str(BASE_DIR / "data" / "chroma")
INSIGHTFACE_MODEL_ROOT = str(BASE_DIR / "data" / "models")
# Ngưỡng tương đồng Cosine tối thiểu để khớp (0.0 đến 1.0, ví dụ 0.50 = 50% độ khớp tương đồng)
SIMILARITY_THRESHOLD = 0.50
INSIGHTFACE_MODEL_NAME = "buffalo_s"
DETECT_SIZE = (320, 320)
ENABLE_OPENVINO = True
# Device options: "GPU" (Intel UHD / Iris Xe Graphics), "CPU" (Intel Raptor Lake VNNI), or "AUTO"
OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "GPU")
# Intel UHD GPU runs at peak accuracy and latency (~2.9-3.0ms) using native FP16.
# Set OPENVINO_USE_INT8=True only on systems with dedicated VNNI CPU inference.
OPENVINO_USE_INT8 = os.getenv("OPENVINO_USE_INT8", "False").lower() in ("true", "1", "yes")
OPENVINO_CACHE_DIR = str(BASE_DIR / "data" / "cache")

FACES_DIR = str(BASE_DIR / "data" / "faces")

os.makedirs(str(BASE_DIR), exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(INSIGHTFACE_MODEL_ROOT, exist_ok=True)
os.makedirs(OPENVINO_CACHE_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)


