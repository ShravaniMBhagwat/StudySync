"""
config.py — Central configuration for StudySync.

All tunable constants live here. Import this module wherever settings
are needed; never hard-code paths or values in other files.
"""

import os

# ── Base directories ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER   = os.path.join(BASE_DIR, "uploads")
MODEL_DIR       = os.path.join(BASE_DIR, "data", "models")
DATASET_DIR     = os.path.join(BASE_DIR, "data", "datasets")
TRAINING_DIR    = os.path.join(BASE_DIR, "data", "training")
CHARTS_DIR      = os.path.join(BASE_DIR, "static", "charts")

# ── File upload settings ──────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"pdf", "txt"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB maximum request size

# ── ML / model settings ───────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 2)
TEST_SPLIT_SIZE    = 0.20   # 80 / 20 train-test split
CV_FOLDS           = 5      # Stratified k-fold cross-validation
KMEANS_CLUSTERS    = 5      # Default number of clusters

# ── Subject labels (edit this list to match your course subjects) ─────────────
# Used by NoteMap classifier; folder names under data/training/ must match.
SUBJECT_LABELS = [
    "Data Structures",
    "Database Management Systems",
    "Operating Systems",
    "Computer Networks",
    "Machine Learning",
    "Artificial Intelligence",
]

# ── Flask settings ────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("STUDYSYNC_SECRET_KEY", "change-this-in-production")
DEBUG      = os.environ.get("FLASK_DEBUG", "true").lower() == "true"

# ── Ensure required runtime directories exist ─────────────────────────────────
for _dir in (UPLOAD_FOLDER, MODEL_DIR, CHARTS_DIR):
    os.makedirs(_dir, exist_ok=True)
