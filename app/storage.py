from pathlib import Path

BASE_DIR = Path("/testbed/ml-explainability-pack")
MODEL_DIR = BASE_DIR / "models"
EXPLAIN_DIR = BASE_DIR / "explanations"
UPLOAD_DIR = BASE_DIR / "uploads"

MODEL_DIR.mkdir(exist_ok=True)
EXPLAIN_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
