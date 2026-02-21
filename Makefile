# ============================================================
# Data Preparation Pipeline for Indoor Localization Project
# ============================================================
# Usage:
#   make data        -> run full data pipeline
#   make clean_data  -> remove interim + processed data
#   make raw         -> sanity check raw data
#   make clean       -> run cleaning + ENU transform
#   make domains     -> split into domains
#   make rps         -> build RPs (train only)
#   make graphs      -> build RP graphs (train only)
#   make samples     -> build train/val samples
# ============================================================

PYTHON = python

# Detect OS (Windows_NT for Windows, otherwise assume Unix)
ifeq ($(OS),Windows_NT)
    RM = powershell -Command "Remove-Item -Recurse -Force"
else
    RM = rm -rf
endif

# -----------------------------
# Step 0 — Inspect raw data
# -----------------------------
raw:
	$(PYTHON) -m src.data.load_raw

# -----------------------------
# Step 1 — Clean + preprocess
# -----------------------------
clean:
	$(PYTHON) -m src.data.clean_preprocess

# -----------------------------
# Step 2 — Build domains
# -----------------------------
domains:
	$(PYTHON) -m src.data.build_domains

# -----------------------------
# Step 3 — Build RPs (train only)
# -----------------------------
rps:
	$(PYTHON) -m src.data.build_rps

# -----------------------------
# Step 4 — Build graphs (train only)
# -----------------------------
graphs:
	$(PYTHON) -m src.data.build_graphs

# -----------------------------
# Step 5 — Build samples (train + val)
# -----------------------------
samples:
	$(PYTHON) -m src.data.build_samples

# -----------------------------
# Full pipeline
# -----------------------------
data: clean domains rps graphs samples
	@echo "========================================"
	@echo " Data pipeline completed successfully!"
	@echo "========================================"

# -----------------------------
# Clean all generated data
# -----------------------------
clean_data:
	$(RM) data/interim/* 2> NUL || true
	$(RM) data/processed/* 2> NUL || true
	@echo "All interim and processed data removed."