# ============================================================
# Indoor Localization — Full Project Pipeline
# ============================================================
# Usage:
#   make data          -> run full data pipeline
#   make train         -> train federated GNN
#   make baselines     -> run all baselines (kNN, MLP, FedMLP, CentralGNN)
#   make eval          -> baselines + plots
#   make plots         -> generate all visualizations
#   make scalability   -> run scalability experiments
#   make all           -> data + train + eval (full pipeline)
#
#   make clean_data    -> remove interim + processed data
#   make clean_results -> remove results
#   make raw           -> sanity check raw data
#   make clean         -> run cleaning + ENU transform
#   make domains       -> split into domains
#   make rps           -> build RPs (train only)
#   make graphs        -> build RP graphs (train only)
#   make samples       -> build train/val samples
# ============================================================

PYTHON = python

# Detect OS (Windows_NT for Windows, otherwise assume Unix)
ifeq ($(OS),Windows_NT)
    RM = powershell -Command "Remove-Item -Recurse -Force"
else
    RM = rm -rf
endif

# ===========================
#  DATA PIPELINE
# ===========================

# Step 0 — Inspect raw data
raw:
	$(PYTHON) -m src.data.load_raw

# Step 1 — Clean + preprocess
clean:
	$(PYTHON) -m src.data.clean_preprocess

# Step 2 — Build domains
domains:
	$(PYTHON) -m src.data.build_domains

# Step 3 — Build RPs (train only)
rps:
	$(PYTHON) -m src.data.build_rps

# Step 4 — Build graphs (train only)
graphs:
	$(PYTHON) -m src.data.build_graphs

# Step 5 — Build samples (train + val)
samples:
	$(PYTHON) -m src.data.build_samples

# Full data pipeline
data: clean domains rps graphs samples
	@echo "========================================"
	@echo " Data pipeline completed successfully!"
	@echo "========================================"

# ===========================
#  TRAINING
# ===========================

# Train the federated GNN model
train:
	$(PYTHON) run_fl_experiment.py

# ===========================
#  BASELINES
# ===========================

# Run all baselines (kNN, Centralized MLP, Federated MLP, Centralized GNN)
baselines:
	$(PYTHON) scripts/run_baselines.py

# ===========================
#  EVALUATION & PLOTS
# ===========================

# Generate all plots from results/ JSON files
plots:
	$(PYTHON) scripts/plot_results.py --plots all

# Run baselines + generate plots
eval: baselines plots
	@echo "========================================"
	@echo " Evaluation completed!"
	@echo " Results: results/"
	@echo " Plots:   results/plots/"
	@echo "========================================"

# ===========================
#  SCALABILITY
# ===========================

# Run all scalability experiments
scalability:
	$(PYTHON) scripts/run_scalability.py --experiments all

# Run specific scalability experiment (usage: make scale-domains)
scale-domains:
	$(PYTHON) scripts/run_scalability.py --experiments domains

scale-rps:
	$(PYTHON) scripts/run_scalability.py --experiments rps

scale-queries:
	$(PYTHON) scripts/run_scalability.py --experiments queries

scale-rounds:
	$(PYTHON) scripts/run_scalability.py --experiments rounds

# ===========================
#  FULL PIPELINE
# ===========================

# Everything: data -> train -> baselines -> plots
all: data train eval
	@echo "========================================"
	@echo " Full pipeline completed!"
	@echo "========================================"

# ===========================
#  CLEANUP
# ===========================

clean_data:
	$(RM) data/interim/* 2> NUL || true
	$(RM) data/processed/* 2> NUL || true
	@echo "All interim and processed data removed."

clean_results:
	$(RM) results/* 2> NUL || true
	@echo "All results removed."

clean_all: clean_data clean_results
	@echo "All generated files removed."

.PHONY: raw clean domains rps graphs samples data train baselines plots eval scalability all clean_data clean_results clean_all