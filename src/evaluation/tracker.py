# src/evaluation/tracker.py
#
# Tracks training and evaluation metrics across FL rounds.
# Saves results to JSON for plotting/comparison.

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class ExperimentTracker:
    """
    Tracks per-round training losses and evaluation metrics for FL experiments.
    
    Usage:
        tracker = ExperimentTracker("fedgnn_attention", results_dir="results")
        tracker.log_round(round_num=1, client_losses={"0_0": 0.05, "0_1": 0.06}, eval_metrics={...})
        tracker.save()
    """

    def __init__(self, experiment_name: str, results_dir: str = "results", config: Optional[dict] = None):
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            "experiment_name": experiment_name,
            "config": config or {},
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rounds": [],
        }

    def log_round(
        self,
        round_num: int,
        client_losses: Dict[str, float],
        eval_metrics: Optional[Dict[str, float]] = None,
        round_time: Optional[float] = None,
    ):
        """Log metrics for one FL round."""
        entry = {
            "round": round_num,
            "client_losses": client_losses,
            "avg_loss": sum(client_losses.values()) / max(len(client_losses), 1),
            "eval_metrics": eval_metrics or {},
            "round_time_sec": round_time,
        }
        self.history["rounds"].append(entry)

    def save(self):
        """Save full experiment history to JSON."""
        self.history["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.history["total_rounds"] = len(self.history["rounds"])
        
        out_path = self.results_dir / f"{self.experiment_name}.json"
        with open(out_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Results saved to {out_path}")
        return out_path

    def get_loss_history(self) -> List[float]:
        """Return list of avg losses per round (for quick plotting)."""
        return [r["avg_loss"] for r in self.history["rounds"]]

    def get_client_loss_history(self, client_id: str) -> List[float]:
        """Return loss trajectory for one client."""
        return [
            r["client_losses"].get(client_id, float("nan"))
            for r in self.history["rounds"]
        ]

    def get_eval_history(self, metric_name: str) -> List[float]:
        """Return trajectory of a specific eval metric across rounds."""
        return [
            r["eval_metrics"].get(metric_name, float("nan"))
            for r in self.history["rounds"]
        ]
