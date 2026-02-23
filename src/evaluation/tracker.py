# src/evaluation/tracker.py
#
# Tracks training and evaluation metrics across FL rounds.
# Saves results to JSON for plotting/comparison.

import json
import os
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

    def __init__(
        self,
        experiment_name: str,
        results_dir: str = "results",
        config: Optional[dict] = None,
        autosave: bool = True,
        tensorboard_log_dir: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.autosave = autosave
        self.out_path = self.results_dir / f"{self.experiment_name}.json"
        self.tb_writer = None
        if tensorboard_log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = Path(tensorboard_log_dir)
                if not tb_dir.is_absolute():
                    tb_dir = Path(tensorboard_log_dir)
                tb_dir = tb_dir / self.experiment_name
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            except Exception as exc:
                print(f"[tracker] TensorBoard disabled: {exc}")
        
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
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("loss/avg", entry["avg_loss"], round_num)
            for client_id, loss in client_losses.items():
                self.tb_writer.add_scalar(f"loss/client_{client_id}", loss, round_num)
            for key, value in (eval_metrics or {}).items():
                self.tb_writer.add_scalar(f"eval/{key}", value, round_num)
        if self.autosave:
            self.save(finalize=False)

    def save(self, finalize: bool = True):
        """Save experiment history to JSON (finalize adds end_time)."""
        self.history["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.history["total_rounds"] = len(self.history["rounds"])
        if finalize:
            self.history["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        tmp_path = self.out_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.history, f, indent=2)
        os.replace(tmp_path, self.out_path)
        if finalize:
            if self.tb_writer is not None:
                self.tb_writer.flush()
                self.tb_writer.close()
            print(f"Results saved to {self.out_path}")
        return self.out_path

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
