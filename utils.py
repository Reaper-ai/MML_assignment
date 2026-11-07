"""
Shared constants and utilities for PID experiments.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict
import numpy as np


DATA_DIR = Path("experiment_data")
PLOTS_DIR = Path("experiment_plots")

for d in (DATA_DIR, PLOTS_DIR):
    d.mkdir(exist_ok=True)


def save_data(data: Any, filename: str) -> None:
    """Save data to JSON or pickle based on extension."""
    filepath = DATA_DIR / filename
    if filename.endswith(".json"):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif filename.endswith(".pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError("Use .json or .pkl extension")


def load_data(filename: str) -> Any:
    """Load data from JSON or pickle based on extension."""
    filepath = DATA_DIR / filename
    if filename.endswith(".json"):
        with open(filepath, "r") as f:
            return json.load(f)
    elif filename.endswith(".pkl"):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Use .json or .pkl extension")


def compute_metrics(result: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute rise time, overshoot, settling time, steady-state error."""
    time = result["time"]
    position = result["position"]
    target = result["target"][0]
    error = result["error"]
    
    steady_state_threshold = 0.02 * abs(target)
    
    idx_10 = np.where(position >= 0.1 * target)[0]
    idx_90 = np.where(position >= 0.9 * target)[0]
    rise_time = time[idx_90[0]] - time[idx_10[0]] if len(idx_10) > 0 and len(idx_90) > 0 else np.nan
    
    max_pos = np.max(position)
    overshoot = ((max_pos - target) / target * 100) if target != 0 else 0.0
    
    settled = np.where(np.abs(error) <= steady_state_threshold)[0]
    settling_time = time[settled[0]] if len(settled) > 0 else np.nan
    
    ss_error = error[-1]
    
    return {
        "rise_time": float(rise_time),
        "overshoot_pct": float(overshoot),
        "settling_time": float(settling_time),
        "steady_state_error": float(ss_error)
    }