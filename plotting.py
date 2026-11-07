"""
Plotting utilities for PID experiments.
All matplotlib plotting code separated from experiment logic.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Any

from utils import PLOTS_DIR, compute_metrics


def plot_multi(results: Dict[str, Dict[str, np.ndarray]], title: str,
               filename: str, colors: Dict[str, str]) -> None:
    """Plot comparison of multiple controllers (4-panel)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for name, res in results.items():
        c = colors.get(name, "black")
        axes[0, 0].plot(res["time"], res["position"], color=c, label=name, linewidth=1.5)
        axes[0, 1].plot(res["time"], res["error"], color=c, label=name, linewidth=1)
        axes[1, 0].plot(res["time"], res["control"], color=c, label=name, linewidth=1)
        axes[1, 1].plot(res["time"], res["velocity"], color=c, label=name, linewidth=1)
    
    target_val = list(results.values())[0]["target"][0]
    axes[0, 0].axhline(target_val, color="r", linestyle="--", label="Target")
    axes[0, 0].set_ylabel("Position (m)")
    axes[0, 0].set_title(f"{title} - Position")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axes[0, 1].set_ylabel("Error (m)")
    axes[0, 1].set_title(f"{title} - Error")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_ylabel("Control Force (N)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_title(f"{title} - Control Signal")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_ylabel("Velocity (m/s)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_title(f"{title} - Velocity")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / filename
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {plot_path}")


def plot_comparison_table(results: Dict[str, Dict[str, np.ndarray]],
                          title: str, filename: str, colors: Dict[str, str]) -> None:
    """Plot position, error, control with metrics table."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2)
    
    ax_pos = fig.add_subplot(gs[0, :])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_ctrl = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    
    metrics_list = []
    for name, res in results.items():
        c = colors.get(name, "black")
        ax_pos.plot(res["time"], res["position"], color=c, label=name, linewidth=1.5)
        ax_err.plot(res["time"], res["error"], color=c, label=name, linewidth=1)
        ax_ctrl.plot(res["time"], res["control"], color=c, label=name, linewidth=1)
        m = compute_metrics(res)
        m["controller"] = name
        metrics_list.append(m)
    
    target_val = list(results.values())[0]["target"][0]
    ax_pos.axhline(target_val, color="r", linestyle="--", label="Target")
    ax_pos.set_ylabel("Position (m)")
    ax_pos.set_title(title)
    ax_pos.legend()
    ax_pos.grid(True, alpha=0.3)
    
    ax_err.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax_err.set_ylabel("Error (m)")
    ax_err.set_xlabel("Time (s)")
    ax_err.legend()
    ax_err.grid(True, alpha=0.3)
    
    ax_ctrl.set_ylabel("Control Force (N)")
    ax_ctrl.set_xlabel("Time (s)")
    ax_ctrl.legend()
    ax_ctrl.grid(True, alpha=0.3)
    
    table_data = [
        ["Controller", "Rise Time (s)", "Overshoot (%)", "Settling Time (s)", "Steady-State Error"]
    ]
    for m in metrics_list:
        table_data.append([
            m["controller"],
            f"{m['rise_time']:.2f}",
            f"{m['overshoot_pct']:.1f}",
            f"{m['settling_time']:.2f}",
            f"{m['steady_state_error']:.4f}"
        ])
    
    table = ax_table.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor("#4CAF50")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "white")
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / filename
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {plot_path}")
    
    print("\n  Metrics Table:")
    print(f"  {'Controller':<12} {'Rise(s)':>8} {'OS(%)':>8} {'Settle(s)':>10} {'SSE':>10}")
    print("  " + "-" * 50)
    for m in metrics_list:
        print(f"  {m['controller']:<12} {m['rise_time']:>8.2f} {m['overshoot_pct']:>8.1f} "
              f"{m['settling_time']:>10.2f} {m['steady_state_error']:>10.4f}")


def plot_windup(results: Dict[str, Dict[str, np.ndarray]], filename: str) -> None:
    """Plot windup vs anti-windup on same graph (2 panels: position + control) - horizontal layout."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    
    for name, color, label in [("PID_windup", "red", "Windup (problem)"), 
                                ("PID_antiwindup", "blue", "Anti-windup (fixed)")]:
        res = results[name]
        axes[0].plot(res["time"], res["position"], color=color, label=label, linewidth=1.5)
    axes[0].axhline(1.0, color="k", linestyle="--", label="Target (1m)", linewidth=1)
    axes[0].axhline(30/20, color="gray", linestyle=":", label="Max reachable (1.5m)", linewidth=1)
    axes[0].set_ylabel("Position (m)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title("(a) Integral Windup: 30N limit, Kp=100, Ki=120")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    for name, color, label in [("PID_windup", "red", "Windup"), 
                                ("PID_antiwindup", "blue", "Anti-windup")]:
        res = results[name]
        axes[1].plot(res["time"], res["control"], color=color, label=label, linewidth=1)
    axes[1].axhline(30, color="k", linestyle="--", label="Limit (30N)", linewidth=1)
    axes[1].set_ylabel("Control Force (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(0, 40)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / filename
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {plot_path}")


def plot_noise(results: Dict[str, Dict[str, np.ndarray]], filename: str) -> None:
    """Plot noisy vs filtered on same graph (2 panels: position + control) - horizontal layout."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    
    for name, color, label in [("PID_noisy", "red", "Noisy sensor (Kd=50)"), 
                                ("PID_filtered", "blue", "Filtered D on PV (fixed)")]:
        res = results[name]
        axes[0].plot(res["time"], res["position"], color=color, label=label, linewidth=1.5, alpha=0.8)
    axes[0].axhline(1.0, color="k", linestyle="--", label="Target (1m)", linewidth=1)
    axes[0].set_ylabel("Position (m)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title("(b) Sensor Noise: ±1mm, Kd=50")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    for name, color, label in [("PID_noisy", "red", "Noisy D on error"), 
                                ("PID_filtered", "blue", "Filtered D on PV (alpha=0.1)")]:
        res = results[name]
        axes[1].plot(res["time"], res["control"], color=color, label=label, linewidth=1, alpha=0.7)
    axes[1].set_ylabel("Control Force (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("(b) Control Signal: Noise Amplification")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / filename
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {plot_path}")


def plot_kick(results: Dict[str, Dict[str, np.ndarray]], filename: str) -> None:
    """Plot derivative kick vs fixed on same graph (2 panels: position + control) - horizontal layout."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    
    for name, color, label in [("PID_kick", "red", "Derivative on Error (kick)"), 
                                ("PID_no_kick", "blue", "Derivative on PV (no kick)")]:
        res = results[name]
        axes[0].plot(res["time"], res["position"], color=color, label=label, linewidth=1.5)
    axes[0].plot(res["time"], res["target"], "k--", label="Target (step at 3s)", linewidth=1)
    axes[0].set_ylabel("Position (m)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title("(c) Derivative Kick: Target step 1->2 at t=3s")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    for name, color, label in [("PID_kick", "red", "D on Error"), 
                                ("PID_no_kick", "blue", "D on Process Variable")]:
        res = results[name]
        axes[1].plot(res["time"], res["control"], color=color, label=label, linewidth=1)
    axes[1].set_ylabel("Control Force (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("(c) Control Signal: Kick Comparison")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / filename
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {plot_path}")


def plot_zoh_sweep(results: Dict[str, Dict[str, np.ndarray]], title: str, filename: str) -> None:
    """Plot ZOH sweep results."""
    ts_keys = [k for k in results.keys() if k.startswith("PID_Ts_")]
    ts_keys.sort(key=lambda x: int(x.split("_")[-1].replace("ms", "")))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.Set1(np.linspace(0, 1, len(ts_keys)))
    
    for idx, key in enumerate(ts_keys):
        res = results[key]
        ts_label = key.replace("PID_Ts_", "")
        c = colors[idx]
        axes[0].plot(res["time"], res["position"], color=c, label=ts_label, linewidth=1.5)
        axes[1].plot(res["time"], res["control"], color=c, label=ts_label, linewidth=1)
    
    axes[0].axhline(1.0, color="k", linestyle="--", label="Target")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title(f"{title} - Position")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel("Control Force (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(f"{title} - Control Signal")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / filename
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {plot_path}")