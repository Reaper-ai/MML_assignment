"""
Plotting utilities for PID experiments.
Compact figures with full axis labels and units for PDF embedding.
"""

import matplotlib.pyplot as plt
import numpy as np
from scripts.utils import PLOTS_DIR, compute_metrics


def _fmt(val: float, decimals: int = 2) -> str:
    return "never" if np.isnan(val) else f"{val:.{decimals}f}"


def plot_multi(results, title, filename, colors):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2))
    
    for name, res in results.items():
        c = colors.get(name, "black")
        axes[0, 0].plot(res["time"], res["position"], color=c, label=name, linewidth=1.2)
        axes[0, 1].plot(res["time"], res["error"], color=c, label=name, linewidth=1)
        axes[1, 0].plot(res["time"], res["control"], color=c, label=name, linewidth=1)
        axes[1, 1].plot(res["time"], res["velocity"], color=c, label=name, linewidth=1)
    
    axes[0, 0].axhline(list(results.values())[0]["target"][0], color="r", linestyle="--", label="Target")
    axes[0, 0].set_title(f"{title} - Position")
    axes[0, 0].set_ylabel("Position (m)")
    
    axes[0, 1].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axes[0, 1].set_title(f"{title} - Error")
    axes[0, 1].set_ylabel("Error (m)")
    
    axes[1, 0].set_title(f"{title} - Control Signal")
    axes[1, 0].set_ylabel("Control Force (N)")
    axes[1, 0].set_xlabel("Time (s)")
    
    axes[1, 1].set_title(f"{title} - Velocity")
    axes[1, 1].set_ylabel("Velocity (m/s)")
    axes[1, 1].set_xlabel("Time (s)")
    
    for ax in axes.flat:
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def plot_comparison_table(results, title, filename, colors):
    fig = plt.figure(figsize=(9, 7.2))
    gs = fig.add_gridspec(3, 2)
    
    ax_pos = fig.add_subplot(gs[0, :])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_ctrl = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    
    metrics_list = []
    for name, res in results.items():
        c = colors.get(name, "black")
        ax_pos.plot(res["time"], res["position"], color=c, label=name, linewidth=1.2)
        ax_err.plot(res["time"], res["error"], color=c, label=name, linewidth=1)
        ax_ctrl.plot(res["time"], res["control"], color=c, label=name, linewidth=1)
        m = compute_metrics(res)
        m["controller"] = name
        metrics_list.append(m)
    
    ax_pos.axhline(list(results.values())[0]["target"][0], color="r", linestyle="--", label="Target")
    ax_pos.set_title(title)
    ax_pos.set_ylabel("Position (m)")
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.3)
    
    ax_err.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax_err.set_ylabel("Error (m)")
    ax_err.set_xlabel("Time (s)")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)
    
    ax_ctrl.set_ylabel("Control Force (N)")
    ax_ctrl.set_xlabel("Time (s)")
    ax_ctrl.legend(fontsize=8)
    ax_ctrl.grid(True, alpha=0.3)
    
    table_data = [["Controller", "Rise Time (s)", "Overshoot (%)", "Settling Time (s)", "SSE (m)"]]
    for m in metrics_list:
        table_data.append([
            m["controller"],
            _fmt(m["rise_time"], 2),
            f"{max(0.0, m['overshoot_pct']):.1f}",
            _fmt(m["settling_time"], 2),
            f"{m['steady_state_error']:.4f}"
        ])
    
    table = ax_table.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.5)
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor("#4CAF50")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "white")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def plot_windup(results, filename):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for name, color, label in [("PID_windup", "red", "Windup"), ("PID_no_windup", "blue", "Anti-windup")]:
        axes[0].plot(results[name]["time"], results[name]["position"], color=color, label=label, linewidth=1.2)
        axes[1].plot(results[name]["time"], results[name]["control"], color=color, label=label, linewidth=1)
    
    axes[0].axhline(1.0, color="k", linestyle="--", label="Target (1m)")
    axes[0].set_title("Position (Windup vs Anti-windup)")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_xlabel("Time (s)")
    
    axes[1].axhline(30, color="k", linestyle="--", label="Limit (30N)")
    axes[1].set_title("Control Signal")
    axes[1].set_ylabel("Control Force (N)")
    axes[1].set_xlabel("Time (s)")
    
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def plot_noise(results, filename):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for name, color, label in [("PID_noisy", "red", "Noisy D on error"), ("PID_filtered", "blue", "Filtered D (alpha=0.1)")]:
        axes[0].plot(results[name]["time"], results[name]["position"], color=color, label=label, linewidth=1.2, alpha=0.85)
        axes[1].plot(results[name]["time"], results[name]["control"], color=color, label=label, linewidth=1, alpha=0.75)
    
    axes[0].axhline(1.0, color="k", linestyle="--", label="Target (1m)")
    axes[0].set_title("Position (±1mm Sensor Noise)")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_xlabel("Time (s)")
    
    axes[1].set_title("Control Signal (Noise Amplification)")
    axes[1].set_ylabel("Control Force (N)")
    axes[1].set_xlabel("Time (s)")
    
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def plot_kick(results, filename):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for name, color, label in [("PID_kick", "red", "D on Error (kick)"), ("PID_no_kick", "blue", "D on PV (no kick)")]:
        axes[0].plot(results[name]["time"], results[name]["position"], color=color, label=label, linewidth=1.2)
        axes[1].plot(results[name]["time"], results[name]["control"], color=color, label=label, linewidth=1)
    
    axes[0].plot(results["PID_kick"]["time"], results["PID_kick"]["target"], "k--", label="Target (1->2m)")
    axes[0].set_title("Position (Target Step at t=3s)")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_xlabel("Time (s)")
    
    axes[1].set_title("Control Signal (Derivative Kick)")
    axes[1].set_ylabel("Control Force (N)")
    axes[1].set_xlabel("Time (s)")
    
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def plot_zoh_sweep(results, filename):
    ts_keys = sorted(
        [k for k in results.keys() if k.startswith("PID_Ts_")],
        key=lambda x: int(x.split("_")[-1].replace("ms", ""))
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(ts_keys)))
    
    for idx, key in enumerate(ts_keys):
        res = results[key]
        label = key.replace("PID_Ts_", "")
        axes[0].plot(res["time"], res["position"], color=colors[idx], label=label, linewidth=1.2)
        axes[1].plot(res["time"], res["control"], color=colors[idx], label=label, linewidth=1)
    
    axes[0].axhline(1.0, color="k", linestyle="--", label="Target")
    axes[0].set_title("Position (ZOH Update Sweep)")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylim(-0.5, 3.5)
    
    axes[1].set_title("Control Signal (ZOH Hold)")
    axes[1].set_ylabel("Control Force (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(-150, 200)
    
    for ax in axes:
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def plot_zn_results(results, Ku, Tu, filename):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8))
    for name, res in results.items():
        axes[0, 0].plot(res["time"], res["position"], label=name, linewidth=1.2)
        axes[0, 1].plot(res["time"], res["control"], label=name, linewidth=1)
        axes[1, 0].plot(res["time"], res["position"], label=name, linewidth=1.2)
    
    axes[0, 0].axhline(1.0, color="k", linestyle="--", label="Target")
    axes[0, 0].set_title(f"Z-N Tuning on 1/(s+1)^3 (Ku={Ku:.1f}, Tu={Tu:.2f}s)")
    axes[0, 0].set_ylabel("Output y(t)")
    axes[0, 0].set_xlabel("Time (s)")
    
    axes[0, 1].set_title("Control Signal")
    axes[0, 1].set_ylabel("Control Input u(t)")
    axes[0, 1].set_xlabel("Time (s)")
    
    axes[1, 0].axhline(1.0, color="k", linestyle="--", label="Target")
    axes[1, 0].set_xlim(0, 5)
    axes[1, 0].set_title("Zoom: First 5s")
    axes[1, 0].set_ylabel("Output y(t)")
    axes[1, 0].set_xlabel("Time (s)")
    
    for ax in axes.flat[:-1]:
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
    axes[1, 1].axis("off")
    table_data = [["Method", "Kp", "Ki", "Kd", "OS%", "Settle(s)", "SSE"]]
    for name, res in results.items():
        m = compute_metrics(res)
        p = res["params"]
        table_data.append([
            name,
            f"{p['Kp']:.2f}",
            f"{p['Ki']:.2f}",
            f"{p['Kd']:.2f}",
            f"{max(0.0, m['overshoot_pct']):.1f}",
            _fmt(m["settling_time"], 1),
            f"{m['steady_state_error']:.4f}"
        ])
    table = axes[1, 1].table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor("#4CAF50")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "white")
                
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def plot_final_comparison(res_basic, res_robust, filename):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].plot(res_basic["time"], res_basic["position"], "r-", label="Basic PID", linewidth=1.2)
    axes[0].plot(res_robust["time"], res_robust["position"], "b-", label="Robust PID", linewidth=1.2)
    axes[0].plot(res_basic["time"], res_basic["target"], "k--", label="Target", linewidth=1)
    axes[0].set_title("Position: Basic vs Robust PID")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_xlabel("Time (s)")
    
    axes[1].plot(res_basic["time"], res_basic["control"], "r-", label="Basic PID", linewidth=1, alpha=0.75)
    axes[1].plot(res_robust["time"], res_robust["control"], "b-", label="Robust PID", linewidth=1, alpha=0.85)
    axes[1].axhline(30, color="k", linestyle="--", label="±30N Limit", linewidth=1)
    axes[1].axhline(-30, color="k", linestyle="--", linewidth=1)
    axes[1].set_title("Control Signal")
    axes[1].set_ylabel("Control Force (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(-40, 40)
    
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()