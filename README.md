# PID Controller Experiments

> **📄 [Full Technical Report](tex/PID.pdf)** — Detailed mathematical analysis, derivations, and experimental results.

This repository contains a comprehensive set of PID controller experiments demonstrating various PID configurations, real-world problems, and solutions.

## Quick Start

```bash
# Install uv (if not already installed)
pip install uv

# Install dependencies and sync environment
uv sync

# Run all experiments
python main.py

# Run specific experiment
python main.py --exp 1    # P-only sweep
python main.py --exp 2    # PI, PD, PID comparison
python main.py --exp 3    # Detailed comparison with metrics table
python main.py --exp 4    # Real-world problems (windup, noise, kick, slow update)
python main.py --exp 5    # Ziegler-Nichols tuning on 1/(s+1)^3
python main.py --exp 6    # Final comparison: Basic PID vs Robust PID

# Override parameters
python main.py --Kp 200 --Ki 80 --Kd 15 --t_max 10
python main.py --exp 1 --Kp_values 10 50 100 200 500
```

## Project Structure

```
.
├── main.py              # CLI entry point
├── scripts/             # All experiment modules
│   ├── __init__.py
│   ├── experiments.py   # 6 experiment implementations
│   ├── plotting.py      # All matplotlib plotting (horizontal layout)
│   ├── utils.py         # Shared constants, save/load, metrics
│   ├── controllers.py   # Clean compositional PID architecture
│   └── system.py        # Spring-mass-damper system
├── experiment_data/     # Saved simulation data (.pkl)
├── experiment_plots/    # Generated plots (.png)
├── pyproject.toml
└── uv.lock
```

## Experiments Overview

### Experiment 1: P-only Sweep (`--exp 1`)
Tests proportional controller with varying Kp values (10, 50, 100, 200, 500).
- Shows how increasing Kp reduces steady-state error but increases overshoot
- Output: `exp1_p_only.png`

### Experiment 2: PI, PD, PID Comparison (`--exp 2`)
Compares four controller types with same gains.
- P, PI, PD, PID on spring-mass-damper system
- Output: `exp2_pi_pd_pid.png`

### Experiment 3: Detailed Comparison (`--exp 3`)
Same as Exp 2 but with metrics table (rise time, overshoot, settling time, SSE).
- Output: `exp3_comparison.png`

### Experiment 4: Real-World Problems (`--exp 4`)
Four real-world PID problems and their fixes:

| Problem | Controller | Fix Applied |
|---------|------------|-------------|
| (a) Integral Windup | `PID_no_windup` | Conditional integration |
| (b) Sensor Noise | `PID_filtered` | D on ERROR + filter (α=0.1) |
| (c) Derivative Kick | `PID_no_kick` | D on PV + filter |
| (d) Slow Update (ZOH) | `PID` (standard) | ZOH sweep Ts=1..100ms |

Outputs: `exp4a_windup.png`, `exp4b_noise.png`, `exp4c_kick.png`, `exp4d_zoh.png`

### Experiment 5: Ziegler-Nichols Tuning (`--exp 5`)
Tunes PID for system 1/(s+1)³ using Ziegler-Nichols method.
1. Finds ultimate gain Ku (analytical: Ku=8.00)
2. Measures ultimate period Tu (analytical: Tu=3.63s)
3. Applies Z-N recipe table
4. Manual improvement (reduce Kp, adjust Kd)

Output: `exp5_zn_tuning.png`

### Experiment 6: Final Comparison (`--exp 6`)
Basic PID vs Robust PID with all 4 fixes active simultaneously:
- 30N actuator limit (windup)
- ±1mm sensor noise (Kd=50)
- Target step 1→2m at t=5s (kick)
- ZOH with Ts=50ms (slow update)

Output: `exp6_final_comparison.png`

## Controller Architecture

### Atomic Building Blocks
| Function | Purpose |
|----------|---------|
| `P(Kp, error)` | Proportional |
| `I(Ki, error, dt, integral)` | Pure integral |
| `I_no_windup(Ki, error, dt, integral, min, max)` | Integral + anti-windup |
| `D(Kd, error, prev_error, dt)` | Derivative on error |
| `D_filtered(Kd, error, prev_error, dt, alpha, prev_filtered)` | Filtered D on error |
| `D_filtered_pv(Kd, pv, prev_pv, dt, alpha, prev_filtered)` | Filtered D on PV |
| `Clamp(value, min, max)` | Output saturation |

### Composed Controllers
| Controller | Composition | Fixes |
|------------|-------------|-------|
| `PID` | `P + I + D` | Standard |
| `PID_no_windup` | `P + I_no_windup + D` | Anti-windup |
| `PID_filtered` | `P + I + D_filtered` | D on ERROR + filter |
| `PID_no_kick` | `P + I + D_filtered_pv` | D on PV + filter |
| `PID_robust_all` | `P + I_no_windup + D_filtered_pv` | **All 4 fixes** |

## Output Files

### Plots (`experiment_plots/`)
- `exp1_p_only.png` - P-only sweep
- `exp2_pi_pd_pid.png` - PI/PD/PID comparison
- `exp3_comparison.png` - Metrics table + plots
- `exp4a_windup.png` - Windup vs anti-windup
- `exp4b_noise.png` - Noisy vs filtered
- `exp4c_kick.png` - Kick vs no-kick
- `exp4d_zoh.png` - ZOH sweep
- `exp5_zn_tuning.png` - Z-N tuning results
- `exp6_final_comparison.png` - Basic vs Robust PID

### Data (`experiment_data/`)
- `.pkl` files with raw simulation data for each experiment

## Reproducing Results

All experiments use fixed random seed (`np.random.seed(0)`) for reproducibility.

```bash
# Full reproduction
python main.py

# Specific experiment with custom params
python main.py --exp 4 --Kp 150 --Ki 100 --Kd 20 --t_max 8
```

## System Under Test

The test system is a **spring-mass-damper**:
- Mass: 1 kg
- Damping: 10 N·s/m
- Spring: 20 N/m
- Target: 1m (step input)

Equation: `m·ẍ + c·ẋ + k·x = u`
- Holding 1m requires 20N (k·x = 20·1 = 20N)
- 30N actuator limit allows 1.5m max position