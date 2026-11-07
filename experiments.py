"""
Experiment scaffold for PID controller analysis.
Experiment conduction logic (simulation, running experiments).
Plotting is delegated to plotting.py.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

from system import Spring
from controllers import (
    Propotional, PI, PD, PID,
    PID_antiwindup, PID_no_kick, PID_filtered,
    Clamp
)
from plotting import (
    plot_multi, plot_comparison_table,
    plot_windup, plot_noise, plot_kick, plot_zoh_sweep
)
from utils import DATA_DIR, PLOTS_DIR, compute_metrics, save_data, load_data


def simulate_system(C: Dict, controller_fn: Callable, controller_args: Dict,
                    target: float = 1.0, add_noise: bool = False,
                    noise_std: float = 0.001, max_force: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Run a single simulation with the given controller.
    
    Returns dict with time, position, velocity, error, control signal arrays.
    Noise only affects controller measurement, not the actual plant physics.
    """
    spring = Spring(k=20.0, d=10.0, m=1.0, x0=0.0)
    
    dt = C["dt"]
    t_max = C["t_max"]
    n_steps = int(t_max / dt) + 1
    time = np.linspace(0, t_max, n_steps)
    position = np.zeros(n_steps)
    velocity = np.zeros(n_steps)
    error = np.zeros(n_steps)
    control = np.zeros(n_steps)
    
    position[0] = 0.0
    velocity[0] = 0.0
    error[0] = target
    
    integral = 0.0
    prev_error = error[0]
    prev_pv = 0.0
    prev_filtered = 0.0
    
    for i in range(1, n_steps):
        x_true = position[i - 1]
        v_true = velocity[i - 1]
        
        # Controller measurement (with noise if enabled)
        pv = x_true
        if add_noise:
            pv += np.random.normal(0, noise_std)
        
        err = target - pv
        error[i] = err
        
        if controller_fn == Propotional:
            u = controller_fn(controller_args["Kp"], err)
        elif controller_fn == PI:
            u, integral = controller_fn(
                controller_args["Kp"], controller_args["Ki"],
                err, prev_error, dt, integral
            )
        elif controller_fn == PD:
            u = controller_fn(
                controller_args["Kp"], controller_args["Kd"],
                err, prev_error, dt
            )
        elif controller_fn == PID:
            u, integral = controller_fn(
                controller_args["Kp"], controller_args["Ki"], controller_args["Kd"],
                err, prev_error, dt, integral
            )
        elif controller_fn == PID_antiwindup:
            u, integral = controller_fn(
                controller_args["Kp"], controller_args["Ki"], controller_args["Kd"],
                err, prev_error, dt, integral,
                controller_args["min_value"], controller_args["max_value"]
            )
        elif controller_fn == PID_no_kick:
            u, integral = controller_fn(
                controller_args["Kp"], controller_args["Ki"], controller_args["Kd"],
                err, pv, prev_pv, dt, integral
            )
        elif controller_fn == PID_filtered:
            u, integral, prev_filtered = controller_fn(
                controller_args["Kp"], controller_args["Ki"], controller_args["Kd"],
                err, pv, prev_pv, dt, integral,
                controller_args["min_value"], controller_args["max_value"],
                controller_args.get("alpha", 0.1),
                prev_filtered
            )
        else:
            raise ValueError(f"Unknown controller: {controller_fn}")
        
        if max_force is not None:
            u = Clamp(u, -max_force, max_force)
        
        control[i] = u
        
        # Plant physics uses TRUE state (no noise)
        spring_force = spring.force(x_true)
        damping_force = spring.damping_force(v_true)
        net_force = spring_force + damping_force + u
        acceleration = net_force / spring.mass
        
        new_velocity = v_true + acceleration * dt
        new_position = x_true + new_velocity * dt
        
        velocity[i] = new_velocity
        position[i] = new_position
        
        prev_error = err
        prev_pv = pv
    
    return {
        "time": time,
        "position": position,
        "velocity": velocity,
        "error": error,
        "control": control,
        "target": np.full_like(time, target)
    }


def experiment_1_p_only(C: Dict) -> Dict[str, Any]:
    """Experiment 1: P controller with different strengths."""
    print("Running Experiment 1: P-only with varying Kp...")
    Kp_values = C["Kp_values"]
    results = {}
    metrics_list = []
    
    for Kp in Kp_values:
        res = simulate_system(C, Propotional, {"Kp": Kp})
        name = f"P_Kp={Kp}"
        results[name] = res
        metrics = compute_metrics(res)
        metrics["Kp"] = Kp
        metrics_list.append(metrics)
        print(f"  Kp={Kp}: rise={metrics['rise_time']:.2f}s, "
              f"OS={metrics['overshoot_pct']:.1f}%, "
              f"settle={metrics['settling_time']:.2f}s, "
              f"SSE={metrics['steady_state_error']:.4f}")
    
    colors = {f"P_Kp={Kp}": c for Kp, c in zip(Kp_values, plt.cm.viridis(np.linspace(0, 1, len(Kp_values))))}
    plot_multi(results, "Experiment 1: P-only Response", "exp1_p_only.png", colors)
    
    save_data({"results": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in results.items()},
               "metrics": metrics_list}, "exp1_p_only.pkl")
    
    return {"results": results, "metrics": metrics_list}


def experiment_2_pi_pd_pid(C: Dict) -> Dict[str, Any]:
    """Experiment 2: PI, PD, and PID controllers."""
    print("Running Experiment 2: PI, PD, PID...")
    Kp, Ki, Kd = C["Kp"], C["Ki"], C["Kd"]
    controllers = {
        "P": (Propotional, {"Kp": Kp}),
        "PI": (PI, {"Kp": Kp, "Ki": Ki}),
        "PD": (PD, {"Kp": Kp, "Kd": Kd}),
        "PID": (PID, {"Kp": Kp, "Ki": Ki, "Kd": Kd}),
    }
    
    results = {}
    for name, (fn, args) in controllers.items():
        res = simulate_system(C, fn, args)
        results[name] = res
        m = compute_metrics(res)
        print(f"  {name}: rise={m['rise_time']:.2f}s, "
              f"OS={m['overshoot_pct']:.1f}%, "
              f"settle={m['settling_time']:.2f}s, "
              f"SSE={m['steady_state_error']:.4f}")
    
    colors = {"P": "blue", "PI": "green", "PD": "orange", "PID": "red"}
    plot_multi(results, "Experiment 2: PI, PD, PID Comparison", "exp2_pi_pd_pid.png", colors)
    
    save_data({"results": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in results.items()}}, "exp2_pi_pd_pid.pkl")
    
    return {"results": results}


def experiment_3_comparison(C: Dict) -> Dict[str, Any]:
    """Experiment 3: Detailed comparison with metrics table."""
    print("Running Experiment 3: Detailed Comparison...")
    Kp, Ki, Kd = C["Kp"], C["Ki"], C["Kd"]
    controllers = {
        "P": (Propotional, {"Kp": Kp}),
        "PI": (PI, {"Kp": Kp, "Ki": Ki}),
        "PD": (PD, {"Kp": Kp, "Kd": Kd}),
        "PID": (PID, {"Kp": Kp, "Ki": Ki, "Kd": Kd}),
    }
    
    results = {}
    for name, (fn, args) in controllers.items():
        res = simulate_system(C, fn, args)
        results[name] = res
    
    colors = {"P": "blue", "PI": "green", "PD": "orange", "PID": "red"}
    plot_comparison_table(results, "Experiment 3: P, PI, PD, PID Comparison",
                          "exp3_comparison.png", colors)
    
    save_data({"results": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in results.items()}}, "exp3_comparison.pkl")
    
    return {"results": results}


def experiment_4_real_world(C: Dict) -> Dict[str, Any]:
    """Experiment 4: Real-world problems and fixes."""
    print("Running Experiment 4: Real-world Problems...")
    Kp, Ki, Kd = C["Kp"], C["Ki"], C["Kd"]
    dt = C["dt"]
    t_max = C["t_max"]
    
    results = {}
    np.random.seed(0)  # Reproducible noise
    
    # (a) Windup: 30N limit (assignment says 30N). Holding 1m needs 20N (k*x = 20*1 = 20N).
    # Use gains that saturate initially: Kp=100 gives 100N at t=0.
    # Raise Ki high enough that integral winds up during saturation, causing overshoot when it unsaturates.
    print("  (a) Windup: max force = 30N, Kp=100, Ki=120 (high Ki to show overshoot)")
    windup_Kp, windup_Ki, windup_Kd = 100.0, 120.0, 10.0
    res_windup = simulate_system(C, PID, {"Kp": windup_Kp, "Ki": windup_Ki, "Kd": windup_Kd}, max_force=30.0)
    results["PID_windup"] = res_windup
    m = compute_metrics(res_windup)
    print(f"    rise={m['rise_time']:.2f}s, OS={m['overshoot_pct']:.1f}%, "
          f"settle={m['settling_time']:.2f}s, SSE={m['steady_state_error']:.4f}")
    print(f"    max control={np.max(np.abs(res_windup['control'])):.1f}N, "
          f"saturated={np.sum(np.abs(res_windup['control']) >= 30)} steps")
    
    print("  (a) Fix: Anti-windup (conditional integration)")
    res_antiwindup = simulate_system(C, PID_antiwindup,
        {"Kp": windup_Kp, "Ki": windup_Ki, "Kd": windup_Kd, "min_value": -30, "max_value": 30})
    results["PID_antiwindup"] = res_antiwindup
    m = compute_metrics(res_antiwindup)
    print(f"    rise={m['rise_time']:.2f}s, OS={m['overshoot_pct']:.1f}%, "
          f"settle={m['settling_time']:.2f}s, SSE={m['steady_state_error']:.4f}")
    print(f"    max control={np.max(np.abs(res_antiwindup['control'])):.1f}N")
    
    # (b) Noisy sensor: Noise only affects measurement, not physics (fixed in simulate_system)
    print("  (b) Noisy sensor: ±1mm, high Kd=50")
    noisy_Kd = 50.0
    res_noisy = simulate_system(C, PID, {"Kp": Kp, "Ki": Ki, "Kd": noisy_Kd},
                                add_noise=True, noise_std=0.001)
    results["PID_noisy"] = res_noisy
    m = compute_metrics(res_noisy)
    print(f"    rise={m['rise_time']:.2f}s, OS={m['overshoot_pct']:.1f}%, "
          f"settle={m['settling_time']:.2f}s, SSE={m['steady_state_error']:.4f}")
    print(f"    control std={np.std(res_noisy['control']):.2f}N")
    
    print("  (b) Fix: Derivative on measurement + filter (alpha=0.1)")
    res_filtered = simulate_system(C, PID_filtered,
        {"Kp": Kp, "Ki": Ki, "Kd": noisy_Kd, "min_value": -100, "max_value": 100, "alpha": 0.1},
        add_noise=True, noise_std=0.001)
    results["PID_filtered"] = res_filtered
    m = compute_metrics(res_filtered)
    print(f"    rise={m['rise_time']:.2f}s, OS={m['overshoot_pct']:.1f}%, "
          f"settle={m['settling_time']:.2f}s, SSE={m['steady_state_error']:.4f}")
    print(f"    control std={np.std(res_filtered['control']):.2f}N")
    
    # (c) Sudden target change (kick) - step at t=3s from 1 to 2
    print("  (c) Sudden target change (kick) at t=3s: 1 -> 2")
    kick_t_max = 8.0
    kick_dt = dt
    n_steps = int(kick_t_max / kick_dt) + 1
    time = np.linspace(0, kick_t_max, n_steps)
    position = np.zeros(n_steps)
    velocity = np.zeros(n_steps)
    error = np.zeros(n_steps)
    control = np.zeros(n_steps)
    target_arr = np.zeros(n_steps)
    
    spring = Spring(k=20.0, d=10.0, m=1.0, x0=0.0)
    position[0] = 0.0
    velocity[0] = 0.0
    target_arr[0] = 1.0
    error[0] = 1.0  # prev_error = initial error (no kick at t=0)
    
    integral = 0.0
    prev_error = 1.0
    prev_pv = 0.0
    
    for i in range(1, n_steps):
        x_true = position[i - 1]
        v_true = velocity[i - 1]
        pv = x_true  # No noise for kick demo
        
        target = 1.0 if time[i] < 3.0 else 2.0
        target_arr[i] = target
        err = target - pv
        error[i] = err
        
        u, integral = PID(Kp, Ki, Kd, err, prev_error, kick_dt, integral)
        control[i] = u
        
        spring_force = spring.force(x_true)
        damping_force = spring.damping_force(v_true)
        net_force = spring_force + damping_force + u
        acceleration = net_force / spring.mass
        
        new_velocity = v_true + acceleration * kick_dt
        new_position = x_true + new_velocity * kick_dt
        
        velocity[i] = new_velocity
        position[i] = new_position
        
        prev_error = err
        prev_pv = pv
    
    res_kick = {"time": time, "position": position, "velocity": velocity,
                "error": error, "control": control, "target": target_arr}
    results["PID_kick"] = res_kick
    m = compute_metrics(res_kick)
    print(f"    max control spike={np.max(np.abs(control)):.1f}N (expected ~Kd*Δr/dt = {Kd*1/0.01:.0f}N)")
    print(f"    rise={m['rise_time']:.2f}s, OS={m['overshoot_pct']:.1f}%, "
          f"settle={m['settling_time']:.2f}s, SSE={m['steady_state_error']:.4f}")
    
    print("  (c) Fix: Derivative on measurement (no D kick)")
    integral = 0.0
    prev_error = 1.0
    prev_pv = 0.0
    position2 = np.zeros(n_steps)
    velocity2 = np.zeros(n_steps)
    error2 = np.zeros(n_steps)
    control2 = np.zeros(n_steps)
    position2[0] = 0.0
    velocity2[0] = 0.0
    error2[0] = 1.0
    
    for i in range(1, n_steps):
        x_true = position2[i - 1]
        v_true = velocity2[i - 1]
        pv = x_true
        
        target = 1.0 if time[i] < 3.0 else 2.0
        err = target - pv
        error2[i] = err
        
        u, integral = PID_no_kick(Kp, Ki, Kd, err, pv, prev_pv, kick_dt, integral)
        control2[i] = u
        
        spring_force = spring.force(x_true)
        damping_force = spring.damping_force(v_true)
        net_force = spring_force + damping_force + u
        acceleration = net_force / spring.mass
        
        new_velocity = v_true + acceleration * kick_dt
        new_position = x_true + new_velocity * kick_dt
        
        velocity2[i] = new_velocity
        position2[i] = new_position
        
        prev_error = err
        prev_pv = pv
    
    res_no_kick = {"time": time, "position": position2, "velocity": velocity2,
                   "error": error2, "control": control2, "target": target_arr}
    results["PID_no_kick"] = res_no_kick
    m = compute_metrics(res_no_kick)
    print(f"    max control spike={np.max(np.abs(control2)):.1f}N (P kick only: Kp*Δr = {Kp*1:.0f}N)")
    print(f"    rise={m['rise_time']:.2f}s, OS={m['overshoot_pct']:.1f}%, "
          f"settle={m['settling_time']:.2f}s, SSE={m['steady_state_error']:.4f}")
    
    # (d) Slow update: ZOH with fixed plant dt=0.001, vary controller Ts
    print("  (d) Slow update: ZOH sweep (plant dt=1ms, controller Ts=1,10,20,50,100ms)")
    dt_sim = 0.001
    Ts_values = [0.001, 0.01, 0.02, 0.05, 0.1]
    t_max_slow = 10.0
    
    for Ts in Ts_values:
        hold = int(round(Ts / dt_sim))
        n_steps = int(t_max_slow / dt_sim) + 1
        time = np.linspace(0, t_max_slow, n_steps)
        position = np.zeros(n_steps)
        velocity = np.zeros(n_steps)
        error = np.zeros(n_steps)
        control = np.zeros(n_steps)
        target_arr = np.full(n_steps, 1.0)
        
        position[0] = 0.0
        velocity[0] = 0.0
        error[0] = 1.0
        
        integral = 0.0
        prev_error = 1.0
        u = 0.0
        
        for i in range(1, n_steps):
            x_true = position[i - 1]
            v_true = velocity[i - 1]
            pv = x_true
            
            err = 1.0 - pv
            error[i] = err
            
            # Update controller only every Ts
            if (i - 1) % hold == 0:
                u, integral = PID(Kp, Ki, Kd, err, prev_error, Ts, integral)
                prev_error = err
            
            control[i] = u
            
            spring_force = spring.force(x_true)
            damping_force = spring.damping_force(v_true)
            net_force = spring_force + damping_force + u
            acceleration = net_force / spring.mass
            
            new_velocity = v_true + acceleration * dt_sim
            new_position = x_true + new_velocity * dt_sim
            
            velocity[i] = new_velocity
            position[i] = new_position
        
        res = {"time": time, "position": position, "velocity": velocity,
               "error": error, "control": control, "target": target_arr}
        results[f"PID_Ts_{int(Ts*1000)}ms"] = res
        m = compute_metrics(res)
        stable = "stable" if np.max(np.abs(position)) < 10 else "UNSTABLE"
        print(f"    Ts={int(Ts*1000)}ms: rise={m['rise_time']:.2f}s, OS={m['overshoot_pct']:.1f}%, "
              f"settle={m['settling_time']:.2f}s, SSE={m['steady_state_error']:.4f} [{stable}]")
    
    # Plot separate figures for each problem (problem + fix on same graph)
    plot_windup(results, "exp4a_windup.png")
    plot_noise(results, "exp4b_noise.png")
    plot_kick(results, "exp4c_kick.png")
    plot_zoh_sweep(results, "Experiment 4d: Slow Update (ZOH Sweep)", "exp4d_zoh.png")
    
    save_data({"results": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in results.items()}}, "exp4_real_world.pkl")
    
    return {"results": results}


def run_all(C: Dict) -> Dict[str, Any]:
    """Run all experiments sequentially."""
    print("=" * 60)
    print("RUNNING ALL PID EXPERIMENTS")
    print("=" * 60)
    
    exp1 = experiment_1_p_only(C)
    print()
    
    exp2 = experiment_2_pi_pd_pid(C)
    print()
    
    exp3 = experiment_3_comparison(C)
    print()
    
    exp4 = experiment_4_real_world(C)
    print()
    
    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Data saved to: {DATA_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print("=" * 60)
    
    return {"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4}


