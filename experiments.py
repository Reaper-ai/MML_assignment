"""
Experiment scaffold for PID controller analysis.
Experiment conduction logic (simulation, running experiments).
"""

import json
from typing import Any, Dict, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt

from system import Spring
from controllers import (
    P, PI, PD, PID,
    PID_no_windup, PID_no_kick, PID_filtered, PID_robust_all,
    Clamp, D_filtered_pv
)
from plotting import (
    plot_multi, plot_comparison_table,
    plot_windup, plot_noise, plot_kick, plot_zoh_sweep, plot_final_comparison
)
from utils import DATA_DIR, PLOTS_DIR, compute_metrics, save_data, load_data

def simulate_system(C: Dict, controller_fn: Callable, controller_args: Dict,
                    target: float = 1.0, add_noise: bool = False,
                    noise_std: float = 0.001, max_force: Optional[float] = None) -> Dict[str, np.ndarray]:
    
    spring = Spring(k=20.0, d=10.0, m=1.0, x0=0.0)
    dt = C["dt"]
    t_max = C["t_max"]
    n_steps = int(t_max / dt) + 1
    
    time = np.linspace(0, t_max, n_steps)
    position, velocity, error, control = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    
    error[0] = target
    integral = 0.0
    prev_error = error[0]
    prev_pv = 0.0
    prev_filtered = 0.0
    
    for i in range(1, n_steps):
        x_true, v_true = position[i - 1], velocity[i - 1]
        
        pv = x_true
        if add_noise:
            pv += np.random.normal(0, noise_std)
        
        err = target - pv
        error[i] = err
        
        if controller_fn == P:
            u_req = controller_fn(controller_args["Kp"], err)
        elif controller_fn == PI:
            u_req, integral = controller_fn(controller_args["Kp"], controller_args["Ki"], err, prev_error, dt, integral)
        elif controller_fn == PD:
            u_req = controller_fn(controller_args["Kp"], controller_args["Kd"], err, prev_error, dt)
        elif controller_fn == PID:
            u_req, integral = controller_fn(controller_args["Kp"], controller_args["Ki"], controller_args["Kd"], err, prev_error, dt, integral)
        elif controller_fn == PID_no_windup:
            u_req, integral = controller_fn(
                controller_args["Kp"], controller_args["Ki"], controller_args["Kd"], err, prev_error, dt, integral,
                controller_args["min_value"], controller_args["max_value"]
            )
        elif controller_fn == PID_filtered:
            u_req, integral, prev_filtered = controller_fn(
                controller_args["Kp"], controller_args["Ki"], controller_args["Kd"], err, prev_error, dt, integral,
                controller_args.get("alpha", 0.1), prev_filtered
            )
        elif controller_fn == PID_no_kick:
            u_req, integral, prev_filtered = controller_fn(
                controller_args["Kp"], controller_args["Ki"], controller_args["Kd"], err, pv, prev_pv, dt, integral,
                controller_args.get("alpha", 0.1), prev_filtered
            )
        elif controller_fn == PID_robust_all:
            u_req, integral, prev_filtered = controller_fn(
                controller_args["Kp"], controller_args["Ki"], controller_args["Kd"], err, pv, prev_pv, dt, integral,
                controller_args["min_value"], controller_args["max_value"], controller_args.get("alpha", 0.1), prev_filtered
            )
        else:
            raise ValueError(f"Unknown controller: {controller_fn}")
        
        # Plant limits the requested force (actuator saturation)
        u_plant = Clamp(u_req, -max_force, max_force) if max_force is not None else u_req
        control[i] = u_plant
        
        spring_force = spring.force(x_true)
        damping_force = spring.damping_force(v_true)
        net_force = spring_force + damping_force + u_plant
        acceleration = net_force / spring.mass
        
        velocity[i] = v_true + acceleration * dt
        position[i] = x_true + velocity[i] * dt
        
        prev_error, prev_pv = err, pv
    
    return {"time": time, "position": position, "velocity": velocity, "error": error, "control": control, "target": np.full_like(time, target)}


def experiment_1_p_only(C: Dict) -> Dict[str, Any]:
    Kp_values = C["Kp_values"]
    results, metrics_list = {}, []
    for Kp in Kp_values:
        res = simulate_system(C, P, {"Kp": Kp})
        results[f"P_Kp={Kp}"] = res
        metrics = compute_metrics(res)
        metrics["Kp"] = Kp
        metrics_list.append(metrics)
    
    plot_multi(results, "Experiment 1: P-only Response", "exp1_p_only.png", {f"P_Kp={Kp}": c for Kp, c in zip(Kp_values, plt.cm.viridis(np.linspace(0, 1, len(Kp_values))))})
    return {"results": results, "metrics": metrics_list}


def experiment_2_pi_pd_pid(C: Dict) -> Dict[str, Any]:
    Kp, Ki, Kd = C["Kp"], C["Ki"], C["Kd"]
    controllers = {"P": (P, {"Kp": Kp}), "PI": (PI, {"Kp": Kp, "Ki": Ki}), "PD": (PD, {"Kp": Kp, "Kd": Kd}), "PID": (PID, {"Kp": Kp, "Ki": Ki, "Kd": Kd})}
    
    results = {}
    for name, (fn, args) in controllers.items():
        results[name] = simulate_system(C, fn, args)
    
    plot_multi(results, "Experiment 2: PI, PD, PID Comparison", "exp2_pi_pd_pid.png", {"P": "blue", "PI": "green", "PD": "orange", "PID": "red"})
    return {"results": results}


def experiment_3_comparison(C: Dict) -> Dict[str, Any]:
    Kp, Ki, Kd = C["Kp"], C["Ki"], C["Kd"]
    controllers = {"P": (P, {"Kp": Kp}), "PI": (PI, {"Kp": Kp, "Ki": Ki}), "PD": (PD, {"Kp": Kp, "Kd": Kd}), "PID": (PID, {"Kp": Kp, "Ki": Ki, "Kd": Kd})}
    
    results = {name: simulate_system(C, fn, args) for name, (fn, args) in controllers.items()}
    plot_comparison_table(results, "Experiment 3: P, PI, PD, PID Comparison", "exp3_comparison.png", {"P": "blue", "PI": "green", "PD": "orange", "PID": "red"})
    return {"results": results}


def experiment_4_real_world(C: Dict) -> Dict[str, Any]:
    Kp, Ki, Kd = C["Kp"], C["Ki"], C["Kd"]
    results = {}
    np.random.seed(0)
    
    # (a) Windup vs Anti-windup
    windup_Kp, windup_Ki, windup_Kd = 100.0, 120.0, 10.0
    results["PID_windup"] = simulate_system(C, PID, {"Kp": windup_Kp, "Ki": windup_Ki, "Kd": windup_Kd}, max_force=30.0)
    results["PID_no_windup"] = simulate_system(C, PID_no_windup, {"Kp": windup_Kp, "Ki": windup_Ki, "Kd": windup_Kd, "min_value": -30, "max_value": 30}, max_force=30.0)
    
    # (b) Noisy sensor vs Filtered
    noisy_Kd = 50.0
    results["PID_noisy"] = simulate_system(C, PID, {"Kp": Kp, "Ki": Ki, "Kd": noisy_Kd}, add_noise=True, noise_std=0.001)
    results["PID_filtered"] = simulate_system(C, PID_filtered, {"Kp": Kp, "Ki": Ki, "Kd": noisy_Kd, "alpha": 0.1}, add_noise=True, noise_std=0.001)
    
    # (c) Kick vs No Kick (Manually simulated step change)
    kick_dt, n_steps = C["dt"], int(8.0 / C["dt"]) + 1
    time = np.linspace(0, 8.0, n_steps)
    position, velocity, error, control, target_arr = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    spring = Spring(k=20.0, d=10.0, m=1.0, x0=0.0)
    target_arr[0], error[0] = 1.0, 1.0
    integral, prev_error = 0.0, 1.0
    
    for i in range(1, n_steps):
        target = 1.0 if time[i] < 3.0 else 2.0
        target_arr[i], pv = target, position[i - 1]
        err = target - pv
        error[i] = err
        
        u, integral = PID(Kp, Ki, Kd, err, prev_error, kick_dt, integral)
        control[i] = u
        velocity[i] = velocity[i - 1] + ((spring.force(pv) + spring.damping_force(velocity[i-1]) + u) / spring.mass) * kick_dt
        position[i] = pv + velocity[i] * kick_dt
        prev_error = err
    results["PID_kick"] = {"time": time, "position": position, "velocity": velocity, "error": error, "control": control, "target": target_arr}
    
    position2, velocity2, error2, control2 = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    error2[0], integral, prev_pv, prev_filtered = 1.0, 0.0, 0.0, 0.0
    
    for i in range(1, n_steps):
        target = 1.0 if time[i] < 3.0 else 2.0
        pv = position2[i - 1]
        err = target - pv
        error2[i] = err
        
        u, integral, prev_filtered = PID_no_kick(Kp, Ki, Kd, err, pv, prev_pv, kick_dt, integral, 0.1, prev_filtered)
        control2[i] = u
        velocity2[i] = velocity2[i - 1] + ((spring.force(pv) + spring.damping_force(velocity2[i-1]) + u) / spring.mass) * kick_dt
        position2[i] = pv + velocity2[i] * kick_dt
        prev_pv = pv
    results["PID_no_kick"] = {"time": time, "position": position2, "velocity": velocity2, "error": error2, "control": control2, "target": target_arr}

    # (d) Slow update: ZOH with fixed plant dt=0.001, vary controller Ts
    dt_sim = 0.001
    Ts_values = [0.001, 0.01, 0.1, 0.2, 0.3]
    t_max_slow = 5.0
    spring = Spring(k=20.0, d=10.0, m=1.0, x0=0.0)
    
    for Ts in Ts_values:
        hold = int(round(Ts / dt_sim))
        n_steps = int(t_max_slow / dt_sim) + 1
        time = np.linspace(0, t_max_slow, n_steps)
        position, velocity, error, control = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
        target_arr = np.full(n_steps, 1.0)
        error[0], integral, prev_error, u = 1.0, 0.0, 1.0, 0.0
        
        for i in range(1, n_steps):
            x_true, v_true = position[i - 1], velocity[i - 1]
            err = 1.0 - x_true
            error[i] = err
            if (i - 1) % hold == 0:
                u, integral = PID(Kp, Ki, Kd, err, prev_error, Ts, integral)
                prev_error = err
            control[i] = u
            accel = (spring.force(x_true) + spring.damping_force(v_true) + u) / spring.mass
            velocity[i] = v_true + accel * dt_sim
            position[i] = x_true + velocity[i] * dt_sim
            
        results[f"PID_Ts_{int(Ts*1000)}ms"] = {
            "time": time, "position": position, "velocity": velocity,
            "error": error, "control": control, "target": target_arr
        }
    plot_zoh_sweep(results, "exp4d_zoh.png")
    
    # Plots
    plot_windup(results, "exp4a_windup.png")
    plot_noise(results, "exp4b_noise.png")
    plot_kick(results, "exp4c_kick.png")
    return {"results": results}


def experiment_5_ziegler_nichols(C: Dict) -> Dict[str, Any]:
    dt, t_max = C["dt"], 20.0
    n_steps, time = int(t_max / dt) + 1, np.linspace(0, t_max, int(t_max / dt) + 1)
    
    Ku, Tu = 8.0, 2 * np.pi / np.sqrt(3) # Analytical constants for 1/(s+1)^3
    
    zn_params = {
        "P": {"Kp": 0.5 * Ku, "Ki": 0, "Kd": 0},
        "PI": {"Kp": 0.45 * Ku, "Ki": 0.54 * Ku / Tu, "Kd": 0},
        "PID": {"Kp": 0.6 * Ku, "Ki": 1.2 * Ku / Tu, "Kd": 0.075 * Ku * Tu},
        "PID_Improved": {"Kp": 0.48 * Ku, "Ki": 1.0 * Ku / Tu, "Kd": 0.1 * Ku * Tu}
    }
    
    results = {}
    for name, params in zn_params.items():
        x1, x2, x3, u, error = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
        integral, prev_err = 0.0, 1.0
        Kp, Ki, Kd = params["Kp"], params["Ki"], params["Kd"]
        
        for i in range(1, n_steps):
            err = 1.0 - x3[i-1]
            error[i] = err
            integral += err * dt
            deriv = (err - prev_err) / dt
            u[i] = Kp * err + Ki * integral + Kd * deriv
            prev_err = err
            
            x1[i] = x1[i-1] + dt * (-x1[i-1] + u[i-1])
            x2[i] = x2[i-1] + dt * (-x2[i-1] + x1[i-1])
            x3[i] = x3[i-1] + dt * (-x3[i-1] + x2[i-1])
            
        results[name] = {"time": time, "position": x3, "control": u, "target": np.ones_like(time), "error": error, "params": params}
    
    from plotting import plot_zn_results
    plot_zn_results(results, Ku, Tu, "exp5_zn_tuning.png")
    return {"results": results, "Ku": Ku, "Tu": Tu}


def experiment_6_final_comparison(C: Dict) -> Dict[str, Any]:
    Kp, Ki, Kd = 100.0, 120.0, 50.0 
    dt_sim, Ts, t_max = 0.001, 0.05, 15.0
    hold, n_steps = int(round(Ts / dt_sim)), int(t_max / dt_sim) + 1
    time = np.linspace(0, t_max, n_steps)
    target = np.ones(n_steps)
    target[int(5.0 / dt_sim):] = 2.0
    spring = Spring(k=20.0, d=10.0, m=1.0, x0=0.0)
    
    def run_sim(robust=False):
        v, x, u, error = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
        integral, prev_err, prev_pv, prev_filt = 0.0, 1.0, 0.0, 0.0
        for i in range(1, n_steps):
            x_true, v_true = x[i-1], v[i-1]
            pv = x_true + np.random.normal(0, 0.001)
            err = target[i] - pv
            error[i] = err
            
            if (i - 1) % hold == 0:
                if robust:
                    u_req, integral, prev_filt = PID_robust_all(Kp, Ki, Kd, err, pv, prev_pv, Ts, integral, -30, 30, 0.1, prev_filt)
                else:
                    u_req = Kp * err + Ki * integral + Kd * (err - prev_err) / Ts
                    integral += err * Ts
                prev_err, prev_pv = err, pv
            else:
                u_req = u[i-1]
            
            u_plant = float(Clamp(u_req, -30, 30))
            u[i] = u_plant
            accel = (spring.force(x_true) + spring.damping_force(v_true) + u_plant) / spring.mass
            v[i] = v_true + accel * dt_sim
            x[i] = x_true + v[i] * dt_sim
        return {"time": time, "position": x, "velocity": v, "control": u, "target": target, "error": error}

    res_basic = run_sim(robust=False)
    res_robust = run_sim(robust=True)
    
    plot_final_comparison(res_basic, res_robust, "exp6_final_comparison.png")
    return {"basic": res_basic, "robust": res_robust}


def run_all(C: Dict) -> Dict[str, Any]:
    print("Running experiments...")
    exp1 = experiment_1_p_only(C)
    exp2 = experiment_2_pi_pd_pid(C)
    exp3 = experiment_3_comparison(C)
    exp4 = experiment_4_real_world(C)
    exp5 = experiment_5_ziegler_nichols(C)
    exp6 = experiment_6_final_comparison(C)
    
    # Export all specific metrics for LaTeX auto-filling
    export_data = {
        "exp1_P_only": {name: compute_metrics(res) for name, res in exp1["results"].items()},
        "exp3_PID_comp": {name: compute_metrics(res) for name, res in exp3["results"].items()},
        "exp4_windup": compute_metrics(exp4["results"]["PID_windup"]),
        "exp4_antiwindup": compute_metrics(exp4["results"]["PID_no_windup"]),
        "exp4_noisy_std": float(np.std(exp4["results"]["PID_noisy"]["control"])),
        "exp4_filtered_std": float(np.std(exp4["results"]["PID_filtered"]["control"])),
        "exp4_kick_spike": float(np.max(np.abs(exp4["results"]["PID_kick"]["control"]))),
        "exp4_nokick_spike": float(np.max(np.abs(exp4["results"]["PID_no_kick"]["control"]))),
        "exp5_ZN_Ku": exp5["Ku"],
        "exp5_ZN_Tu": exp5["Tu"],
        "exp5_ZN_metrics": {name: compute_metrics(res) for name, res in exp5["results"].items()},
        "exp6_basic": compute_metrics(exp6["basic"]),
        "exp6_robust": compute_metrics(exp6["robust"]),
        "exp6_basic_std": float(np.std(exp6["basic"]["control"])),
        "exp6_robust_std": float(np.std(exp6["robust"]["control"]))
    }
    
    with open(DATA_DIR / "metrics_export.json", "w") as f:
        json.dump(export_data, f, indent=2)
        
    print(f"Data saved to: {DATA_DIR}/metrics_export.json")
    return {}