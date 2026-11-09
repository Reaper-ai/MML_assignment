#!/usr/bin/env python3
"""
Main entry point for PID experiments.
Handles CLI arguments and runs experiments.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments import run_all, experiment_1_p_only, experiment_2_pi_pd_pid
from experiments import experiment_3_comparison, experiment_4_real_world, experiment_5_ziegler_nichols


def build_config(args: argparse.Namespace) -> dict:
    """Build constants dictionary from CLI arguments."""
    return {
        "dt": args.dt,
        "t_max": args.t_max,
        "Kp": args.Kp,
        "Ki": args.Ki,
        "Kd": args.Kd,
        "Kp_values": args.Kp_values,
    }


def main():
    parser = argparse.ArgumentParser(description="PID Controller Experiments")
    
    parser.add_argument("--dt", type=float, default=0.01, help="Time step (default: 0.01)")
    parser.add_argument("--t_max", type=float, default=5.0, help="Simulation time (default: 5.0)")
    parser.add_argument("--Kp", type=float, default=100.0, help="Proportional gain (default: 100)")
    parser.add_argument("--Ki", type=float, default=50.0, help="Integral gain (default: 50)")
    parser.add_argument("--Kd", type=float, default=10.0, help="Derivative gain (default: 10)")
    parser.add_argument("--Kp_values", type=float, nargs="+", default=[10, 50, 100, 200, 500],
                        help="Kp values for Exp 1 (default: 10 50 100 200 500)")
    
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4, 5, 0], default=0,
                        help="Run specific experiment (0=all, default: 0)")
    
    args = parser.parse_args()
    
    C = build_config(args)
    
    print("Configuration:")
    for k, v in C.items():
        print(f"  {k}: {v}")
    print()
    
    if args.exp == 1:
        experiment_1_p_only(C)
    elif args.exp == 2:
        experiment_2_pi_pd_pid(C)
    elif args.exp == 3:
        experiment_3_comparison(C)
    elif args.exp == 4:
        experiment_4_real_world(C)
    elif args.exp == 5:
        experiment_5_ziegler_nichols(C)
    else:
        run_all(C)


if __name__ == "__main__":
    main()