#!/usr/bin/env python3
"""
main.py
=======
Main entry point demonstrating all modules of the LEO-SFC system.

Usage
-----
# Quick demo (1 instance, 2 epochs):
  python main.py --mode demo

# Full experiment (30 instances, base scenario):
  python main.py --mode experiment --config config/base.json

# Run unit tests:
  python main.py --mode test

# Reproduce paper figures from existing results:
  python main.py --mode figures --results data/results/base_results.json

  # Generate all figures from results:
python main.py --mode figures --results data/results/base_matlab_results.json
Assumptions
-----------
- PuLP is installed: pip install pulp
- matplotlib is installed: pip install matplotlib
- The system is run from the leo_sfc/ directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure package is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#from src.instance_generator import InstanceGenerator, load_config, print_instance_info

# To use MATLAB
from src.instance_generator_matlab import InstanceGeneratorMatlab as InstanceGenerator, load_config
from src.instance_generator import print_instance_info   # keep for the pretty-printer
# To HERE

from src.preprocessing import run_preprocessing
from src.sa import run_sa
from src.milp import solve_epoch
from src.baselines import solve_b1, solve_b2, solve_b3
from src.metrics import (
    compute_risk_exact, compute_risk_bounds, compute_cap_use,
    compute_migrations, verify_risk_bounds, verify_migration_epigraph,
    check_capacity_constraints,
)
from src.experiment import run_experiment
from src.placement_logger import serialize_placement
from analysis.placement_report import print_placement_record


# ---------------------------------------------------------------------------
# Demo mode: single instance walkthrough with console output
# ---------------------------------------------------------------------------

def run_demo(config_path: str = "config/base.json") -> None:
    """
    Demonstrate the full pipeline on one instance, two epochs.
    Prints detailed per-step output for understanding module interactions.
    """
    print("\n" + "=" * 65)
    print("  LEO-SFC Demo: Risk-Aware SFC Provisioning Pipeline")
    print("=" * 65)

    # ── 1. Load config and create instance ─────────────────────────
    config = load_config(config_path)
    config["num_epochs"] = 2
    config["sa_iterations"] = 1000   # Reduced for demo speed
    config["time_limit_s"] = 30
    config["verify_propositions"] = True

    gen = InstanceGenerator(config, seed=42)
    sfcs         = gen.service_function_chains()
    risk_params  = gen.risk_parameters(sfcs)

    print(f"\n[1] Instance generated")
    print(f"    Satellites:    {config['num_satellites']}")
    print(f"    Slices:        {len(sfcs)}")
    print(f"    Users/slice:   {config['users_per_slice']}")
    print(f"    Function types: {config['function_types']}")
    print(f"    Objective weights: ω_cap={config['omega_cap']}, "
          f"ω_risk={config['omega_risk']}, ω_mig={config['omega_mig']}")

    # Print detailed instance breakdown (epoch-0 snapshot for VNF/topology info)
    _topo0 = gen.topology_snapshot(0)
    _vnf0  = gen.vnf_instances(_topo0)
    print_instance_info(sfcs, risk_params, _vnf0, _topo0, config)

    prev_placement = None

    for epoch in range(2):
        print(f"\n{'─'*60}")
        print(f"  EPOCH {epoch}")
        print(f"{'─'*60}")

        # ── 2. Topology snapshot ────────────────────────────────────
        topo          = gen.topology_snapshot(epoch)
        vnf_instances = gen.vnf_instances(topo)
        print(f"\n[2] Topology snapshot (epoch {epoch})")
        print(f"    Satellites:   {topo.num_satellites}")
        print(f"    ISL edges:    {sum(len(v) for v in topo.isl_neighbors.values()) // 2}")
        avg_cap = sum(topo.cpu_capacity.values()) / len(topo.cpu_capacity)
        print(f"    Avg CPU cap:  {avg_cap:.1f} units")

        # ── 3. Preprocessing ────────────────────────────────────────
        t0 = time.perf_counter()
        preprocess = run_preprocessing(
            prev_placement, topo, sfcs, risk_params, vnf_instances, config
        )
        prep_time = time.perf_counter() - t0

        pi_true = sum(1 for v in preprocess.pi.values() if v)
        pi_total = len(preprocess.pi)
        print(f"\n[3] Preprocessing done in {prep_time:.3f}s")
        print(f"    π=1 (prev feasible): {pi_true}/{pi_total} user-func pairs")
        print(f"    Norm bounds: CapUse_bar={preprocess.cap_use_bar:.1f}, "
              f"Risk_bar={preprocess.risk_bar:.3f}, Mig_bar={preprocess.mig_bar:.3f}")

        # ── 4. SA warm-start ────────────────────────────────────────
        print(f"\n[4] Simulated Annealing warm-start ({config['sa_iterations']} iterations)")
        sa_placement, sa_time = run_sa(
            sfcs, topo, vnf_instances, risk_params, preprocess, config, seed=42 + epoch
        )
        if sa_placement is not None:
            sa_risk = compute_risk_exact(sa_placement, sfcs, risk_params, vnf_instances)
            sa_cap, sa_util = compute_cap_use(sa_placement, topo, vnf_instances)
            print(f"    SA time:       {sa_time:.2f}s")
            print(f"    SA Risk^ex:    {sa_risk:.4f}")
            print(f"    SA CapUse:     {sa_cap:.2f} ({sa_util:.1f}%)")
            ok, viols = check_capacity_constraints(sa_placement, topo, vnf_instances)
            print(f"    C3 feasible:   {'YES' if ok else f'NO ({len(viols)} violations)'}")
        else:
            print("    SA: no feasible placement found")

        # ── 5. Proposed MILP ────────────────────────────────────────
        print(f"\n[5] Proposed MILP (coarse risk model)")
        r_prop = solve_epoch(
            sfcs, topo, vnf_instances, risk_params,
            preprocess, prev_placement, config,
            warmstart=sa_placement,
            method="proposed_coarse",
        )
        r_prop.sa_time_s = sa_time
        print(f"    Status:        {r_prop.status}")
        print(f"    Solve time:    {r_prop.solve_time_s:.2f}s (+ SA {sa_time:.2f}s)")
        print(f"    MIP Gap:       {r_prop.mip_gap_pct:.3f}%")
        print(f"    Risk^ex:       {r_prop.risk_ex:.4f}")
        print(f"    Risk^LB:       {r_prop.risk_lb:.4f}")
        print(f"    Risk^UB:       {r_prop.risk_ub:.4f}")
        print(f"    Prop 1 (bounds): "
              f"{'OK' if verify_risk_bounds(r_prop.risk_ex, r_prop.risk_lb, r_prop.risk_ub) else 'VIOLATED'}")
        print(f"    CapUse:        {r_prop.cap_use:.2f}")
        print(f"    Avoidable mig: {r_prop.n_avoidable_migrations}")

        # ── 5b. Proposed MILP (exact risk model) — optional ─────────────
        r_exact = None
        if config.get("run_exact_model", False):
            print(f"\n[5b] Proposed MILP (exact risk model)")
            r_exact = solve_epoch(
                sfcs, topo, vnf_instances, risk_params,
                preprocess, prev_placement, config,
                warmstart=sa_placement,
                method="proposed_exact",
            )
            r_exact.sa_time_s = sa_time
            print(f"    Status:        {r_exact.status}")
            print(f"    Solve time:    {r_exact.solve_time_s:.2f}s")
            print(f"    MIP Gap:       {r_exact.mip_gap_pct:.3f}%")
            print(f"    Risk^ex:       {r_exact.risk_ex:.4f}")
            print(f"    Risk^LB:       {r_exact.risk_lb:.4f}")
            print(f"    Risk^UB:       {r_exact.risk_ub:.4f}")
            print(f"    CapUse:        {r_exact.cap_use:.2f}")
            print(f"    Avoidable mig: {r_exact.n_avoidable_migrations}")

        # ── 6. Baselines ────────────────────────────────────────────
        print(f"\n[6] Baseline methods")
        results = {}
        for name, solver in [
            ("B1", lambda: solve_b1(sfcs, topo, vnf_instances, risk_params, preprocess, None, config)),
            ("B2", lambda: solve_b2(sfcs, topo, vnf_instances, risk_params, preprocess, None, config)),
            ("B3", lambda: solve_b3(sfcs, topo, vnf_instances, risk_params, preprocess, None, config, seed=42)),
        ]:
            r = solver()
            results[name] = r
            print(f"    {name}: Risk^ex={r.risk_ex:.4f}, "
                  f"CapUse={r.cap_use:.2f}, "
                  f"time={r.solve_time_s:.2f}s")

        # ── 7. Comparison table ──────────────────────────────────────
        print(f"\n[7] Summary comparison (epoch {epoch})")
        print(f"    {'Method':<20} {'Risk^ex':>10} {'CapUse':>8} {'Avoid.Mig':>10}")
        print(f"    {'-'*52}")
        proposed_rows = [("Proposed (coarse)", r_prop)]
        if r_exact is not None:
            proposed_rows.append(("Proposed (exact)", r_exact))
        for name, r in proposed_rows + [(k, v) for k, v in results.items()]:
            print(f"    {name:<20} {r.risk_ex:>10.4f} {r.cap_use:>8.2f} {r.n_avoidable_migrations:>10}")

        if r_prop.status != "infeasible":
            prev_placement = r_prop.placement

    print("\n" + "=" * 65)
    print("  Demo complete.")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Placements mode: run one instance, print per-hop placement + delay tables
# ---------------------------------------------------------------------------

def run_placements(config_path: str = "config/base.json",
                   epochs: int = 3) -> None:
    """
    Re-run the proposed_coarse solver for `epochs` epochs (pure-Python
    topology, no MATLAB required) and print the full placement report
    for every epoch — satellite assignments, positions, and delay breakdown.

    Uses the pure-Python InstanceGenerator so it works standalone.
    Results are printed to stdout; nothing is written to disk.
    """
    # Use pure-Python generator to avoid MATLAB dependency
    from src.instance_generator import InstanceGenerator as PythonGen, load_config as _load
    from src.preprocessing import run_preprocessing
    from src.sa import run_sa
    from src.milp import solve_epoch
    from src.baselines import solve_b1, solve_b2, solve_b3

    print("\n" + "=" * 65)
    print("  LEO-SFC Placement Report")
    print("=" * 65)

    config = _load(config_path)
    config["num_epochs"]    = epochs
    config["sa_iterations"] = 5000
    config["time_limit_s"]  = 60

    gen          = PythonGen(config, seed=42)
    sfcs         = gen.service_function_chains()
    risk_params  = gen.risk_parameters(sfcs)

    print(f"\n  Slices : {len(sfcs)}")
    for sfc in sfcs:
        ns = "N" if sfc.ground_lat >= 0 else "S"
        ew = "E" if sfc.ground_lon >= 0 else "W"
        loc = f"{abs(sfc.ground_lat):.1f}°{ns} {abs(sfc.ground_lon):.1f}°{ew}"
        print(f"    Slice {sfc.slice_id}: {loc}  chain={sfc.functions}  "
              f"users={sfc.user_ids}  crit={sfc.criticality:.3f}")

    prev_placements = {"proposed": None, "B1": None, "B2": None, "B3": None}
    METHODS = ["proposed_coarse", "B1", "B2", "B3"]

    for epoch in range(epochs):
        topo          = gen.topology_snapshot(epoch)
        vnf_instances = gen.vnf_instances(topo)
        preprocess    = run_preprocessing(
            prev_placements["proposed"], topo, sfcs, risk_params, vnf_instances, config
        )

        sa_placement, sa_time = run_sa(
            sfcs, topo, vnf_instances, risk_params, preprocess, config,
            seed=42 + epoch,
        )

        solvers = {
            "proposed_coarse": lambda: solve_epoch(
                sfcs, topo, vnf_instances, risk_params, preprocess,
                prev_placements["proposed"], config,
                warmstart=sa_placement, method="proposed_coarse"),
            "B1": lambda: solve_b1(sfcs, topo, vnf_instances, risk_params,
                                   preprocess, prev_placements["B1"], config),
            "B2": lambda: solve_b2(sfcs, topo, vnf_instances, risk_params,
                                   preprocess, prev_placements["B2"], config),
            "B3": lambda: solve_b3(sfcs, topo, vnf_instances, risk_params,
                                   preprocess, prev_placements["B3"], config,
                                   seed=42 + epoch),
        }

        for method in METHODS:
            r = solvers[method]()
            r.epoch = epoch

            if r.status == "infeasible":
                print(f"\n  [Epoch {epoch} · {method}] INFEASIBLE — skipping placement report")
                continue

            key = "proposed" if method == "proposed_coarse" else method
            prev_placements[key] = r.placement

            rec = serialize_placement(
                r.placement, sfcs, topo, vnf_instances,
                method, instance_id=0, epoch=epoch, scenario_id=config["scenario_id"]
            )
            print_placement_record(rec)


# ---------------------------------------------------------------------------
# Figures mode: regenerate plots from existing results
# ---------------------------------------------------------------------------

def run_figures(
    results_path: str,
    output_dir: str = "data/results",
    methods: list = None,
) -> None:
    """
    Generate all evaluation figures from a saved results JSON file.

    Produces up to 10 PDF figures plus an extended statistics table on stdout.

    Usage:
      python main.py --mode figures --results data/results/base_matlab_results.json
      python main.py --mode figures --results data/results/base_matlab_results.json \\
          --methods B1 proposed_coarse proposed_exact
    """
    from analysis.figures import generate_all_figures
    generate_all_figures(results_path, output_dir=output_dir, verbose=True,
                         methods=methods or None)


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Run the full unit test suite."""
    import pytest
    result = pytest.main(["tests/test_all.py", "-v", "--tb=short"])
    sys.exit(result)


# ---------------------------------------------------------------------------
# Argument parsing and dispatch
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LEO-SFC: Risk-Aware SFC Provisioning in LEO Networks"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "experiment", "figures", "test", "placements"],
        default="demo",
        help="Execution mode (default: demo)",
    )
    parser.add_argument(
        "--config",
        default="config/base.json",
        help="Path to scenario configuration JSON",
    )
    parser.add_argument(
        "--results",
        default="data/results/base_results.json",
        help="Path to results JSON (for figures mode)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit number of instances (for quick runs)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs for placements mode (default: 3)",
    )
    parser.add_argument(
        "--placements-file",
        default=None,
        help="Load pre-saved *_placements.json instead of re-solving",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        metavar="METHOD",
        default=None,
        help="Subset of methods to include in figures "
             "(e.g. --methods B1 proposed_coarse proposed_exact). "
             "Default: all methods present in the results file.",
    )

    args = parser.parse_args()

    # Strip commas users may type as separators (e.g. --methods B1, B2, B3)
    if args.methods:
        args.methods = [m.strip(",") for m in args.methods if m.strip(",")]

    if args.mode == "demo":
        run_demo(args.config)
    elif args.mode == "experiment":
        run_experiment(
            args.config,
            output_dir=args.output_dir,
            verbose=True,
            max_instances=args.max_instances,
        )
    elif args.mode == "figures":
        run_figures(args.results, args.output_dir, methods=args.methods)
    elif args.mode == "test":
        run_tests()
    elif args.mode == "placements":
        if args.placements_file:
            from analysis.placement_report import display_placements
            display_placements(args.placements_file)
        else:
            run_placements(args.config, epochs=args.epochs)


if __name__ == "__main__":
    main()