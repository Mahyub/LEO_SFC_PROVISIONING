"""
src/experiment.py
=================
Top-level experiment orchestrator.

Runs all four methods (B1, B2, B3, Proposed) across all instances,
epochs, and scenarios, logging every result to a JSON results file.

Usage
-----
>>> from src.experiment import run_experiment
>>> run_experiment("config/base.json", output_dir="data/results")

The orchestrator:
  1. Loads config and generates instances with unique seeds.
  2. For each instance:
       a. Generates topology snapshots for all epochs.
       b. Runs preprocessing each epoch.
       c. Runs SA warm-start.
       d. Solves with the proposed method (coarse + optional exact).
       e. Solves with B1, B2, B3 baselines.
  3. Writes all results to JSON (one file per scenario).
  4. Prints a summary table.

Logging schema:
  scenario_id, instance_id, method, epoch, status, obj_value,
  gap_pct, solve_time_s, sa_time_s, risk_ex, risk_lb, risk_ub,
  cap_use, mig_cost, n_migrations, n_avoidable_migrations,
  norm_base (B1 risk for normalization)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# from .instance_generator import InstanceGenerator, load_config, INSTANCES_PER_SAT

# To use MATLAB
from .instance_generator_matlab import InstanceGeneratorMatlab as InstanceGenerator, load_config
from .instance_generator import INSTANCES_PER_SAT   # keep this for the constant
# To HERE

from .preprocessing import run_preprocessing
from .sa import run_sa
from .milp import solve_epoch
from .baselines import solve_b1, solve_b2, solve_b3
from .metrics import (
    compute_risk_exact, compute_risk_bounds, compute_cap_use,
    compute_migrations, verify_risk_bounds, verify_migration_epigraph,
    check_capacity_constraints,
)
from .types import PlacementState, SolverResult
from .placement_logger import serialize_placement
from .user_visibility_logger import serialize_user_visibility


# ---------------------------------------------------------------------------
# Result record (flat dict for JSON serialization)
# ---------------------------------------------------------------------------

def _result_to_record(r: SolverResult, scenario_id: str, instance_id: int) -> dict:
    return {
        "scenario_id":          scenario_id,
        "instance_id":          instance_id,
        "method":               r.method,
        "epoch":                r.epoch,
        "status":               r.status,
        "obj_value":            r.obj_value,
        "mip_gap_pct":          r.mip_gap_pct,
        "solve_time_s":         r.solve_time_s,
        "sa_time_s":            r.sa_time_s,
        "risk_ex":              r.risk_ex,
        "risk_lb":              r.risk_lb,
        "risk_ub":              r.risk_ub,
        "cap_use":              r.cap_use,
        "cap_use_pct":          r.cap_use_pct,
        "active_sat_util_pct":  r.active_sat_util_pct,
        "peak_sat_util_pct":    r.peak_sat_util_pct,
        "max_isl_load":         r.max_isl_load,
        "avg_isl_load":         r.avg_isl_load,
        "n_active_isl_links":   r.n_active_isl_links,
        "mig_cost":             r.mig_cost,
        "n_migrations":         r.n_migrations,
        "n_avoidable_mig":      r.n_avoidable_migrations,
        "delay_compliance_pct": r.delay_compliance_pct,
        "risk_bound_tightness": r.risk_bound_tightness,
    }


# ---------------------------------------------------------------------------
# Single-instance runner
# ---------------------------------------------------------------------------

def run_instance(
    config: dict,
    instance_id: int,
    output_records: list,
    verbose: bool = True,
    config_path: str = "config/base.json",
    placement_records: Optional[list] = None,
    user_vis_records: Optional[list] = None,
) -> None:
    """
    Run all methods across all epochs for one simulation instance.

    Results are appended to output_records in-place.
    If placement_records is a list, full per-hop placement detail is appended
    there for every method and epoch (requires config["save_placements"]=True).
    """
    seed = config["random_seed_base"] + instance_id * 42
    scenario_id = config["scenario_id"]
    n_epochs = config.get("num_epochs", 10)

    gen = InstanceGenerator(config, seed, config_path=config_path)

    # Generate SFCs and risk parameters (fixed across epochs for one instance)
    sfcs        = gen.service_function_chains()
    risk_params = gen.risk_parameters(sfcs)

    if verbose:
        print(f"  Instance {instance_id:2d} | "
              f"slices={len(sfcs)} | users/slice={config['users_per_slice']}")

    # Pre-generate all epoch topologies in one MATLAB call so satellites
    # are propagated continuously rather than restarted from t=0 each epoch.
    if hasattr(gen, "precompute_epochs"):
        gen.precompute_epochs(n_epochs)

    # Previous placements per method (updated each epoch)
    prev: Dict[str, Optional[PlacementState]] = {
        "proposed": None, "B1": None, "B2": None, "B3": None
    }

    for epoch in range(n_epochs):
        topo         = gen.topology_snapshot(epoch)
        vnf_instances = gen.vnf_instances(topo)

        # ── User visibility snapshot ───────────────────────────────────
        if user_vis_records is not None:
            user_vis_records.append(
                serialize_user_visibility(
                    sfcs, topo, config, instance_id, epoch, scenario_id
                )
            )

        # ── Preprocessing ──────────────────────────────────────────────
        preprocess = run_preprocessing(
            prev["proposed"], topo, sfcs, risk_params, vnf_instances, config
        )

        # ── Stage 2: SA warm-start ─────────────────────────────────────
        sa_placement, sa_time = run_sa(
            sfcs, topo, vnf_instances, risk_params, preprocess, config,
            seed=seed + epoch
        )
        warmstart = sa_placement  # May be None if greedy init failed

        # ── Stage 3: Proposed MILP (coarse model) ─────────────────────
        r_prop = solve_epoch(
            sfcs, topo, vnf_instances, risk_params,
            preprocess, prev["proposed"], config,
            warmstart=warmstart,
            method="proposed_coarse",
        )
        r_prop.sa_time_s   = sa_time
        r_prop.instance_id = instance_id
        r_prop.epoch       = epoch
        prev["proposed"]   = r_prop.placement if r_prop.status != "infeasible" else prev["proposed"]
        output_records.append(_result_to_record(r_prop, scenario_id, instance_id))
        if placement_records is not None and r_prop.status != "infeasible":
            placement_records.append(
                serialize_placement(r_prop.placement, sfcs, topo, vnf_instances,
                                    "proposed_coarse", instance_id, epoch, scenario_id)
            )

        # ── Stage 3b: Proposed MILP (exact risk model) — optional ─────────
        if config.get("run_exact_model", False):
            r_exact = solve_epoch(
                sfcs, topo, vnf_instances, risk_params,
                preprocess, prev["proposed"], config,
                warmstart=warmstart,
                method="proposed_exact",
            )
            r_exact.sa_time_s   = sa_time
            r_exact.instance_id = instance_id
            r_exact.epoch       = epoch
            output_records.append(_result_to_record(r_exact, scenario_id, instance_id))
            if placement_records is not None and r_exact.status != "infeasible":
                placement_records.append(
                    serialize_placement(r_exact.placement, sfcs, topo, vnf_instances,
                                        "proposed_exact", instance_id, epoch, scenario_id)
                )
            if verbose:
                print(f"  Epoch {epoch:2d} | "
                      f"Exact  risk={r_exact.risk_ex:.3f} t={r_exact.solve_time_s:.1f}s")

        # ── Baselines ─────────────────────────────────────────────────
        baseline_results = {}
        for method, solver in [
            ("B1", lambda: solve_b1(sfcs, topo, vnf_instances, risk_params, preprocess, prev["B1"], config)),
            ("B2", lambda: solve_b2(sfcs, topo, vnf_instances, risk_params, preprocess, prev["B2"], config)),
            ("B3", lambda: solve_b3(sfcs, topo, vnf_instances, risk_params, preprocess, prev["B3"], config, seed=seed+epoch)),
        ]:
            r = solver()
            r.instance_id = instance_id
            r.epoch       = epoch
            if r.status != "infeasible":
                prev[method] = r.placement
            output_records.append(_result_to_record(r, scenario_id, instance_id))
            if placement_records is not None and r.status != "infeasible":
                placement_records.append(
                    serialize_placement(r.placement, sfcs, topo, vnf_instances,
                                        method, instance_id, epoch, scenario_id)
                )
            baseline_results[method] = r

        if verbose:
            b1, b2, b3 = baseline_results["B1"], baseline_results["B2"], baseline_results["B3"]
            print(f"  Epoch {epoch:2d} | "
                  f"Coarse risk={r_prop.risk_ex:.3f} t={r_prop.solve_time_s:.1f}s | "
                  f"B1 risk={b1.risk_ex:.3f} t={b1.solve_time_s:.1f}s | "
                  f"B2 risk={b2.risk_ex:.3f} t={b2.solve_time_s:.1f}s | "
                  f"B3 risk={b3.risk_ex:.3f} t={b3.solve_time_s:.1f}s")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    config_path: str,
    output_dir: str = "data/results",
    verbose: bool = True,
    max_instances: Optional[int] = None,
) -> str:
    """
    Run the full experiment for one scenario config.

    Returns the path to the output JSON results file.
    """
    config = load_config(config_path)
    scenario_id  = config["scenario_id"]
    n_instances  = config.get("num_instances", 30)
    if max_instances is not None:
        n_instances = min(n_instances, max_instances)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scenario_id}_results.json")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {scenario_id}")
        print(f"  Satellites:    {config['num_satellites']}")
        print(f"  Slices:        {config['num_slices']}")
        print(f"  Users/slice:   {config['users_per_slice']}")
        print(f"  Epochs:        {config['num_epochs']}")
        print(f"  Instances:     {n_instances}")
        print(f"  omega: cap={config['omega_cap']} risk={config['omega_risk']} mig={config['omega_mig']}")
        print(f"{'='*60}")

    all_records: List[dict] = []
    save_pl = config.get("save_placements", True)
    save_uv = config.get("save_user_visibility", True)
    all_placement_records: List[dict] = [] if save_pl else None  # type: ignore[assignment]
    all_user_vis_records:  List[dict] = [] if save_uv else None  # type: ignore[assignment]
    t_exp_start = time.perf_counter()

    for inst_id in range(n_instances):
        run_instance(config, inst_id, all_records, verbose=verbose,
                     config_path=config_path,
                     placement_records=all_placement_records,
                     user_vis_records=all_user_vis_records)

    t_exp = time.perf_counter() - t_exp_start

    # Write aggregate results
    with open(output_path, "w") as f:
        json.dump({
            "scenario_id": scenario_id,
            "config":      config,
            "n_instances": n_instances,
            "wall_time_s": t_exp,
            "records":     all_records,
        }, f, indent=2, default=str)

    # Write detailed placement data when requested
    if save_pl and all_placement_records:
        pl_path = os.path.join(output_dir, f"{scenario_id}_placements.json")
        with open(pl_path, "w") as f:
            json.dump({
                "scenario_id": scenario_id,
                "config":      config,
                "records":     all_placement_records,
            }, f, indent=2, default=str)
        if verbose:
            print(f"Placement detail -> {pl_path}")

    # Write user visibility data when requested
    if save_uv and all_user_vis_records:
        uv_path = os.path.join(output_dir, f"{scenario_id}_user_visibility.json")
        with open(uv_path, "w") as f:
            json.dump({
                "scenario_id": scenario_id,
                "config":      config,
                "records":     all_user_vis_records,
            }, f, indent=2, default=str)
        if verbose:
            print(f"User visibility  -> {uv_path}")

    if verbose:
        print(f"\nExperiment complete in {t_exp:.1f}s -> {output_path}")
        _print_summary(all_records)

    return output_path


def _print_summary(records: List[dict]) -> None:
    """Print a condensed summary table of mean metrics per method."""
    from collections import defaultdict
    import statistics

    by_method: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_method[r["method"]].append(r)

    print(f"\n{'Method':<20} {'Risk^ex':>10} {'CPU%':>8} {'Migrations':>12} {'Runtime(s)':>12}")
    print("-" * 65)
    for method in ["B1", "B2", "B3", "proposed_coarse", "proposed_exact"]:
        recs = by_method.get(method, [])
        if not recs:
            continue
        _no_sol = {"infeasible", "timelimit_nofeas"}
        risks    = [r["risk_ex"]         for r in recs if r["status"] not in _no_sol]
        cpu_pcts = [r["cap_use_pct"]     for r in recs if r["status"] not in _no_sol]
        migs     = [r["n_avoidable_mig"] for r in recs if r["status"] not in _no_sol]
        times    = [r["solve_time_s"]    for r in recs if r["status"] not in _no_sol]

        mean_risk = statistics.mean(risks)    if risks    else float("nan")
        mean_cpu  = statistics.mean(cpu_pcts) if cpu_pcts else float("nan")
        mean_migs = statistics.mean(migs)     if migs     else float("nan")
        mean_time = statistics.mean(times)    if times    else float("nan")

        print(f"{method:<20} {mean_risk:>10.4f} {mean_cpu:>7.1f}% {mean_migs:>12.1f} {mean_time:>12.2f}")
