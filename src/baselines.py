"""
src/baselines.py
================
Implementations of the three baseline methods for comparison:

  B1 — Res-Min:        Minimize CPU only (no risk, no migration).
  B2 — Risk-Unaware:   Minimize CPU with delay constraint (no risk).
  B3 — Greedy:         Nearest-feasible satellite heuristic; no MILP.

B1 and B2 reuse the MILP infrastructure with modified objective weights..
B3 is a pure Python greedy assignment, O(|N|*U*F*S).

All three use the same SolverResult interface and evaluate Risk^ex
via the exact formula for fair comparison, regardless of their
optimization approach.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Tuple

from .types import (
    FuncID, InstID, PlacementKey, PlacementState,
    PreprocessResult, RiskParameters, SatID, SFC,
    SolverResult, SliceID, TopologySnapshot, UserID, VNFInstance,
)
from .metrics import (
    compute_risk_exact, compute_risk_bounds, compute_cap_use,
    compute_migrations, compute_delay_compliance, compute_peak_sat_util,
    compute_isl_load,
)
from .milp import solve_epoch
from .instance_generator import INSTANCES_PER_SAT
from .visibility import precompute_user_visibility


# ---------------------------------------------------------------------------
# B1 — Resource-Minimizing Baseline
# ---------------------------------------------------------------------------

def solve_b1(
    sfcs: List[SFC],
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    risk_params: RiskParameters,
    preprocess: PreprocessResult,
    prev_placement: Optional[PlacementState],
    config: dict,
) -> SolverResult:
    """
    B1: Minimize CPU activation overhead only.
    omega = (1.0, 0.0, 0.0) — no risk or migration term.
    Same MILP structure as proposed; only the objective changes.
    """
    result = solve_epoch(
        sfcs, topo, vnf_instances, risk_params,
        preprocess, prev_placement, config,
        warmstart=None,
        method="B1",
    )
    result.method = "B1"
    return result


# ---------------------------------------------------------------------------
# B2 — Risk-Unaware Baseline
# ---------------------------------------------------------------------------

def solve_b2(
    sfcs: List[SFC],
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    risk_params: RiskParameters,
    preprocess: PreprocessResult,
    prev_placement: Optional[PlacementState],
    config: dict,
) -> SolverResult:
    """
    B2: Minimize CPU + migration cost; no risk term.
    omega = (0.5, 0.0, 0.5) as set in solve_epoch for method="B2".

    Differs from B1 (pure resource minimization) in that migration
    stability is penalized, so B2 tends to keep VNF placements stable
    across epochs even though it ignores cross-slice risk entirely.
    """
    result = solve_epoch(
        sfcs, topo, vnf_instances, risk_params,
        preprocess, prev_placement, config,
        warmstart=None,
        method="B2",
    )
    result.method = "B2"
    return result


# ---------------------------------------------------------------------------
# B3 — Greedy Heuristic
# ---------------------------------------------------------------------------

def solve_b3(
    sfcs: List[SFC],
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    risk_params: RiskParameters,
    preprocess: PreprocessResult,
    prev_placement: Optional[PlacementState],
    config: dict,
    seed: int = 0,
) -> SolverResult:
    """
    B3: Greedy nearest-feasible satellite assignment.

    Algorithm:
    1. Shuffle all (n,u,f) triples.
    2. For each, select the satellite s* with minimum hop count from
       satellite 0 (proxy for user access satellite) that has sufficient
       residual CPU.
    3. Assign to the first available instance on s*.
    4. No migration stabilization; re-greedify from scratch each epoch.

    Risk^ex is still evaluated via the exact formula post-hoc.
    """
    t_start = time.perf_counter()
    rng = random.Random(seed)

    state = PlacementState(epoch=topo.epoch)
    S = topo.num_satellites

    # Compute visible satellites per slice and derive BFS ordering.
    # Ingress function: only visible satellites are candidates.
    # Other functions: full constellation, ordered by hop distance from the
    # nearest visible satellite (replacing the old satellite-0 proxy).
    min_el = config.get("min_elevation_deg", 10.0)
    vis_sats = precompute_user_visibility(topo, sfcs, min_el)

    def _ordered_sats(sfc_idx: int, u: UserID) -> List[SatID]:
        vis = vis_sats.get((sfc_idx, u), [])
        source = vis[0] if vis else 0
        hop_dist = _bfs_hops(source, topo)
        vis_set  = set(vis)
        visible   = sorted(vis_set,       key=lambda s: hop_dist.get(s, 999))
        invisible = sorted(set(range(S)) - vis_set, key=lambda s: hop_dist.get(s, 999))
        return visible + invisible

    sfc_sat_order = {
        (sfc_idx, u): _ordered_sats(sfc_idx, u)
        for sfc_idx, sfc in enumerate(sfcs)
        for u in sfc.user_ids
    }
    vis_sets = {
        (sfc_idx, u): set(vis_sats.get((sfc_idx, u), []))
        for sfc_idx, sfc in enumerate(sfcs)
        for u in sfc.user_ids
    }

    # Shuffle (sfc_idx, u, f_pos) triples for random tie-breaking
    triples: List[Tuple[int, UserID, int]] = [          # (sfc_idx, u, f_pos)
        (sfc_idx, u, f_pos)
        for sfc_idx, sfc in enumerate(sfcs)
        for u in sfc.user_ids
        for f_pos in range(len(sfc.functions))
    ]
    rng.shuffle(triples)

    # Greedy assignment
    for (sfc_idx, u, f_pos) in triples:
        sfc    = sfcs[sfc_idx]
        n      = sfc.slice_id
        f_type = sfc.functions[f_pos]

        # Ingress: restricted to this user's visible satellites
        if f_pos == 0:
            vis_u = vis_sets.get((sfc_idx, u), set())
            candidates = [s for s in sfc_sat_order[(sfc_idx, u)] if s in vis_u]
            if not candidates:                          # coverage-gap fallback
                candidates = sfc_sat_order[(sfc_idx, u)]
        else:
            candidates = sfc_sat_order[(sfc_idx, u)]

        placed = False
        for s in candidates:
            for i in range(INSTANCES_PER_SAT):
                vi = vnf_instances.get((f_type, i, s))
                if vi is None:
                    continue
                cur_load = state.cpu_load.get(s, 0.0)
                is_active = state.instance_active.get((f_type, i, s), False)
                added = vi.per_user_cpu + (0.0 if is_active else vi.activation_cpu)
                if cur_load + added <= topo.cpu_capacity.get(s, 0.0) + 1e-9:
                    state.assignment[(n, u, f_pos, f_type)] = (i, s)
                    state.cpu_load[s] = cur_load + added
                    state.instance_active[(f_type, i, s)] = True
                    placed = True
                    break
            if placed:
                break

    solve_time = time.perf_counter() - t_start

    # Evaluate metrics
    risk_ex = compute_risk_exact(state, sfcs, risk_params, vnf_instances)
    risk_lb, risk_ub = compute_risk_bounds(state, sfcs, risk_params)
    cap_use, cap_use_pct, active_sat_util_pct = compute_cap_use(state, topo, vnf_instances)
    n_total, n_avoid, mig_cost = compute_migrations(
        state, prev_placement, preprocess, risk_params
    )
    delay_compliance_pct, _ = compute_delay_compliance(state, sfcs, topo, vnf_instances)
    peak_sat_util_pct = compute_peak_sat_util(state, topo, vnf_instances)
    max_isl_load, avg_isl_load, n_active_isl_links = compute_isl_load(state, sfcs, topo)
    risk_bound_tightness = risk_ex / risk_ub if risk_ub > 1e-9 else 1.0

    return SolverResult(
        placement=state,
        risk_ex=risk_ex,
        risk_lb=risk_lb,
        risk_ub=risk_ub,
        cap_use=cap_use,
        cap_use_pct=cap_use_pct,
        active_sat_util_pct=active_sat_util_pct,
        peak_sat_util_pct=peak_sat_util_pct,
        max_isl_load=max_isl_load,
        avg_isl_load=avg_isl_load,
        n_active_isl_links=n_active_isl_links,
        mig_cost=mig_cost,
        delay_compliance_pct=delay_compliance_pct,
        risk_bound_tightness=risk_bound_tightness,
        obj_value=float("nan"),  # No objective for pure heuristic
        solve_time_s=solve_time,
        sa_time_s=0.0,
        mip_gap_pct=0.0,
        status="heuristic",
        n_migrations=n_total,
        n_avoidable_migrations=n_avoid,
        method="B3",
        instance_id=0,
        epoch=topo.epoch,
    )


def _bfs_hops(source: SatID, topo: TopologySnapshot) -> Dict[SatID, int]:
    """BFS from source satellite to compute hop distances."""
    from collections import deque
    dist: Dict[SatID, int] = {source: 0}
    queue: deque = deque([source])
    while queue:
        s = queue.popleft()
        for t in topo.neighbors_of(s):
            if t not in dist:
                dist[t] = dist[s] + 1
                queue.append(t)
    return dist
