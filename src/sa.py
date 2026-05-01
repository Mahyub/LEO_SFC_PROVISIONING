"""
src/sa.py
=========
Simulated Annealing (SA) warm-start module.

Produces a high-quality feasible placement to serve as a warm-start
incumbent for the MILP solver, following the algorithm pseudocode in
Phase 1.4 of the Implementation Guide.

Key engineering decisions (from Phase 5.3):
  - Placement represented as Dict[(n,u,f) -> (i,s)] for O(1) lookup.
  - Incremental CPU load counters: only two satellites updated per move.
  - Precomputed temperature lookup table: avoids pow() in the hot loop.
  - Best-solution tracking: returns best seen (not final state).
  - Geometric cooling schedule: T_k = T0 * (Tend/T0)^(k/I).

Assumptions
-----------
- A move proposes reassigning one (n,u,f) to a new (i,s).
- Constraint C1 (unique assignment) is maintained by construction.
- C3 (capacity) is checked before accepting any move.
- C5 (delay) is verified approximately using the minimum ISL delay
  for each consecutive function pair hop.
- The risk delta is computed in O(n_s) using co-location counters.
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Tuple

from .types import (
    FuncID, InstID, PlacementKey, PlacementState, PlacementVal,
    PreprocessResult, RiskParameters, SatID, SFC, SliceID,
    TopologySnapshot, UserID, VNFInstance,
)
from .metrics import compute_risk_exact, compute_cap_use
from .instance_generator import INSTANCES_PER_SAT
from .visibility import (
    precompute_user_visibility,
    compute_access_delay_ms,
    all_pairs_sp_delays,
)


# ---------------------------------------------------------------------------
# Objective evaluation helper
# ---------------------------------------------------------------------------

def _compute_objective(
    cap_use: float,
    risk_ex: float,
    mig_cost: float,
    preprocess: PreprocessResult,
    config: dict,
) -> float:
    """
    Compute the normalized weighted objective (eq:obj).

    obj = omega_cap  * (CapUse / CapUse_bar)
        + omega_risk * (Risk^LB / Risk_bar)
        + omega_mig  * (Mig / Mig_bar)
    """
    w_cap  = config.get("omega_cap", 0.3)
    w_risk = config.get("omega_risk", 0.5)
    w_mig  = config.get("omega_mig", 0.2)

    norm_cap  = cap_use  / max(preprocess.cap_use_bar, 1e-9)
    norm_risk = risk_ex  / max(preprocess.risk_bar,    1e-9)
    norm_mig  = mig_cost / max(preprocess.mig_bar,     1e-9)

    return w_cap * norm_cap + w_risk * norm_risk + w_mig * norm_mig


# ---------------------------------------------------------------------------
# Greedy initializer
# ---------------------------------------------------------------------------

def _greedy_init(
    sfcs: List[SFC],
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    config: dict,
    rng: random.Random,
) -> Optional[PlacementState]:
    """
    Greedy nearest-feasible initializer.

    For each (sfc, u, f_pos) in random order:
      - Ingress function (f_pos == 0): only considers satellites visible
        from the slice's ground station (visibility-aware access).
      - Subsequent functions: considers all satellites (ISL-routed).

    Within the candidate set the satellite with lowest BFS hop count from
    the nearest visible satellite is preferred.

    Returns None if no feasible assignment can be found.
    """
    state = PlacementState(epoch=topo.epoch)
    min_el = config.get("min_elevation_deg", 10.0)

    # Precompute visible satellites per (sfc_idx, user_id)
    vis_sats = precompute_user_visibility(topo, sfcs, min_el)

    # Build BFS-ordered satellite list for a given (sfc_idx, user_id):
    # visible satellites first, then remaining by hop distance from the
    # nearest visible satellite.
    def _sat_order(sfc_idx: int, u: int) -> List[int]:
        vis = vis_sats.get((sfc_idx, u), [])
        source = vis[0] if vis else 0
        from collections import deque
        dist: Dict[int, int] = {source: 0}
        queue: deque = deque([source])
        while queue:
            s = queue.popleft()
            for t in topo.neighbors_of(s):
                if t not in dist:
                    dist[t] = dist[s] + 1
                    queue.append(t)
        vis_set = set(vis)
        visible   = sorted(vis_set,              key=lambda s: dist.get(s, 999))
        invisible = sorted(set(range(topo.num_satellites)) - vis_set,
                           key=lambda s: dist.get(s, 999))
        return visible + invisible

    sat_orders = {
        (sfc_idx, u): _sat_order(sfc_idx, u)
        for sfc_idx, sfc in enumerate(sfcs)
        for u in sfc.user_ids
    }
    vis_sets = {(sfc_idx, u): set(vis_sats.get((sfc_idx, u), []))
                for sfc_idx, sfc in enumerate(sfcs)
                for u in sfc.user_ids}

    # Randomize (sfc_idx, u_idx, f_pos) assignment order
    assignments: List[Tuple[int, int, int]] = []   # (sfc_idx, u_idx, f_pos)
    for sfc_idx, sfc in enumerate(sfcs):
        for u_idx in range(len(sfc.user_ids)):
            for f_pos in range(len(sfc.functions)):
                assignments.append((sfc_idx, u_idx, f_pos))
    rng.shuffle(assignments)

    for (sfc_idx, u_idx, f_pos) in assignments:
        sfc    = sfcs[sfc_idx]
        n      = sfc.slice_id
        u      = sfc.user_ids[u_idx]
        f_type = sfc.functions[f_pos]

        # Ingress: restrict to this user's visible satellites; others use
        # the full BFS order (ISL-routed, no visibility restriction).
        if f_pos == 0:
            vis_u = vis_sets.get((sfc_idx, u), set())
            candidates = [s for s in sat_orders[(sfc_idx, u)] if s in vis_u]
            if not candidates:
                candidates = sat_orders[(sfc_idx, u)]
        else:
            candidates = sat_orders[(sfc_idx, u)]

        placed = False
        for s in candidates:
            for i in range(INSTANCES_PER_SAT):
                vi = vnf_instances.get((f_type, i, s))
                if vi is None:
                    continue
                cur_load = state.cpu_load.get(s, 0.0)
                is_active = state.instance_active.get((f_type, i, s), False)
                added = vi.per_user_cpu + (0.0 if is_active else vi.activation_cpu)
                if cur_load + added <= topo.cpu_capacity.get(s, 0.0):
                    ext_key: PlacementKey = (n, u, f_pos, f_type)
                    state.assignment[ext_key] = (i, s)
                    state.cpu_load[s] = cur_load + added
                    state.instance_active[(f_type, i, s)] = True
                    placed = True
                    break
            if placed:
                break

        if not placed:
            return None  # Infeasible instance — caller must handle

    return state


# ---------------------------------------------------------------------------
# E2E delay helper
# ---------------------------------------------------------------------------

def _e2e_delay_ms(
    state: PlacementState,
    sfc: SFC,
    sfc_idx: int,
    u: UserID,
    override_key: PlacementKey,
    override_val: PlacementVal,
    access_delay_ms: Dict[Tuple[int, int, int], float],
    sp_delay_ms: Dict[Tuple[int, int], float],
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
) -> float:
    """
    Compute E2E delay (ms) for user u in SFC after applying one assignment
    override (override_key -> override_val).  Returns inf if any function
    has no current assignment.

    Components:
      access delay  : user ground station → ingress satellite (f_pos 0)
      ISL delay     : shortest-path between each consecutive function pair
      processing    : VNF proc_delay_ms summed over all functions
    """
    n = sfc.slice_id
    sat_seq: List[Tuple[InstID, SatID, FuncID]] = []
    for f_pos, f_type in enumerate(sfc.functions):
        k = (n, u, f_pos, f_type)
        val = override_val if k == override_key else state.assignment.get(k)
        if val is None:
            return float("inf")
        i_v, s_v = val
        sat_seq.append((i_v, s_v, f_type))

    total = 0.0
    # Access delay: user terminal → ingress satellite (per-user location)
    _, s0, _ = sat_seq[0]
    total += access_delay_ms.get((sfc_idx, u, s0), 0.0)
    # ISL shortest-path delay between consecutive functions
    for fp in range(len(sat_seq) - 1):
        _, s_a, _ = sat_seq[fp]
        _, s_b, _ = sat_seq[fp + 1]
        total += sp_delay_ms.get((s_a, s_b), float("inf"))
    # VNF processing delay at each hop
    for i_v, s_v, f_type in sat_seq:
        vi = vnf_instances.get((f_type, i_v, s_v))
        if vi:
            total += vi.proc_delay_ms
    return total


# ---------------------------------------------------------------------------
# Move proposal and acceptance
# ---------------------------------------------------------------------------

def _try_move(
    state: PlacementState,
    key: PlacementKey,
    new_i: InstID,
    new_s: SatID,
    sfcs: List[SFC],
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    risk_params: RiskParameters,
    preprocess: PreprocessResult,
    config: dict,
    sfc_by_slice: Dict[int, Tuple[int, SFC]],
    access_delay_ms: Dict[Tuple[int, int], float],
    sp_delay_ms: Dict[Tuple[int, int], float],
) -> Optional[float]:
    """
    Check feasibility of reassigning (n,u,f) to (new_i, new_s) and compute
    the resulting incremental objective change Delta.

    Returns Delta if the move is feasible, None if infeasible (C3 or C5 violated).
    """
    n, u, f_pos, f = key
    old_i, old_s = state.assignment[key]

    if old_i == new_i and old_s == new_s:
        return 0.0  # No-op

    vi_old = vnf_instances.get((f, old_i, old_s))
    vi_new = vnf_instances.get((f, new_i, new_s))
    if vi_new is None:
        return None

    # --- Check C3 (capacity) on new satellite ---
    new_load_before = state.cpu_load.get(new_s, 0.0)
    is_new_active   = state.instance_active.get((f, new_i, new_s), False)
    added_cpu       = vi_new.per_user_cpu + (0.0 if is_new_active else vi_new.activation_cpu)

    if new_load_before + added_cpu > topo.cpu_capacity.get(new_s, 0.0) + 1e-9:
        return None  # C3 violated

    # --- Check C5 (E2E delay budget) ---
    if sfc_by_slice and access_delay_ms:
        entry = sfc_by_slice.get(n)
        if entry is not None:
            sfc_idx, sfc = entry
            budget = sfc.e2e_budget_ms.get(u, float("inf"))
            if not math.isinf(budget):
                delay = _e2e_delay_ms(
                    state, sfc, sfc_idx, u,
                    override_key=key,
                    override_val=(new_i, new_s),
                    access_delay_ms=access_delay_ms,
                    sp_delay_ms=sp_delay_ms,
                    vnf_instances=vnf_instances,
                )
                if delay > budget + 1e-6:
                    return None  # C5 violated

    # --- Compute CapUse delta ---
    # Old satellite: removing user; may deactivate instance if last user
    old_users_on_inst = sum(
        1 for (n2, u2, fp2, f2), (i2, s2) in state.assignment.items()
        if f2 == f and i2 == old_i and s2 == old_s and (n2, u2, fp2, f2) != (n, u, f_pos, f)
    )
    freed_cpu = vi_old.per_user_cpu if vi_old else 0.0
    if old_users_on_inst == 0 and vi_old:
        freed_cpu += vi_old.activation_cpu  # instance deactivates

    delta_cap = (added_cpu - freed_cpu) / max(preprocess.cap_use_bar, 1e-9)

    # --- Compute Risk delta (O(n_s)) ---
    # Find all slices currently co-located at (f, old_i, old_s) and (f, new_i, new_s)
    old_coloc_slices: Dict[SliceID, int] = {}
    new_coloc_slices: Dict[SliceID, int] = {}
    for (n2, u2, _, f2), (i2, s2) in state.assignment.items():
        if f2 == f and i2 == old_i and s2 == old_s and (n2, u2) != (n, u):
            old_coloc_slices[n2] = old_coloc_slices.get(n2, 0) + 1
        if f2 == f and i2 == new_i and s2 == new_s:
            new_coloc_slices[n2] = new_coloc_slices.get(n2, 0) + 1

    # Risk removed: user u of slice n leaves (f, old_i, old_s)
    risk_removed = 0.0
    for n2, cnt2 in old_coloc_slices.items():
        if n2 == n:
            continue
        w = risk_params.risk_weight(min(n, n2), max(n, n2), f)
        risk_removed += w * cnt2  # 1 user of slice n times cnt2 users of slice n2

    # Risk added: user u of slice n joins (f, new_i, new_s)
    risk_added = 0.0
    for n2, cnt2 in new_coloc_slices.items():
        if n2 == n:
            continue
        w = risk_params.risk_weight(min(n, n2), max(n, n2), f)
        risk_added += w * cnt2

    delta_risk = (risk_added - risk_removed) / max(preprocess.risk_bar, 1e-9)

    # --- Compute Migration delta ---
    w_mig = config.get("omega_mig", 0.2)
    # If pi[key] was True (prev placement was feasible), moving incurs cost
    pi_val = preprocess.pi.get(key, False)
    prev_i, prev_s = state.assignment.get(key, (new_i, new_s))  # current = old
    was_keeping = (old_i == prev_i and old_s == prev_s)  # already migrated?
    if pi_val and was_keeping and (new_i != old_i or new_s != old_s):
        # Moving away from the feasible previous placement incurs mig cost
        d = risk_params.migration_cost_D.get(f, 1.0)
        delta_mig = d / max(preprocess.mig_bar, 1e-9)
    else:
        delta_mig = 0.0

    delta = (config.get("omega_cap", 0.3) * delta_cap
             + config.get("omega_risk", 0.5) * delta_risk
             + config.get("omega_mig", 0.2) * delta_mig)

    return delta


def _apply_move(
    state: PlacementState,
    key: PlacementKey,
    new_i: InstID,
    new_s: SatID,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    delta_obj: float,
) -> None:
    """Apply a validated move to the state in-place. O(1)."""
    n, u, _, f = key
    old_i, old_s = state.assignment[key]

    vi_old = vnf_instances.get((f, old_i, old_s))
    vi_new = vnf_instances.get((f, new_i, new_s))

    # Remove from old satellite
    if vi_old:
        state.cpu_load[old_s] = state.cpu_load.get(old_s, 0.0) - vi_old.per_user_cpu
        # Check if instance becomes inactive
        still_used = any(
            f2 == f and i2 == old_i and s2 == old_s and (n2, u2, fp2, f2) != key
            for (n2, u2, fp2, f2), (i2, s2) in state.assignment.items()
        )
        if not still_used:
            state.instance_active[(f, old_i, old_s)] = False
            state.cpu_load[old_s] = state.cpu_load.get(old_s, 0.0) - vi_old.activation_cpu

    # Add to new satellite
    if vi_new:
        is_active = state.instance_active.get((f, new_i, new_s), False)
        if not is_active:
            state.instance_active[(f, new_i, new_s)] = True
            state.cpu_load[new_s] = state.cpu_load.get(new_s, 0.0) + vi_new.activation_cpu
        state.cpu_load[new_s] = state.cpu_load.get(new_s, 0.0) + vi_new.per_user_cpu

    # Update assignment
    state.assignment[key] = (new_i, new_s)
    state.obj_value += delta_obj


# ---------------------------------------------------------------------------
# Main SA solver
# ---------------------------------------------------------------------------

def run_sa(
    sfcs: List[SFC],
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    risk_params: RiskParameters,
    preprocess: PreprocessResult,
    config: dict,
    seed: int,
) -> Tuple[Optional[PlacementState], float]:
    """
    Run the SA warm-start and return (best_placement, sa_time_seconds).

    Algorithm (Phase 1.4 pseudocode, Stages 2):
    1. Greedy initialization
    2. SA loop: propose, check feasibility, compute Delta, Metropolis accept
    3. Track best solution found; return it (not the final state)

    Returns None if no feasible initial placement can be found.
    """
    t_start = time.perf_counter()
    rng = random.Random(seed)

    T0   = config.get("sa_T0", 1.0)
    Tend = config.get("sa_Tend", 0.01)
    I    = config.get("sa_iterations", 50_000)
    F    = config.get("function_types", [])
    S    = topo.num_satellites

    # Precompute delay tables for C5 checking in _try_move
    # sfc_by_slice: slice_id -> (sfc_idx, sfc) for O(1) lookup
    sfc_by_slice: Dict[int, Tuple[int, SFC]] = {
        sfc.slice_id: (idx, sfc) for idx, sfc in enumerate(sfcs)
    }

    # access_delay_ms[(sfc_idx, u, s)]: propagation delay from user u's
    # terminal location to satellite s.  Each user has an independent location.
    access_delay_ms: Dict[Tuple[int, int, int], float] = {}
    if topo.sat_positions:
        for sfc_idx, sfc in enumerate(sfcs):
            for u in sfc.user_ids:
                lat_u, lon_u = sfc.user_location(u)
                for s, (lat_s, lon_s, alt_s) in topo.sat_positions.items():
                    access_delay_ms[(sfc_idx, u, s)] = compute_access_delay_ms(
                        lat_u, lon_u, lat_s, lon_s, alt_s
                    )

    # sp_delay_ms[(s_a, s_b)]: shortest-path ISL delay between all satellite pairs
    sp_delay_ms: Dict[Tuple[int, int], float] = all_pairs_sp_delays(topo)

    # Stage 2a: Greedy initialization
    state = _greedy_init(sfcs, topo, vnf_instances, config, rng)
    if state is None:
        sa_time = time.perf_counter() - t_start
        return None, sa_time

    # Precompute temperature schedule (avoids pow() in hot loop)
    temps = [T0 * (Tend / T0) ** (k / max(I, 1)) for k in range(I + 1)]

    # Build key list for random sampling
    all_keys: List[PlacementKey] = list(state.assignment.keys())

    # Compute initial objective
    cap_use, *_ = compute_cap_use(state, topo, vnf_instances)
    risk_ex = compute_risk_exact(state, sfcs, risk_params, vnf_instances)
    state.obj_value = _compute_objective(cap_use, risk_ex, 0.0, preprocess, config)

    best_state   = state.copy()
    best_obj     = state.obj_value

    # Stage 2b: SA loop
    for k in range(I):
        T_k = temps[k]

        # Propose: random (n,u,f) and random new (i,s)
        key  = rng.choice(all_keys)
        new_s = rng.randint(0, S - 1)
        new_i = rng.randint(0, INSTANCES_PER_SAT - 1)

        # Check feasibility and compute delta
        delta = _try_move(
            state, key, new_i, new_s,
            sfcs, topo, vnf_instances, risk_params, preprocess, config,
            sfc_by_slice=sfc_by_slice,
            access_delay_ms=access_delay_ms,
            sp_delay_ms=sp_delay_ms,
        )

        if delta is None:
            continue  # Infeasible move

        # Metropolis acceptance
        if delta <= 0 or rng.random() < math.exp(-delta / max(T_k, 1e-12)):
            _apply_move(state, key, new_i, new_s, vnf_instances, delta)

            # Track best
            if state.obj_value < best_obj:
                best_obj   = state.obj_value
                best_state = state.copy()

    sa_time = time.perf_counter() - t_start
    return best_state, sa_time
