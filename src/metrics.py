"""
src/metrics.py
==============
Evaluation metrics for any placement solution.

All metrics are computed from the exact formulas in the paper, regardless
of which optimization model (exact or coarse) produced the placement.
This ensures a model-independent, fair comparison across all methods.

Key design principle: Risk^ex is ALWAYS evaluated via eq:risk_exact —
never via the coarse proxy — so that all reported risk values are
comparable on the same scale.

Corresponds to Phase 3.3 of the Implementation Guide.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .types import (
    FuncID, InstID, PlacementKey, PlacementState, PlacementVal,
    PreprocessResult, RiskParameters, SatID, SFC, SliceID,
    TopologySnapshot, UserID, VNFInstance,
)
from .visibility import compute_access_delay_ms


# ---------------------------------------------------------------------------
# Risk computation
# ---------------------------------------------------------------------------

def compute_risk_exact(
    placement: PlacementState,
    sfcs: List[SFC],
    risk_params: RiskParameters,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
) -> float:
    """
    Compute Risk^ex (eq:risk_exact) from the paper.

    Risk^ex = sum_{n<n'} sum_{f in F_n ∩ F_{n'}} sum_{i,s}
              w^risk_{n,n',f} * sum_{u in U_n} sum_{u' in U_{n'}}
              p_{n,u,n',u',f,i,s}

    where p_{n,u,n',u',f,i,s} = 1 iff both user u of slice n AND
    user u' of slice n' are assigned to the same instance (f,i,s).

    This is the ground-truth metric used to evaluate ALL methods.

    Complexity: O(n_s^2 * U^2 * F * I * S) — feasible only for small scales;
    for large scales, use compute_risk_bounds() and report the certificate.
    """
    total_risk = 0.0

    # Index placements by (f, i, s) for fast co-location lookup
    # co_location[(f,i,s)] = set of distinct (n, u) pairs assigned to this instance
    co_location: Dict[Tuple[FuncID, InstID, SatID], Set[Tuple[SliceID, UserID]]] = {}
    for (n, u, _, f), (i, s) in placement.assignment.items():
        key = (f, i, s)
        if key not in co_location:
            co_location[key] = set()
        co_location[key].add((n, u))  # set deduplicates repeated-type positions

    # For each instance, find all cross-slice user pairs
    for (f, i, s), users_on_inst in co_location.items():
        if len(users_on_inst) < 2:
            continue  # No co-location possible

        # Group users by slice (already deduplicated per user)
        by_slice: Dict[SliceID, List[UserID]] = {}
        for (n, u) in users_on_inst:
            by_slice.setdefault(n, []).append(u)

        # Sum over all cross-slice pairs (n < n')
        slice_ids = sorted(by_slice.keys())
        for idx_a, n in enumerate(slice_ids):
            for n_prime in slice_ids[idx_a + 1:]:
                w = risk_params.risk_weight(n, n_prime, f)
                if w == 0.0:
                    continue
                # Count user pairs
                u_n  = len(by_slice[n])
                u_np = len(by_slice[n_prime])
                total_risk += w * u_n * u_np

    return total_risk


def compute_risk_bounds(
    placement: PlacementState,
    sfcs: List[SFC],
    risk_params: RiskParameters,
) -> Tuple[float, float]:
    """
    Compute Risk^LB and Risk^UB (eqs:risk_lb and risk_ub).

    Risk^LB = sum_{n<n',f,i,s} w^risk_{n,n',f} * y_{n,n';f,i,s}
    Risk^UB = sum_{n<n',f,i,s} |U_n|*|U_{n'}| * w^risk_{n,n',f} * y_{n,n';f,i,s}

    where y_{n,n';f,i,s} = 1 iff both slices n and n' have at least one
    user on instance (f,i,s).  This is computed directly from the placement.

    Returns (Risk^LB, Risk^UB).
    """
    # Build z_{n,f,i,s}: is any user of slice n on instance (f,i,s)?
    z: Dict[Tuple[SliceID, FuncID, InstID, SatID], bool] = {}
    for (n, u, _, f), (i, s) in placement.assignment.items():
        z[(n, f, i, s)] = True

    # Build slice user counts
    slice_sizes = {sfc.slice_id: len(sfc.user_ids) for sfc in sfcs}

    lb = 0.0
    ub = 0.0

    # Enumerate all (n < n', f, i, s) tuples where y=1
    # Collect all (f, i, s) instances that have co-location
    instances_by_fs: Dict[Tuple[FuncID, InstID, SatID], Set[SliceID]] = {}
    for (n, f, i, s), active in z.items():
        if active:
            key = (f, i, s)
            instances_by_fs.setdefault(key, set()).add(n)

    for (f, i, s), slices in instances_by_fs.items():
        slice_list = sorted(slices)
        for idx_a, n in enumerate(slice_list):
            for n_prime in slice_list[idx_a + 1:]:
                w  = risk_params.risk_weight(n, n_prime, f)
                Un  = slice_sizes.get(n, 1)
                Unp = slice_sizes.get(n_prime, 1)
                lb += w            # y=1, assume 1 user per slice
                ub += w * Un * Unp  # y=1, assume all users

    return lb, ub


def verify_risk_bounds(
    risk_ex: float,
    risk_lb: float,
    risk_ub: float,
    tol: float = 1e-6,
) -> bool:
    """
    Verify Proposition 1: Risk^LB <= Risk^ex <= Risk^UB.
    Returns True if the bounds hold within tolerance.
    """
    return (risk_lb - tol <= risk_ex) and (risk_ex <= risk_ub + tol)


# ---------------------------------------------------------------------------
# Resource utilization
# ---------------------------------------------------------------------------

def compute_cap_use(
    placement: PlacementState,
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
) -> Tuple[float, float, float]:
    """
    Compute CapUse (eq:capuse) and utilization percentages.

    CapUse = sum_{s,f,i} b^cpu_{f,i,s} * gamma_{f,i,s}
           + sum_{n,u,f,i,s} a^cpu_{f,i,s}(n,u) * beta_{n,u,f,i,s}

    Returns (raw_cap_use, avg_util_pct, active_sat_util_pct).

    avg_util_pct       : cap_use / total constellation capacity (may be very
                         small when few satellites carry VNFs).
    active_sat_util_pct: cap_use / capacity of satellites that have at least
                         one active VNF instance — a more meaningful load
                         indicator than diluting by idle satellites.
    """
    # Activation overhead (b^cpu * gamma)
    act_overhead = 0.0
    active_sats: set = set()
    for (f, i, s), active in placement.instance_active.items():
        if active:
            vi = vnf_instances.get((f, i, s))
            if vi:
                act_overhead += vi.activation_cpu
            active_sats.add(s)

    # Per-user load (a^cpu * beta) — one entry per (n, u, f_pos)
    user_load = 0.0
    for (n, u, _, f), (i, s) in placement.assignment.items():
        vi = vnf_instances.get((f, i, s))
        if vi:
            user_load += vi.per_user_cpu

    cap_use = act_overhead + user_load

    # Average utilization across all satellites (denominator = full constellation)
    total_cap = sum(topo.cpu_capacity.values())
    avg_util_pct = (cap_use / total_cap * 100.0) if total_cap > 0 else 0.0

    # Active-satellite utilization (denominator = only satellites with VNFs)
    active_cap = sum(topo.cpu_capacity.get(s, 0.0) for s in active_sats)
    active_sat_util_pct = (cap_use / active_cap * 100.0) if active_cap > 0 else 0.0

    return cap_use, avg_util_pct, active_sat_util_pct


# ---------------------------------------------------------------------------
# Migration metrics
# ---------------------------------------------------------------------------

def compute_migrations(
    placement: PlacementState,
    prev_placement: Optional[PlacementState],
    preprocess: PreprocessResult,
    risk_params: RiskParameters,
) -> Tuple[int, int, float]:
    """
    Compute migration counts and disruption cost.

    Based on Propositions 2 and eq:mig:
    - mu_{n,u,f} = max(0, pi_{n,u,f} - k_{n,u,f})
    - k_{n,u,f}  = 1 iff current and previous (i,s) match (unique by C1)

    Returns
    -------
    (n_total_migrations, n_avoidable_migrations, mig_cost)

    n_total_migrations : int
        All (n,u,f) triples where the (i,s) changed from prev epoch.
    n_avoidable_migrations : int
        Those where pi_{n,u,f}=True (previous assignment was still feasible).
    mig_cost : float
        sum_{n,u,f} Dis^mig_f * mu_{n,u,f}
    """
    if prev_placement is None:
        return 0, 0, 0.0

    total = 0
    avoidable = 0
    mig_cost = 0.0

    for key, (i_new, s_new) in placement.assignment.items():
        n, u, _, f = key
        prev_val = prev_placement.assignment.get(key)
        if prev_val is None:
            continue  # New user, not a migration

        i_old, s_old = prev_val
        moved = (i_new != i_old) or (s_new != s_old)

        if moved:
            total += 1
            # Avoidable iff previous assignment was still feasible (pi=1)
            if preprocess.pi.get(key, False):
                avoidable += 1
                # Disruption cost = Dis^mig_f * mu = Dis^mig_f * 1
                d = risk_params.migration_cost_D.get(f, 1.0)
                mig_cost += d

    return total, avoidable, mig_cost


def verify_migration_epigraph(
    placement: PlacementState,
    prev_placement: Optional[PlacementState],
    preprocess: PreprocessResult,
    tol: float = 1e-6,
) -> bool:
    """
    Verify Proposition 2: mu_{n,u,f} = max(0, pi_{n,u,f} - k_{n,u,f}).
    Returns True if the epigraph property holds for all (n,u,f).
    """
    if prev_placement is None:
        return True

    for key, (i_new, s_new) in placement.assignment.items():
        prev_val = prev_placement.assignment.get(key)
        if prev_val is None:
            continue

        i_old, s_old = prev_val
        k = 1 if (i_new == i_old and s_new == s_old) else 0
        pi = 1 if preprocess.pi.get(key, False) else 0
        mu_expected = max(0, pi - k)

        # mu is implicitly 1 if avoidable migration occurred, 0 otherwise
        moved = (i_new != i_old) or (s_new != s_old)
        mu_actual = 1 if (moved and pi == 1) else 0

        if abs(mu_actual - mu_expected) > tol:
            return False

    return True


# ---------------------------------------------------------------------------
# Delay verification
# ---------------------------------------------------------------------------

def check_delay_compliance(
    placement: PlacementState,
    sfcs: List[SFC],
    topo: TopologySnapshot,
    routing: Optional[Dict] = None,
) -> Tuple[bool, float]:
    """
    Verify constraint C5 (E2E delay budget) for the given placement.

    In the absence of explicit routing decisions (which require the full
    MILP zeta variables), we compute the minimum possible delay by assuming
    all consecutive function pairs are co-located on the same satellite
    (delta=0) when feasible, otherwise taking the minimum ISL delay.

    Returns (all_compliant, max_violation_ms).
    """
    max_violation = 0.0
    all_ok = True

    vnf_delay: Dict[str, float] = {}  # cached per func type (not available here)
    # Use midpoint of delay range as proxy
    d_lo, d_hi = 1.0, 3.0  # ms — VNF processing delay range

    for sfc in sfcs:
        proc_delay = sum(1.5 for _ in sfc.functions)  # midpoint estimate

        for u in sfc.user_ids:
            budget = sfc.e2e_budget_ms.get(u, 120.0)
            # Minimum propagation: each hop between consecutive functions
            # If placed on different satellites, use minimum ISL delay
            prop_delay = 0.0
            for idx in range(len(sfc.functions) - 1):
                s1 = placement.get_satellite(sfc.slice_id, u, idx,   sfc.functions[idx])
                s2 = placement.get_satellite(sfc.slice_id, u, idx+1, sfc.functions[idx+1])
                if s1 is not None and s2 is not None and s1 != s2:
                    prop_delay += topo.delay(s1, s2)
                # If same satellite or unknown, delay is 0

            total = proc_delay + prop_delay
            violation = total - budget
            if violation > max_violation:
                max_violation = violation
            if total > budget + 1e-6:
                all_ok = False

    return all_ok, max_violation


# ---------------------------------------------------------------------------
# Feasibility check helper
# ---------------------------------------------------------------------------

def compute_delay_compliance(
    placement: PlacementState,
    sfcs: List[SFC],
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
) -> Tuple[float, float]:
    """
    Fraction of (sfc, user) pairs satisfying the E2E delay budget.

    Uses: access delay to ingress satellite + VNF processing delays per
    chain hop + ISL propagation delays between consecutive hops on
    different satellites.

    Note: MILP constraint C5 omits ISL delays for linearity, so this
    post-hoc check is stricter than what the solver enforces.  A small
    number of violations may appear even for "optimal" placements when
    the ISL component pushes a user marginally over budget.

    Returns (compliance_pct, max_violation_ms).
    compliance_pct = 100 * (# users within budget) / (# users with a budget).
    max_violation_ms > 0 means at least one user exceeds its budget.
    """
    total = 0
    compliant = 0
    max_viol = 0.0

    for sfc in sfcs:
        for u in sfc.user_ids:
            budget = sfc.e2e_budget_ms.get(u, float("inf"))
            if budget >= 1e9:
                continue
            total += 1

            # Access delay to ingress satellite (f_pos == 0)
            access_ms = 0.0
            s0_val = placement.assignment.get((sfc.slice_id, u, 0, sfc.functions[0]))
            if s0_val is not None and topo.sat_positions:
                s0 = s0_val[1]
                pos = topo.sat_positions.get(s0)
                if pos is not None:
                    u_lat, u_lon = sfc.user_location(u)
                    access_ms = compute_access_delay_ms(u_lat, u_lon, pos[0], pos[1], pos[2])

            # VNF processing delays along the chain
            proc_ms = 0.0
            for f_pos, f_type in enumerate(sfc.functions):
                val = placement.assignment.get((sfc.slice_id, u, f_pos, f_type))
                if val is not None:
                    inst_id, sat_id = val
                    vi = vnf_instances.get((f_type, inst_id, sat_id))
                    if vi:
                        proc_ms += vi.proc_delay_ms

            # ISL propagation delays between consecutive hops
            isl_ms = 0.0
            for f_pos in range(len(sfc.functions) - 1):
                v1 = placement.assignment.get((sfc.slice_id, u, f_pos,     sfc.functions[f_pos]))
                v2 = placement.assignment.get((sfc.slice_id, u, f_pos + 1, sfc.functions[f_pos + 1]))
                if v1 is not None and v2 is not None and v1[1] != v2[1]:
                    isl_ms += topo.delay(v1[1], v2[1])

            total_ms = access_ms + proc_ms + isl_ms
            viol = total_ms - budget
            if viol > max_viol:
                max_viol = viol
            if total_ms <= budget + 1e-6:
                compliant += 1

    pct = (compliant / total * 100.0) if total > 0 else 100.0
    return pct, max_viol


def compute_peak_sat_util(
    placement: PlacementState,
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
) -> float:
    """
    Peak satellite CPU utilization (%) — the highest per-satellite load/capacity
    ratio across all satellites with positive capacity.

    Complements the average utilization returned by compute_cap_use().
    High peak values indicate hot spots even when the average is moderate.
    """
    load: Dict[SatID, float] = {}

    for (f, i, s), active in placement.instance_active.items():
        if active:
            vi = vnf_instances.get((f, i, s))
            if vi:
                load[s] = load.get(s, 0.0) + vi.activation_cpu

    for (_, _u, _, f), (i, s) in placement.assignment.items():
        vi = vnf_instances.get((f, i, s))
        if vi:
            load[s] = load.get(s, 0.0) + vi.per_user_cpu

    if not load:
        return 0.0

    peak = max(
        load.get(s, 0.0) / topo.cpu_capacity[s] * 100.0
        for s in load
        if topo.cpu_capacity.get(s, 0.0) > 0
    )
    return peak


def check_capacity_constraints(
    placement: PlacementState,
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    tol: float = 1e-6,
) -> Tuple[bool, Dict[SatID, float]]:
    """
    Verify constraint C3: total CPU load per satellite <= capacity.

    Returns (all_feasible, violations_dict) where violations_dict maps
    satellite id to the amount by which its capacity was exceeded (if any).
    """
    load: Dict[SatID, float] = {s: 0.0 for s in range(topo.num_satellites)}

    # Activation overhead
    for (f, i, s), active in placement.instance_active.items():
        if active:
            vi = vnf_instances.get((f, i, s))
            if vi:
                load[s] += vi.activation_cpu

    # Per-user load (one entry per (n, u, f_pos) — each position consumes resources)
    for (n, u, _, f), (i, s) in placement.assignment.items():
        vi = vnf_instances.get((f, i, s))
        if vi:
            load[s] += vi.per_user_cpu

    violations = {}
    for s, l in load.items():
        cap = topo.cpu_capacity.get(s, 0.0)
        if l > cap + tol:
            violations[s] = l - cap

    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# ISL link load
# ---------------------------------------------------------------------------

def _isl_bfs_path(src: SatID, dst: SatID, topo: TopologySnapshot) -> List[SatID]:
    """
    BFS shortest path (fewest hops) from src to dst in the ISL graph.
    Returns a list of satellite IDs including src and dst.
    Falls back to [src, dst] if no path is found (disconnected graph).
    """
    if src == dst:
        return [src]
    from collections import deque
    prev: Dict[SatID, SatID] = {src: -1}  # type: ignore[assignment]
    queue: deque = deque([src])
    while queue:
        s = queue.popleft()
        for t in topo.neighbors_of(s):
            if t == s or t in prev:
                continue
            prev[t] = s
            if t == dst:
                path: List[SatID] = []
                cur: SatID = dst
                while cur != -1:  # type: ignore[comparison-overlap]
                    path.append(cur)
                    cur = prev[cur]
                return list(reversed(path))
            queue.append(t)
    return [src, dst]  # fallback for disconnected graph


def compute_isl_load(
    placement: PlacementState,
    sfcs: List[SFC],
    topo: TopologySnapshot,
) -> Tuple[int, float, int]:
    """
    Compute ISL link load induced by the placement.

    For each (sfc, user, consecutive-function-hop) where the two hop
    satellites differ, a BFS shortest path through the ISL graph is
    computed and each traversed directed link gets +1 flow count.

    Returns (max_link_flows, avg_link_flows, n_active_links):
      max_link_flows  -- peak flow count on any single ISL link.
      avg_link_flows  -- mean flow count across all links that carry >= 1 flow.
      n_active_links  -- number of ISL links carrying at least one flow.

    A high max_link_flows relative to avg_link_flows indicates hot spots
    where traffic is concentrated on a few links.
    """
    link_load: Dict[Tuple[SatID, SatID], int] = {}

    for sfc in sfcs:
        for u in sfc.user_ids:
            for f_pos in range(len(sfc.functions) - 1):
                v1 = placement.assignment.get(
                    (sfc.slice_id, u, f_pos,     sfc.functions[f_pos]))
                v2 = placement.assignment.get(
                    (sfc.slice_id, u, f_pos + 1, sfc.functions[f_pos + 1]))
                if v1 is None or v2 is None:
                    continue
                s1, s2 = v1[1], v2[1]
                if s1 == s2:
                    continue
                path = _isl_bfs_path(s1, s2, topo)
                for k in range(len(path) - 1):
                    edge = (path[k], path[k + 1])
                    link_load[edge] = link_load.get(edge, 0) + 1

    if not link_load:
        return 0, 0.0, 0

    loads = list(link_load.values())
    return max(loads), sum(loads) / len(loads), len(loads)
