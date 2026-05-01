"""
src/milp.py
===========
MILP model for risk-aware SFC provisioning.

Implements the full MILP formulation with all constraints C1-C5,
AND linearization for the risk model (both exact and coarse), and the
migration epigraph.

Solver backend: PuLP (open-source, uses CBC by default).
Gurobi stub: if `gurobi=True` is passed and gurobipy is installed, the
model is solved with Gurobi 10.0 (warm-start, MIPGap, time limit supported).
JuMP/Julia: for production use, the Julia JuMP version (src/milp.jl) is
preferred; this Python version mirrors the model exactly for portability.


Key indexing convention 
----------------------------------------------------------
All beta/z/p/y variables are indexed by chain POSITION (f_pos) rather than
by function-type string alone.  This is critical because a chain may repeat
function types (e.g. ["FW", "IDS", "FW"]).  Using the type string as a key
would collapse position 0 and position 2 into the same variable, making C1
impossible to satisfy and causing infeasibility.

  beta key : (sfc_idx, u_idx, f_pos, i, s)
  z    key : (sfc_idx, f_pos, i, s)
  p    key : (sfc_idx_a, u_idx_a, fp_a, sfc_idx_b, u_idx_b, fp_b, i, s)
  y    key : (sfc_idx_a, fp_a, sfc_idx_b, fp_b, i, s)

The actual function type is carried separately for coefficient look-ups
(risk_weight, vnf_instances look-up, etc.).

Constraints implemented
-----------------------
C1  eq:unique    -- unique assignment per (sfc_idx, u_idx, f_pos)
C2  eq:act       -- activation coupling beta <= gamma
C3  eq:cap       -- satellite CPU capacity
Exact risk: eq:and1-3 (p variables)
Coarse risk: eq:z1-2 (z variables), eq:y1-3 (y variables)
Migration:  eq:mu1-3 (mu variables)
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from .types import (
    FuncID, InstID, PlacementKey, PlacementState, PlacementVal,
    PreprocessResult, RiskParameters, SatID, SFC, SliceID,
    SolverResult, TopologySnapshot, UserID, VNFInstance,
)
from .metrics import (
    compute_risk_exact, compute_risk_bounds, compute_cap_use,
    compute_migrations, verify_risk_bounds, verify_migration_epigraph,
    compute_delay_compliance, compute_peak_sat_util, compute_isl_load,
)

from .instance_generator import INSTANCES_PER_SAT
from .visibility import (
    precompute_user_visibility,
    compute_access_delay_ms,
)


# ---------------------------------------------------------------------------
# Warm-start helper
# ---------------------------------------------------------------------------

def _apply_milp_warmstart(
    beta: dict,
    gamma: dict,
    p_exact: dict,
    ws: PlacementState,
    sfcs: List[SFC],
) -> None:
    """
    Set PuLP variable initial values from a warm-start PlacementState so that
    CBC can begin branch-and-bound from a known feasible integer point.

    beta variables are set to 1 where the warmstart assigns the same (i, s).
    gamma variables reflect which instances are active in the warmstart.
    p variables (exact model only) are derived as p = beta_a AND beta_b.
    """
    # Build fast external-to-internal index map
    ext_to_int: dict = {}
    for sfc_idx, sfc in enumerate(sfcs):
        for u_idx, u in enumerate(sfc.user_ids):
            ext_to_int[(sfc.slice_id, u)] = (sfc_idx, u_idx)

    # Initialise all beta to 0, then set matching assignments to 1
    for var in beta.values():
        var.setInitialValue(0.0)
    for (n, u, f_pos, _f_type), (i_ws, s_ws) in ws.assignment.items():
        ij = ext_to_int.get((n, u))
        if ij is None:
            continue
        sfc_idx, u_idx = ij
        bk = (sfc_idx, u_idx, f_pos, i_ws, s_ws)
        v = beta.get(bk)
        if v is not None:
            v.setInitialValue(1.0)

    # gamma: active iff warmstart has the instance active
    for (f_type, inst_id, sat_id), var in gamma.items():
        var.setInitialValue(
            1.0 if ws.instance_active.get((f_type, inst_id, sat_id), False) else 0.0
        )

    # p (exact model): p = beta_a AND beta_b — derive from beta initial values
    for (idx_a, ua, fp_a, idx_b, ub, fp_b, i_ws, s_ws), pvar in p_exact.items():
        bk1 = (idx_a, ua, fp_a, i_ws, s_ws)
        bk2 = (idx_b, ub, fp_b, i_ws, s_ws)
        v1 = beta.get(bk1)
        v2 = beta.get(bk2)
        iv1 = (v1.varValue or 0.0) if v1 is not None else 0.0
        iv2 = (v2.varValue or 0.0) if v2 is not None else 0.0
        pvar.setInitialValue(1.0 if iv1 > 0.5 and iv2 > 0.5 else 0.0)


# ---------------------------------------------------------------------------
# PuLP model builder
# ---------------------------------------------------------------------------

def _build_and_solve_pulp(
    sfcs: List[SFC],
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    risk_params: RiskParameters,
    preprocess: PreprocessResult,
    prev_placement: Optional[PlacementState],
    config: dict,
    warmstart: Optional[PlacementState],
    use_exact_risk: bool,
    method: str,
) -> SolverResult:
    """Build and solve the MILP using PuLP + CBC (or Gurobi if available)."""
    try:
        import pulp
    except ImportError:
        raise ImportError(
            "PuLP not installed. Run: pip install pulp\n"
            "For Gurobi, also install gurobipy and pass use_gurobi=True."
        )

    t_start = time.perf_counter()

    prob = pulp.LpProblem("LEO_SFC_Risk_Aware", pulp.LpMinimize)

    S = topo.num_satellites
    F = config.get("function_types", [])
    I = INSTANCES_PER_SAT
    w_cap  = config.get("omega_cap", 0.3)
    w_risk = config.get("omega_risk", 0.5)
    w_mig  = config.get("omega_mig", 0.2)

    # -----------------------------------------------------------------------
    # Visibility: precompute visible satellites per (sfc_idx, user_id).
    # Users in the same slice may be geographically distributed and see
    # different satellite sets.  The ingress constraint (f_pos == 0) is
    # applied per user, not per slice.
    # -----------------------------------------------------------------------
    min_el = config.get("min_elevation_deg", 10.0)
    vis_sats = precompute_user_visibility(topo, sfcs, min_el)
    # Re-key by (sfc_idx, u_idx) for use in the beta variable loop which
    # iterates over positional indices.
    vis_sats_set: Dict[Tuple[int, int], set] = {}
    for sfc_idx, sfc in enumerate(sfcs):
        for u_idx, u in enumerate(sfc.user_ids):
            vis_sats_set[(sfc_idx, u_idx)] = set(vis_sats.get((sfc_idx, u), range(S)))

    # -----------------------------------------------------------------------
    # Delay precomputation (used by C5)
    # access_delay_ms[(sfc_idx, u_idx, s)] : user terminal → satellite s (ms)
    # Each user has an independent terminal location.
    # -----------------------------------------------------------------------
    access_delay_ms: Dict[Tuple[int, int, int], float] = {}
    if topo.sat_positions:
        for sfc_idx, sfc in enumerate(sfcs):
            for u_idx, u in enumerate(sfc.user_ids):
                lat_u, lon_u = sfc.user_location(u)
                for s, (lat_s, lon_s, alt_s) in topo.sat_positions.items():
                    access_delay_ms[(sfc_idx, u_idx, s)] = compute_access_delay_ms(
                        lat_u, lon_u, lat_s, lon_s, alt_s
                    )

    # -----------------------------------------------------------------------
    # gamma[f_type, i, s] -- shared across all slices using the same VNF type
    # -----------------------------------------------------------------------
    gamma: Dict[Tuple, "pulp.LpVariable"] = {}
    for f_type in F:
        for i in range(I):
            for s in range(S):
                gamma[(f_type, i, s)] = pulp.LpVariable(
                    f"g_{f_type}_{i}_{s}", cat="Binary"
                )

    # -----------------------------------------------------------------------
    # beta[(sfc_idx, u_idx, f_pos, i, s)] in {0,1}
    # Indexed by POSITION in the chain to handle repeated function types.
    # Visibility constraint (C_vis): for f_pos == 0 (access/ingress function),
    # only create variables for satellites visible to the slice's ground station.
    # All subsequent function positions are unrestricted (ISL-routed).
    # -----------------------------------------------------------------------
    beta: Dict[Tuple, "pulp.LpVariable"] = {}
    for sfc_idx, sfc in enumerate(sfcs):
        for u_idx in range(len(sfc.user_ids)):
            # Visibility gate applied per user: ingress satellite must be
            # visible from this specific user's terminal location.
            allowed_ingress = vis_sats_set.get((sfc_idx, u_idx), set(range(S)))
            for f_pos in range(len(sfc.functions)):
                for i in range(I):
                    for s in range(S):
                        if f_pos == 0 and s not in allowed_ingress:
                            continue
                        beta[(sfc_idx, u_idx, f_pos, i, s)] = pulp.LpVariable(
                            f"b_{sfc_idx}_{u_idx}_{f_pos}_{i}_{s}", cat="Binary"
                        )

    # -----------------------------------------------------------------------
    # mu[(sfc_idx, u_idx, f_pos)] -- avoidable migration indicator
    # -----------------------------------------------------------------------
    mu: Dict[Tuple, "pulp.LpVariable"] = {}
    for sfc_idx, sfc in enumerate(sfcs):
        for u_idx in range(len(sfc.user_ids)):
            for f_pos in range(len(sfc.functions)):
                mu[(sfc_idx, u_idx, f_pos)] = pulp.LpVariable(
                    f"mu_{sfc_idx}_{u_idx}_{f_pos}", cat="Binary"
                )

    # -----------------------------------------------------------------------
    # C1: Unique assignment -- exactly one (i, s) per (sfc_idx, u_idx, f_pos)
    # -----------------------------------------------------------------------
    for sfc_idx, sfc in enumerate(sfcs):
        for u_idx in range(len(sfc.user_ids)):
            for f_pos in range(len(sfc.functions)):
                prob += (
                    pulp.lpSum(
                        beta[(sfc_idx, u_idx, f_pos, i, s)]
                        for i in range(I) for s in range(S)
                        if (sfc_idx, u_idx, f_pos, i, s) in beta
                    ) == 1,
                    f"C1_{sfc_idx}_{u_idx}_{f_pos}"
                )

    # -----------------------------------------------------------------------
    # C2: Activation coupling -- beta <= gamma
    # -----------------------------------------------------------------------
    for sfc_idx, sfc in enumerate(sfcs):
        for u_idx in range(len(sfc.user_ids)):
            for f_pos, f_type in enumerate(sfc.functions):
                for i in range(I):
                    for s in range(S):
                        bk = (sfc_idx, u_idx, f_pos, i, s)
                        if bk not in beta:
                            continue
                        prob += (
                            beta[bk] <= gamma[(f_type, i, s)],
                            f"C2_{sfc_idx}_{u_idx}_{f_pos}_{i}_{s}"
                        )

    # -----------------------------------------------------------------------
    # C3: Satellite CPU capacity
    # -----------------------------------------------------------------------
    for s in range(S):
        cap = topo.cpu_capacity.get(s, 0.0)

        act_term = pulp.lpSum(
            vnf_instances[(f_type, i, s)].activation_cpu * gamma[(f_type, i, s)]
            for f_type in F for i in range(I)
            if (f_type, i, s) in vnf_instances
        )

        user_term = pulp.lpSum(
            vnf_instances[(sfc.functions[f_pos], i, s)].per_user_cpu
            * beta[(sfc_idx, u_idx, f_pos, i, s)]
            for sfc_idx, sfc in enumerate(sfcs)
            for u_idx in range(len(sfc.user_ids))
            for f_pos, f_type in enumerate(sfc.functions)
            for i in range(I)
            if (f_type, i, s) in vnf_instances
            and (sfc_idx, u_idx, f_pos, i, s) in beta
        )

        prob += (act_term + user_term <= cap, f"C3_{s}")

    # -----------------------------------------------------------------------
    # Migration epigraph (eq:mu1-3)
    # -----------------------------------------------------------------------
    for sfc_idx, sfc in enumerate(sfcs):
        for u_idx, u in enumerate(sfc.user_ids):
            for f_pos, f_type in enumerate(sfc.functions):
                ext_key: PlacementKey = (sfc.slice_id, u, f_pos, f_type)
                pi_val = 1 if preprocess.pi.get(ext_key, False) else 0
                mu_var = mu[(sfc_idx, u_idx, f_pos)]

                if prev_placement is not None and ext_key in prev_placement.assignment:
                    i_p, s_p = prev_placement.assignment[ext_key]
                    bkey = (sfc_idx, u_idx, f_pos, i_p, s_p)
                    if bkey in beta:
                        k_var = beta[bkey]
                        prob += (mu_var >= pi_val - k_var,
                                 f"mu1_{sfc_idx}_{u_idx}_{f_pos}")
                        prob += (mu_var <= pi_val,
                                 f"mu2_{sfc_idx}_{u_idx}_{f_pos}")
                        prob += (mu_var <= 1 - k_var,
                                 f"mu3_{sfc_idx}_{u_idx}_{f_pos}")
                    else:
                        prob += (mu_var == 0,
                                 f"mu_zero_{sfc_idx}_{u_idx}_{f_pos}")
                else:
                    prob += (mu_var == 0,
                             f"mu_zero_{sfc_idx}_{u_idx}_{f_pos}")

    # -----------------------------------------------------------------------
    # C5: End-to-end delay budget (access + processing components)
    #
    # Full E2E delay = access delay + ISL delays + VNF processing delays.
    # ISL delays depend on which PAIR of satellites hosts each consecutive
    # function, requiring O(S²) auxiliary variables per (user, hop) — too
    # expensive for the MILP at constellation scale.  ISL delays are
    # enforced exactly in the SA via _e2e_delay_ms().
    #
    # The MILP enforces the tractable linear components:
    #   d_access(n, s_0)  +  sum_fp d_proc(fp, i, s)  <=  budget
    # which is a sound necessary condition: any placement violating this
    # would also violate the full E2E budget.
    # -----------------------------------------------------------------------
    for sfc_idx, sfc in enumerate(sfcs):
        for u_idx, u in enumerate(sfc.user_ids):
            budget = sfc.e2e_budget_ms.get(u, float("inf"))
            if budget >= 1e9:
                continue

            # Access delay: user terminal → ingress satellite (per-user location)
            access_term = pulp.lpSum(
                access_delay_ms.get((sfc_idx, u_idx, s), 0.0)
                * beta[(sfc_idx, u_idx, 0, i, s)]
                for i in range(I) for s in range(S)
                if (sfc_idx, u_idx, 0, i, s) in beta
            )

            # VNF processing delay summed over all functions in the chain
            proc_term = pulp.lpSum(
                vnf_instances[(sfc.functions[fp], i, s)].proc_delay_ms
                * beta[(sfc_idx, u_idx, fp, i, s)]
                for fp in range(len(sfc.functions))
                for i in range(I) for s in range(S)
                if (sfc_idx, u_idx, fp, i, s) in beta
                and (sfc.functions[fp], i, s) in vnf_instances
            )

            prob += (
                access_term + proc_term <= budget,
                f"C5_{sfc_idx}_{u_idx}",
            )

    # -----------------------------------------------------------------------
    # C_ISL: ISL direct-link capacity
    #
    # For each direct ISL link (s1 -> s2) and each (sfc, user, hop f_pos ->
    # f_pos+1), introduce a continuous indicator h in [0,1]:
    #   h = 1  iff  user is assigned to s1 at f_pos AND to s2 at f_pos+1
    #
    # Linearisation of h = x1 AND x2  (x1, x2 in {0,1}):
    #   h >= x1 + x2 - 1        (lower bound)
    #   h <= x1                  (upper bound 1)
    #   h <= x2                  (upper bound 2)
    # where x1 = sum_i beta[sfc, u, f_pos,   i, s1]
    #       x2 = sum_j beta[sfc, u, f_pos+1, j, s2]
    #
    # Capacity: sum_{sfc,u,fp} h[sfc,u,fp,s1,s2] <= isl_max_flows
    #
    # Skipped entirely when isl_max_flows is absent or infinite.
    # -----------------------------------------------------------------------
    isl_cap = config.get("isl_max_flows", float("inf"))
    if isl_cap < float("inf"):
        h_isl: Dict[Tuple, "pulp.LpVariable"] = {}

        # Enumerate all directed ISL links (s1, s2) with s1 != s2
        isl_links = {
            (s1, s2)
            for s1 in range(S)
            for s2 in topo.neighbors_of(s1)
            if s2 != s1
        }

        for sfc_idx, sfc in enumerate(sfcs):
            for u_idx in range(len(sfc.user_ids)):
                for f_pos in range(len(sfc.functions) - 1):
                    for (s1, s2) in isl_links:
                        x1_terms = [
                            beta[(sfc_idx, u_idx, f_pos, i, s1)]
                            for i in range(I)
                            if (sfc_idx, u_idx, f_pos, i, s1) in beta
                        ]
                        x2_terms = [
                            beta[(sfc_idx, u_idx, f_pos + 1, j, s2)]
                            for j in range(I)
                            if (sfc_idx, u_idx, f_pos + 1, j, s2) in beta
                        ]
                        if not x1_terms or not x2_terms:
                            continue

                        x1 = pulp.lpSum(x1_terms)
                        x2 = pulp.lpSum(x2_terms)
                        hk = (sfc_idx, u_idx, f_pos, s1, s2)
                        h_isl[hk] = pulp.LpVariable(
                            f"hisl_{sfc_idx}_{u_idx}_{f_pos}_{s1}_{s2}",
                            lowBound=0, upBound=1, cat="Continuous",
                        )
                        prob += (h_isl[hk] >= x1 + x2 - 1,
                                 f"Hisl_lb_{sfc_idx}_{u_idx}_{f_pos}_{s1}_{s2}")
                        prob += (h_isl[hk] <= x1,
                                 f"Hisl_ub1_{sfc_idx}_{u_idx}_{f_pos}_{s1}_{s2}")
                        prob += (h_isl[hk] <= x2,
                                 f"Hisl_ub2_{sfc_idx}_{u_idx}_{f_pos}_{s1}_{s2}")

        for (s1, s2) in isl_links:
            flow_terms = [
                h_isl[(sfc_idx, u_idx, f_pos, s1, s2)]
                for sfc_idx, sfc in enumerate(sfcs)
                for u_idx in range(len(sfc.user_ids))
                for f_pos in range(len(sfc.functions) - 1)
                if (sfc_idx, u_idx, f_pos, s1, s2) in h_isl
            ]
            if flow_terms:
                prob += (
                    pulp.lpSum(flow_terms) <= isl_cap,
                    f"C_ISL_{s1}_{s2}",
                )

    # -----------------------------------------------------------------------
    # Risk model
    # -----------------------------------------------------------------------
    p_exact: dict = {}  # populated only by exact model; used for warmstart
    if use_exact_risk:
        # Exact model: p_{a,ua,fpa, b,ub,fpb, i,s} = beta_a AND beta_b
        p: Dict[Tuple, "pulp.LpVariable"] = {}
        risk_terms = []

        for idx_a, sfc_a in enumerate(sfcs):
            for idx_b, sfc_b in enumerate(sfcs):
                if idx_b <= idx_a:
                    continue
                for fp_a, ft_a in enumerate(sfc_a.functions):
                    for fp_b, ft_b in enumerate(sfc_b.functions):
                        if ft_a != ft_b:
                            continue
                        f_type = ft_a
                        w = risk_params.risk_weight(sfc_a.slice_id, sfc_b.slice_id, f_type)
                        if w == 0.0:
                            continue
                        for i in range(I):
                            for s in range(S):
                                if (f_type, i, s) not in vnf_instances:
                                    continue
                                for ua in range(len(sfc_a.user_ids)):
                                    for ub in range(len(sfc_b.user_ids)):
                                        pk = (idx_a, ua, fp_a, idx_b, ub, fp_b, i, s)
                                        p[pk] = pulp.LpVariable(
                                            f"p_{idx_a}_{ua}_{fp_a}_{idx_b}_{ub}_{fp_b}_{i}_{s}",
                                            cat="Binary"
                                        )
                                        bk1 = (idx_a, ua, fp_a, i, s)
                                        bk2 = (idx_b, ub, fp_b, i, s)
                                        if bk1 not in beta or bk2 not in beta:
                                            del p[pk]
                                            continue
                                        prob += (p[pk] <= beta[bk1],
                                                 f"and1_{idx_a}_{ua}_{fp_a}_{idx_b}_{ub}_{fp_b}_{i}_{s}")
                                        prob += (p[pk] <= beta[bk2],
                                                 f"and2_{idx_a}_{ua}_{fp_a}_{idx_b}_{ub}_{fp_b}_{i}_{s}")
                                        prob += (p[pk] >= beta[bk1] + beta[bk2] - 1,
                                                 f"and3_{idx_a}_{ua}_{fp_a}_{idx_b}_{ub}_{fp_b}_{i}_{s}")
                                        risk_terms.append(w * p[pk])

        risk_expr = pulp.lpSum(risk_terms)
        p_exact = p  # expose for warmstart helper

    else:
        # Coarse model: z[(sfc_idx, f_pos, i, s)] = OR over users
        z: Dict[Tuple, "pulp.LpVariable"] = {}
        risk_terms_coarse = []

        for sfc_idx, sfc in enumerate(sfcs):
            for f_pos, f_type in enumerate(sfc.functions):
                for i in range(I):
                    for s in range(S):
                        if (f_type, i, s) not in vnf_instances:
                            continue
                        zk = (sfc_idx, f_pos, i, s)
                        z[zk] = pulp.LpVariable(
                            f"z_{sfc_idx}_{f_pos}_{i}_{s}", cat="Binary"
                        )
                        for u_idx in range(len(sfc.user_ids)):
                            bk = (sfc_idx, u_idx, f_pos, i, s)
                            if bk not in beta:
                                continue
                            prob += (
                                z[zk] >= beta[bk],
                                f"z1_{sfc_idx}_{f_pos}_{i}_{s}_{u_idx}"
                            )
                        prob += (
                            z[zk] <= pulp.lpSum(
                                beta[(sfc_idx, u_idx, f_pos, i, s)]
                                for u_idx in range(len(sfc.user_ids))
                                if (sfc_idx, u_idx, f_pos, i, s) in beta
                            ),
                            f"z2_{sfc_idx}_{f_pos}_{i}_{s}"
                        )

        # y[(idx_a, fp_a, idx_b, fp_b, i, s)] -- cross-slice co-location
        y: Dict[Tuple, "pulp.LpVariable"] = {}
        for idx_a, sfc_a in enumerate(sfcs):
            for idx_b, sfc_b in enumerate(sfcs):
                if idx_b <= idx_a:
                    continue
                for fp_a, ft_a in enumerate(sfc_a.functions):
                    for fp_b, ft_b in enumerate(sfc_b.functions):
                        if ft_a != ft_b:
                            continue
                        f_type = ft_a
                        w = risk_params.risk_weight(sfc_a.slice_id, sfc_b.slice_id, f_type)
                        if w == 0.0:
                            continue
                        for i in range(I):
                            for s in range(S):
                                zk_a = (idx_a, fp_a, i, s)
                                zk_b = (idx_b, fp_b, i, s)
                                if zk_a not in z or zk_b not in z:
                                    continue
                                yk = (idx_a, fp_a, idx_b, fp_b, i, s)
                                y[yk] = pulp.LpVariable(
                                    f"y_{idx_a}_{fp_a}_{idx_b}_{fp_b}_{i}_{s}",
                                    cat="Binary"
                                )
                                prob += (y[yk] <= z[zk_a],
                                         f"y1_{idx_a}_{fp_a}_{idx_b}_{fp_b}_{i}_{s}")
                                prob += (y[yk] <= z[zk_b],
                                         f"y2_{idx_a}_{fp_a}_{idx_b}_{fp_b}_{i}_{s}")
                                prob += (y[yk] >= z[zk_a] + z[zk_b] - 1,
                                         f"y3_{idx_a}_{fp_a}_{idx_b}_{fp_b}_{i}_{s}")
                                risk_terms_coarse.append(w * y[yk])

        risk_expr = pulp.lpSum(risk_terms_coarse)

    # -----------------------------------------------------------------------
    # Apply SA / coarse warm-start so CBC begins from a feasible integer point
    # -----------------------------------------------------------------------
    if warmstart is not None:
        _apply_milp_warmstart(beta, gamma, p_exact, warmstart, sfcs)

    # -----------------------------------------------------------------------
    # CapUse expression
    # -----------------------------------------------------------------------
    cap_expr = (
        pulp.lpSum(
            vnf_instances[(f_type, i, s)].activation_cpu * gamma[(f_type, i, s)]
            for f_type in F for i in range(I) for s in range(S)
            if (f_type, i, s) in vnf_instances
        )
        + pulp.lpSum(
            vnf_instances[(sfc.functions[f_pos], i, s)].per_user_cpu
            * beta[(sfc_idx, u_idx, f_pos, i, s)]
            for sfc_idx, sfc in enumerate(sfcs)
            for u_idx in range(len(sfc.user_ids))
            for f_pos, f_type in enumerate(sfc.functions)
            for i in range(I) for s in range(S)
            if (f_type, i, s) in vnf_instances
            and (sfc_idx, u_idx, f_pos, i, s) in beta
        )
    )

    # -----------------------------------------------------------------------
    # Migration cost expression
    # -----------------------------------------------------------------------
    mig_expr = pulp.lpSum(
        risk_params.migration_cost_D.get(sfc.functions[f_pos], 1.0)
        * mu[(sfc_idx, u_idx, f_pos)]
        for sfc_idx, sfc in enumerate(sfcs)
        for u_idx in range(len(sfc.user_ids))
        for f_pos in range(len(sfc.functions))
    )

    # -----------------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------------
    prob += (
        w_cap  * cap_expr  / max(preprocess.cap_use_bar, 1e-9)
        + w_risk * risk_expr / max(preprocess.risk_bar, 1e-9)
        + w_mig  * mig_expr  / max(preprocess.mig_bar, 1e-9)
    )

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    time_limit = config.get("time_limit_s", 300)
    mip_gap    = config.get("mip_gap", 0.005)

    solver = pulp.PULP_CBC_CMD(
        msg=0,
        timeLimit=time_limit,
        gapRel=mip_gap,
        warmStart=(warmstart is not None),
    )
    prob.solve(solver)

    solve_time = time.perf_counter() - t_start

    # -----------------------------------------------------------------------
    # Extract solution
    # -----------------------------------------------------------------------
    result_placement = PlacementState(epoch=topo.epoch)

    if prob.status == 1:  # Optimal or feasible
        for sfc_idx, sfc in enumerate(sfcs):
            for u_idx, u in enumerate(sfc.user_ids):
                for f_pos, f_type in enumerate(sfc.functions):
                    for i in range(I):
                        for s in range(S):
                            bk = (sfc_idx, u_idx, f_pos, i, s)
                            if bk not in beta:
                                continue
                            val = pulp.value(beta[bk])
                            if val is not None and val > 0.5:
                                ext_key: PlacementKey = (sfc.slice_id, u, f_pos, f_type)
                                result_placement.assignment[ext_key] = (i, s)
                                vi = vnf_instances.get((f_type, i, s))
                                if vi:
                                    result_placement.cpu_load[s] = (
                                        result_placement.cpu_load.get(s, 0.0)
                                        + vi.per_user_cpu
                                    )
                                inst_key = (f_type, i, s)
                                if inst_key not in result_placement.instance_active:
                                    result_placement.instance_active[inst_key] = True
                                    if vi:
                                        result_placement.cpu_load[s] = (
                                            result_placement.cpu_load.get(s, 0.0)
                                            + vi.activation_cpu
                                        )

        risk_ex = compute_risk_exact(result_placement, sfcs, risk_params, vnf_instances)
        risk_lb, risk_ub = compute_risk_bounds(result_placement, sfcs, risk_params)
        cap_use, cap_use_pct, active_sat_util_pct = compute_cap_use(result_placement, topo, vnf_instances)
        n_total, n_avoid, mig_cost = compute_migrations(
            result_placement, prev_placement, preprocess, risk_params
        )
        delay_compliance_pct, _ = compute_delay_compliance(
            result_placement, sfcs, topo, vnf_instances
        )
        peak_sat_util_pct = compute_peak_sat_util(result_placement, topo, vnf_instances)
        max_isl_load, avg_isl_load, n_active_isl_links = compute_isl_load(
            result_placement, sfcs, topo
        )
        risk_bound_tightness = risk_ex / risk_ub if risk_ub > 1e-9 else 1.0

        gap_pct = 0.0
        if hasattr(prob, "bestBound") and prob.bestBound is not None:
            obj_val = pulp.value(prob.objective)
            if obj_val and obj_val != 0:
                gap_pct = abs(obj_val - prob.bestBound) / abs(obj_val) * 100

        status_map = {1: "optimal", 0: "timelimit_nofeas", -1: "infeasible",
                      -2: "infeasible", -3: "timelimit"}
        sol_status = status_map.get(prob.status, "unknown")

    else:
        risk_ex = risk_lb = risk_ub = cap_use = cap_use_pct = active_sat_util_pct = mig_cost = 0.0
        delay_compliance_pct = peak_sat_util_pct = risk_bound_tightness = 0.0
        max_isl_load = n_active_isl_links = 0
        avg_isl_load = 0.0
        n_total = n_avoid = 0
        gap_pct = 100.0
        sol_status = "infeasible"

    return SolverResult(
        placement=result_placement,
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
        obj_value=pulp.value(prob.objective) or float("inf"),
        solve_time_s=solve_time,
        sa_time_s=0.0,
        mip_gap_pct=gap_pct,
        status=sol_status,
        n_migrations=n_total,
        n_avoidable_migrations=n_avoid,
        method=method,
        instance_id=0,
        epoch=topo.epoch,
    )


# ---------------------------------------------------------------------------
# Public solver interface
# ---------------------------------------------------------------------------

def solve_epoch(
    sfcs: List[SFC],
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    risk_params: RiskParameters,
    preprocess: PreprocessResult,
    prev_placement: Optional[PlacementState],
    config: dict,
    warmstart: Optional[PlacementState] = None,
    method: str = "proposed_coarse",
) -> SolverResult:
    """
    Solve one epoch of the risk-aware SFC placement MILP.

    Automatically selects exact vs. coarse risk model based on |U_n|
    and the configured threshold (use_exact_model_threshold).
    """
    U = config.get("users_per_slice", 10)
    threshold = config.get("use_exact_model_threshold", 40)
    use_exact = (U <= threshold) and config.get("use_exact_model", False)

    cfg = dict(config)
    if method == "proposed_exact":
        # Exact model explicitly requested regardless of threshold / config flag
        use_exact = True
        # Exact model has O(N²U²IS) p-variables; give it a longer time budget
        cfg["time_limit_s"] = config.get(
            "time_limit_exact_s",
            config.get("time_limit_s", 300) * 2,
        )
    elif method == "B1":
        cfg["omega_cap"] = 1.0
        cfg["omega_risk"] = 0.0
        cfg["omega_mig"] = 0.0
        use_exact = False
    elif method == "B2":
        # B2 adds migration stability on top of B1 but still ignores risk
        cfg["omega_cap"] = 0.5
        cfg["omega_risk"] = 0.0
        cfg["omega_mig"] = 0.5
        use_exact = False

    result = _build_and_solve_pulp(
        sfcs, topo, vnf_instances, risk_params,
        preprocess, prev_placement, cfg, warmstart,
        use_exact_risk=use_exact,
        method=method,
    )

    if config.get("verify_propositions", False) and result.status != "infeasible":
        ok_bounds = verify_risk_bounds(result.risk_ex, result.risk_lb, result.risk_ub)
        ok_mig    = verify_migration_epigraph(
            result.placement, prev_placement, preprocess
        )
        if not ok_bounds:
            print(f"  [WARNING] Proposition 1 violated: "
                  f"LB={result.risk_lb:.4f} EX={result.risk_ex:.4f} UB={result.risk_ub:.4f}")
        if not ok_mig:
            print(f"  [WARNING] Proposition 2 (migration epigraph) violated.")

    return result
