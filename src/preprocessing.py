"""
src/preprocessing.py
====================
Preprocessing step executed once per epoch before the SA warm-start and
MILP solve.  Produces three outputs used by all downstream modules:

  1. pi_{n,u,f}    — feasibility filter for the previous placement
  2. Normalisation denominators (CapUse_bar, Risk_bar, Mig_bar)

The normalisation denominators are computed via lightweight LP relaxation
approximations rather than full LP solves (which require an LP solver).
The approximations are upper bounds guaranteed by construction.

Assumptions
-----------
- LP relaxation bounds are estimated analytically to avoid a circular
  dependency on the MILP solver.  Full LP solves can be substituted
  when a solver is available by calling the stub methods marked with
  `# STUB: replace with LP solve`.
- pi_{n,u,f} is True iff the target satellite from the previous epoch
  is still an ISL neighbor of at least one satellite visible to the slice
  AND still has sufficient residual CPU capacity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .types import (
    FuncID, InstID, PlacementKey, PlacementState, PreprocessResult,
    RiskParameters, SatID, SFC, SliceID, TopologySnapshot, UserID, VNFInstance,
)


def compute_pi(
    prev_placement: Optional[PlacementState],
    topo: TopologySnapshot,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    sfcs: List[SFC],
) -> Dict[PlacementKey, bool]:
    """
    Compute pi_{n,u,f} for all user-function pairs.

    pi_{n,u,f} = True iff the previous (i,s) assignment of (n,u,f):
      (a) The satellite s is still present in the current topology.
      (b) The satellite still has sufficient CPU headroom to keep the
          instance active (at least one additional user load unit).

    If prev_placement is None (first epoch), all pi values are False.
    Complexity: O(|N| * U * F).
    """
    pi: Dict[PlacementKey, bool] = {}

    if prev_placement is None:
        # First epoch — no previous placement, no avoidable migrations
        for sfc in sfcs:
            for u in sfc.user_ids:
                for f_pos, f in enumerate(sfc.functions):
                    pi[(sfc.slice_id, u, f_pos, f)] = False
        return pi

    # Build current residual CPU capacity after existing committed loads
    # (Use the previous placement's cpu_load as a proxy for current load)
    committed_load: Dict[SatID, float] = dict(prev_placement.cpu_load)

    for sfc in sfcs:
        for u in sfc.user_ids:
            for f_pos, f in enumerate(sfc.functions):
                key: PlacementKey = (sfc.slice_id, u, f_pos, f)
                prev_val = prev_placement.assignment.get(key)

                if prev_val is None:
                    pi[key] = False
                    continue

                i_prev, s_prev = prev_val

                # Condition (a): satellite exists in current topology
                if s_prev >= topo.num_satellites:
                    pi[key] = False
                    continue

                # Condition (b): sufficient residual CPU capacity
                vi = vnf_instances.get((f, i_prev, s_prev))
                if vi is None:
                    pi[key] = False
                    continue

                residual = topo.cpu_capacity.get(s_prev, 0.0) - committed_load.get(s_prev, 0.0)
                # Headroom: at least the per-user load for this user
                pi[key] = residual >= vi.per_user_cpu

    return pi


def compute_normalisation_bounds(
    topo: TopologySnapshot,
    sfcs: List[SFC],
    risk_params: RiskParameters,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    config: dict,
) -> Tuple[float, float, float]:
    """
    Estimate upper bounds for CapUse, Risk, and Mig used as normalisation
    denominators in the objective (eq:obj).

    These are analytical upper bounds (not LP solutions) that guarantee
    all normalized objective terms lie in [0, 1].

    CapUse_bar
    ----------
    Worst case: all instances active on all satellites + all users assigned.
    We cap this by the total CPU capacity (since CapUse <= sum Cap^cpu_s).

    Risk_bar
    --------
    Worst case: all slice pairs co-located on every instance with all users.
    sum_{n<n'} sum_f w^risk_{n,n',f} * |U_n| * |U_{n'}|
    (summed over all instances where both function types appear in the chains)

    Mig_bar
    -------
    Worst case: all user-function pairs migrate.
    sum_{n,u,f} Dis^mig_f

    Returns (CapUse_bar, Risk_bar, Mig_bar), each guaranteed >= 1.0.
    """
    # CapUse_bar: total satellite CPU capacity (hard upper bound)
    cap_use_bar = max(1.0, sum(topo.cpu_capacity.values()))

    # Risk_bar: sum over all cross-slice pairs of maximum risk per pair
    risk_bar = 0.0
    slice_ids = [sfc.slice_id for sfc in sfcs]
    U = config.get("users_per_slice", 10)
    function_types = config.get("function_types", [])

    for idx_a, n in enumerate(slice_ids):
        for n_prime in slice_ids[idx_a + 1:]:
            for f in function_types:
                w = risk_params.risk_weight(n, n_prime, f)
                risk_bar += w * U * U  # worst case: all users co-located

    risk_bar = max(1.0, risk_bar)

    # Mig_bar: all user-function pairs migrate
    mig_bar = 0.0
    for sfc in sfcs:
        for u in sfc.user_ids:
            for f in sfc.functions:
                d = risk_params.migration_cost_D.get(f, 1.0)
                mig_bar += d

    mig_bar = max(1.0, mig_bar)

    return cap_use_bar, risk_bar, mig_bar


def run_preprocessing(
    prev_placement: Optional[PlacementState],
    topo: TopologySnapshot,
    sfcs: List[SFC],
    risk_params: RiskParameters,
    vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    config: dict,
) -> PreprocessResult:
    """
    Main preprocessing entry point called once per epoch.

    Returns a PreprocessResult with pi, cap_use_bar, risk_bar, mig_bar.
    """
    pi = compute_pi(prev_placement, topo, vnf_instances, sfcs)
    cap_bar, risk_bar, mig_bar = compute_normalisation_bounds(
        topo, sfcs, risk_params, vnf_instances, config
    )
    return PreprocessResult(
        pi=pi,
        cap_use_bar=cap_bar,
        risk_bar=risk_bar,
        mig_bar=mig_bar,
    )
