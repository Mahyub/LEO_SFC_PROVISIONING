"""
src/types.py
============
Shared dataclasses and type definitions for the LEO-SFC system.

Every module imports from here; nothing else imports from module internals,
keeping the dependency graph acyclic and interfaces clean.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, FrozenSet
import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
SatID   = int   # satellite index in [0, |S|)
SliceID = int   # slice index in [0, n_s)
UserID  = int   # user index within a slice, in [0, |U_n|)
FuncID  = str   # VNF type string, e.g. "FW", "IDS", "ENC", "TM"
InstID  = int   # instance index per (function, satellite)

# A placement key uniquely identifies one user-function-position assignment
PlacementKey = Tuple[SliceID, UserID, int, FuncID]  # (n, u, f_pos, f_type)
# A placement value is the chosen (instance, satellite) pair
PlacementVal = Tuple[InstID, SatID]


# ---------------------------------------------------------------------------
# Topology snapshot (one epoch)
# ---------------------------------------------------------------------------
@dataclass
class TopologySnapshot:
    """
    Represents the LEO constellation graph at one orchestration epoch.

    Fields
    ------
    epoch : int
        Epoch index (0-based).
    num_satellites : int
        |S| — total number of satellites.
    isl_neighbors : Dict[SatID, List[SatID]]
        Adjacency list. isl_neighbors[s] = list of satellites reachable via
        one ISL hop from s.  Includes s itself (delta_{s,s}=0 for intra-sat).
    isl_delay_ms : Dict[Tuple[SatID,SatID], float]
        Propagation delay in milliseconds for each directed ISL pair (s, s').
        isl_delay_ms[(s, s)] = 0 for intra-satellite "hops".
    cpu_capacity : Dict[SatID, float]
        Cap^cpu_s for each satellite, in abstract CPU units.
    """
    epoch: int
    num_satellites: int
    isl_neighbors: Dict[SatID, List[SatID]]
    isl_delay_ms: Dict[Tuple[SatID, SatID], float]
    cpu_capacity: Dict[SatID, float]
    # Satellite positions: {sat_id: (lat_deg, lon_deg, alt_km)}.
    # Empty dict means no position data available (visibility skipped).
    sat_positions: Dict[SatID, Tuple[float, float, float]] = field(default_factory=dict)

    def neighbors_of(self, s: SatID) -> List[SatID]:
        """Return ISL neighbors (including self for intra-satellite hops)."""
        return self.isl_neighbors.get(s, [s])

    def delay(self, s: SatID, t: SatID) -> float:
        """Return ISL delay from s to t (0 if s == t)."""
        if s == t:
            return 0.0
        return self.isl_delay_ms.get((s, t), float("inf"))


# ---------------------------------------------------------------------------
# Service function chain definition
# ---------------------------------------------------------------------------
@dataclass
class SFC:
    """
    Ordered security service function chain for one slice.

    Fields
    ------
    slice_id : SliceID
    functions : List[FuncID]
        Ordered VNF types F_n = (f^n_1, ..., f^n_{L_n}).
    user_ids : List[UserID]
        U_n — user indices within this slice.
    e2e_budget_ms : Dict[UserID, float]
        Per-user E2E delay budget T_bar_{n,u}.
    criticality : float
        C[n] drawn from [1, 3].
    ground_lat, ground_lon : float
        Representative slice-centre coordinates (WGS-84 degrees).
        Used for display and as a fallback when user_locations is empty.
    user_locations : Dict[UserID, Tuple[float, float]]
        Per-user terminal location (lat_deg, lon_deg).  Users in the same
        slice may be geographically distributed and therefore experience
        different satellite visibility conditions each epoch.  When empty,
        all users fall back to (ground_lat, ground_lon).
    """
    slice_id: SliceID
    functions: List[FuncID]
    user_ids: List[UserID]
    e2e_budget_ms: Dict[UserID, float]
    criticality: float
    ground_lat: float = 0.0
    ground_lon: float = 0.0
    user_locations: Dict[UserID, Tuple[float, float]] = field(default_factory=dict)

    def user_location(self, u: UserID) -> Tuple[float, float]:
        """Return (lat_deg, lon_deg) for user u, falling back to slice centre."""
        return self.user_locations.get(u, (self.ground_lat, self.ground_lon))

    @property
    def length(self) -> int:
        """L_n — number of VNFs in the chain."""
        return len(self.functions)

    @property
    def consecutive_pairs(self) -> List[Tuple[FuncID, FuncID]]:
        """Pi_n — consecutive function pairs (f^n_l, f^n_{l+1})."""
        return [(self.functions[i], self.functions[i + 1])
                for i in range(len(self.functions) - 1)]


# ---------------------------------------------------------------------------
# Risk model parameters
# ---------------------------------------------------------------------------
@dataclass
class RiskParameters:
    """
    Encapsulates all risk model parameters.

    Fields
    ------
    sensitivity_R : Dict[FuncID, float]
        R[f] — function sensitivity (NIST SP 800-53 Low/Moderate/High mapped).
    criticality_C : Dict[SliceID, float]
        C[n] — slice criticality drawn from [1, 3].
    policy_phi : Dict[Tuple[SliceID,SliceID], float]
        Phi[n,n'] — bilateral isolation policy coefficient in [0, 1].
        Symmetric: phi[(n,n')] == phi[(n',n)] for n != n'.
    migration_cost_D : Dict[FuncID, float]
        Dis^mig_f — per-function migration disruption cost.
    """
    sensitivity_R: Dict[FuncID, float]
    criticality_C: Dict[SliceID, float]
    policy_phi: Dict[Tuple[SliceID, SliceID], float]
    migration_cost_D: Dict[FuncID, float]

    def risk_weight(self, n: SliceID, np_: SliceID, f: FuncID) -> float:
        """
        Compute w^risk_{n,n',f} = R[f] * Phi[n,n'] * C[n] * C[n'].

        The multiplicative structure means any zero factor (zero sensitivity,
        zero criticality product, or zero isolation weight) collapses the
        entire risk contribution to zero, consistent with ISO 31000:2018.
        """
        R   = self.sensitivity_R.get(f, 0.0)
        Phi = self.policy_phi.get((min(n, np_), max(n, np_)), 0.5)
        Cn  = self.criticality_C.get(n, 1.0)
        Cnp = self.criticality_C.get(np_, 1.0)
        return R * Phi * Cn * Cnp


# ---------------------------------------------------------------------------
# Instance (placement candidate)
# ---------------------------------------------------------------------------
@dataclass
class VNFInstance:
    """
    A VNF instance candidate on one satellite.

    Fields
    ------
    func_type : FuncID
    instance_id : InstID   — index within (func_type, satellite)
    satellite : SatID
    activation_cpu : float — b^cpu_{f,i,s}: fixed CPU overhead when active
    per_user_cpu : float   — a^cpu_{f,i,s}(n,u): per-user CPU load
    proc_delay_ms : float  — VNF processing delay in milliseconds (default 0)
    """
    func_type: FuncID
    instance_id: InstID
    satellite: SatID
    activation_cpu: float
    per_user_cpu: float
    proc_delay_ms: float = 0.0


# ---------------------------------------------------------------------------
# Placement state (used by SA and as MILP warm-start)
# ---------------------------------------------------------------------------
@dataclass
class PlacementState:
    """
    A complete feasible placement for one epoch.

    The primary representation is a dict mapping each (slice, user, f_pos, func)
    4-tuple to the chosen (instance, satellite) pair.  All constraint
    verification operates on this dict, enabling O(1) lookup and update.

    Fields
    ------
    assignment : Dict[PlacementKey, PlacementVal]
        Core placement mapping.
    cpu_load : Dict[SatID, float]
        Running total CPU load per satellite (maintained incrementally).
    instance_active : Dict[Tuple[FuncID,InstID,SatID], bool]
        Whether each (f,i,s) instance is active (has >= 1 user assigned).
    obj_value : float
        Cached weighted objective value (updated incrementally by SA).
    epoch : int
    """
    assignment: Dict[PlacementKey, PlacementVal] = field(default_factory=dict)
    cpu_load: Dict[SatID, float] = field(default_factory=dict)
    instance_active: Dict[Tuple[FuncID, InstID, SatID], bool] = field(default_factory=dict)
    obj_value: float = float("inf")
    epoch: int = 0

    def copy(self) -> "PlacementState":
        """Deep copy — used to snapshot the best SA solution."""
        return PlacementState(
            assignment=dict(self.assignment),
            cpu_load=dict(self.cpu_load),
            instance_active=dict(self.instance_active),
            obj_value=self.obj_value,
            epoch=self.epoch,
        )

    def get_satellite(self, n: SliceID, u: UserID, f_pos: int, f: FuncID) -> Optional[SatID]:
        """Return the satellite for user u of slice n at chain position f_pos, or None."""
        val = self.assignment.get((n, u, f_pos, f))
        return val[1] if val is not None else None


# ---------------------------------------------------------------------------
# Preprocessing output
# ---------------------------------------------------------------------------
@dataclass
class PreprocessResult:
    """
    Output of the preprocessing step (run once per epoch before SA/MILP).

    Fields
    ------
    pi : Dict[PlacementKey, bool]
        pi_{n,u,f} = True iff the previous assignment of (n,u,f) remains
        feasible at the current epoch (satellite still visible and has
        sufficient CPU capacity).
    cap_use_bar : float
        Upper bound on CapUse (from LP relaxation), used for normalization.
    risk_bar : float
        Upper bound on Risk (from LP relaxation).
    mig_bar : float
        Upper bound on Mig (from LP relaxation).
    """
    pi: Dict[PlacementKey, bool]
    cap_use_bar: float
    risk_bar: float
    mig_bar: float


# ---------------------------------------------------------------------------
# Solver result
# ---------------------------------------------------------------------------
@dataclass
class SolverResult:
    """
    Output of one epoch solve (SA or MILP).

    Fields
    ------
    placement : PlacementState
    risk_ex : float          — Risk^ex computed via exact formula (eq:risk_exact)
    risk_lb : float          — Risk^LB (coarse lower bound)
    risk_ub : float          — Risk^UB (coarse upper bound)
    cap_use : float          — Raw CPU utilization (absolute units)
    cap_use_pct : float      — Average CPU utilization across all satellites (%)
    active_sat_util_pct : float — CPU utilization restricted to satellites with active VNFs (%)
    max_isl_load : int   — Peak number of (user, hop) flows through any single ISL link
    avg_isl_load : float — Mean flows per active ISL link
    n_active_isl_links : int — ISL links carrying at least one flow
    mig_cost : float         — Raw migration cost
    obj_value : float        — Weighted normalized objective
    solve_time_s : float     — Wall-clock MILP solve time (SA time reported separately)
    sa_time_s : float        — SA warm-start time
    mip_gap_pct : float      — Final MIPGap reported by solver (0 for heuristics)
    status : str             — "optimal", "timelimit", "infeasible", "heuristic"
    n_migrations : int       — Total migrations (avoidable + forced)
    n_avoidable_migrations : int
    method : str             — "B1", "B2", "B3", "proposed_coarse", "proposed_exact"
    instance_id : int
    epoch : int
    """
    placement: PlacementState
    risk_ex: float = 0.0
    risk_lb: float = 0.0
    risk_ub: float = 0.0
    cap_use: float = 0.0
    cap_use_pct: float = 0.0
    active_sat_util_pct: float = 0.0
    peak_sat_util_pct: float = 0.0
    max_isl_load: int = 0
    avg_isl_load: float = 0.0
    n_active_isl_links: int = 0
    mig_cost: float = 0.0
    delay_compliance_pct: float = 100.0
    risk_bound_tightness: float = 1.0
    obj_value: float = float("inf")
    solve_time_s: float = 0.0
    sa_time_s: float = 0.0
    mip_gap_pct: float = 0.0
    status: str = "unknown"
    n_migrations: int = 0
    n_avoidable_migrations: int = 0
    method: str = "unknown"
    instance_id: int = 0
    epoch: int = 0
