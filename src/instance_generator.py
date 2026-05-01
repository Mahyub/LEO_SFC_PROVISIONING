"""
src/instance_generator.py
=========================
Generates reproducible simulation instances: topology snapshots, SFCs,
risk parameters, and VNF instance candidates.

Corresponds to Phase 4 of the Implementation Guide (Scenario Generation)
and the instance generation protocol (Steps 1-7).

Assumptions
-----------
- LEO topology is approximated as a Walker-like mesh: satellites are arranged
  in `orbital_planes` planes, each with `num_satellites // orbital_planes`
  satellites.  ISL neighbors are the 2 intra-plane neighbors and 2
  cross-plane neighbors (4 neighbors total at most).
- ISL delay is drawn uniformly from [isl_delay_range_ms] (simulating the
  variation caused by different inter-satellite distances during a snapshot).
- Two VNF instances per (function_type, satellite) pair (I_f = 2).
- CPU parameters: activation overhead b^cpu ~ Uniform(0.5, 2.0),
  per-user load a^cpu ~ Uniform(0.1, 0.5).  These are abstract units.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import (
    FuncID, InstID, PlacementKey, RiskParameters,
    SatID, SFC, SliceID, TopologySnapshot, UserID, VNFInstance,
)
from .visibility import walker_star_positions


# Number of VNF instances per (function_type, satellite)
INSTANCES_PER_SAT = 10


class InstanceGenerator:
    """
    Generates one simulation instance given a scenario config and a seed.

    Usage
    -----
    >>> gen = InstanceGenerator(config, seed=42)
    >>> topo = gen.topology_snapshot(epoch=0)
    >>> sfcs = gen.service_function_chains()
    >>> risk = gen.risk_parameters()
    >>> vnfs = gen.vnf_instances(topo)
    """

    def __init__(self, config: dict, seed: int):
        self.config = config
        self.seed   = seed
        self.rng    = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Derived constants
        self.S  = config["num_satellites"]
        self.ns = config["num_slices"]
        self.planes = config["orbital_planes"]
        self.sats_per_plane = max(1, self.S // self.planes)
        self.F  = config["function_types"]

    # ------------------------------------------------------------------
    # Public generators
    # ------------------------------------------------------------------

    def topology_snapshot(self, epoch: int = 0) -> TopologySnapshot:
        """
        Generate an ISL topology snapshot for the given epoch.

        The random seed is perturbed by epoch so successive epochs produce
        slightly different delay values (simulating orbital dynamics) while
        the neighbor topology stays fixed within a simulation run.
        """
        epoch_rng = np.random.default_rng(self.seed + epoch * 1000)

        # Build ISL neighbor graph (Walker-like mesh)
        neighbors: Dict[SatID, List[SatID]] = {}
        for s in range(self.S):
            nbrs = [s]  # self-loop for intra-satellite transitions
            plane   = s // self.sats_per_plane
            pos     = s %  self.sats_per_plane

            # Intra-plane neighbors (wrap-around)
            prev_pos = (pos - 1) % self.sats_per_plane
            next_pos = (pos + 1) % self.sats_per_plane
            nbrs.append(plane * self.sats_per_plane + prev_pos)
            nbrs.append(plane * self.sats_per_plane + next_pos)

            # Cross-plane neighbors (only if enough planes)
            if self.planes > 1:
                prev_plane = (plane - 1) % self.planes
                next_plane = (plane + 1) % self.planes
                nbrs.append(prev_plane * self.sats_per_plane + pos)
                nbrs.append(next_plane * self.sats_per_plane + pos)

            # Clip to valid satellite indices
            neighbors[s] = list({max(0, min(self.S - 1, n)) for n in nbrs})

        # Assign ISL delays
        d_lo, d_hi = self.config["isl_delay_range_ms"]
        delays: Dict[Tuple[SatID, SatID], float] = {}
        for s, nbrs in neighbors.items():
            for t in nbrs:
                if s == t:
                    delays[(s, t)] = 0.0
                elif (t, s) not in delays:
                    # Symmetrize with slight per-epoch variation
                    base = float(epoch_rng.uniform(d_lo, d_hi))
                    delays[(s, t)] = base
                    delays[(t, s)] = base

        # Assign CPU capacities (fixed per instance, vary per satellite)
        cap_lo, cap_hi = self.config["cpu_capacity_range"]
        cap: Dict[SatID, float] = {
            s: float(self.np_rng.uniform(cap_lo, cap_hi)) for s in range(self.S)
        }

        # Compute Walker-Star satellite positions for visibility checks.
        # epoch_s = epoch index × epoch duration (seconds).
        epoch_duration_s = float(self.config.get("epoch_duration_s", 60.0))
        raw_positions = walker_star_positions(
            num_sats       = self.S,
            num_planes     = self.planes,
            altitude_km    = float(self.config.get("orbit_altitude_km", 550.0)),
            inclination_deg= float(self.config.get("orbit_inclination_deg", 53.0)),
            phasing        = int(self.config.get("walker_phasing", 1)),
            epoch_s        = epoch * epoch_duration_s,
        )
        sat_positions: Dict[SatID, tuple] = {
            s: raw_positions[s] for s in range(self.S)
        }

        return TopologySnapshot(
            epoch=epoch,
            num_satellites=self.S,
            isl_neighbors=neighbors,
            isl_delay_ms=delays,
            cpu_capacity=cap,
            sat_positions=sat_positions,
        )

    def service_function_chains(self) -> List[SFC]:
        """
        Generate SFCs for all slices.

        Each slice gets a random SFC length from sfc_length_range, a random
        ordered selection of function types, a set of users, random E2E
        budgets, and a random criticality value C[n] ~ Uniform(criticality_C_range).
        """
        sfcs: List[SFC] = []
        l_lo, l_hi = self.config["sfc_length_range"]
        b_lo, b_hi = self.config["e2e_budget_range_ms"]
        c_lo, c_hi = self.config["criticality_C_range"]
        U = self.config["users_per_slice"]
        ground_stations = self.config.get("slice_ground_stations", [])

        user_spread_deg = float(self.config.get("user_spread_deg", 0.0))

        for n in range(self.ns):
            L = self.rng.randint(l_lo, l_hi)
            chain = self.rng.choices(self.F, k=L)
            users = list(range(U))
            budgets: Dict[UserID, float] = {
                u: self.rng.uniform(b_lo, b_hi) for u in users
            }
            crit = self.rng.uniform(c_lo, c_hi)

            # Slice-centre ground location: use configured list, cycling if needed.
            if ground_stations:
                gs = ground_stations[n % len(ground_stations)]
                g_lat = float(gs.get("lat", 0.0))
                g_lon = float(gs.get("lon", 0.0))
            else:
                g_lat, g_lon = 0.0, 0.0

            # Per-user terminal locations: distribute users around the slice
            # centre by up to ±user_spread_deg.  When spread is 0 all users
            # share the slice centre (backward-compatible default).
            user_locs: Dict[UserID, Tuple[float, float]] = {}
            for u in users:
                if user_spread_deg > 0.0:
                    d_lat = self.rng.uniform(-user_spread_deg, user_spread_deg)
                    d_lon = self.rng.uniform(-user_spread_deg, user_spread_deg)
                    user_locs[u] = (g_lat + d_lat, g_lon + d_lon)
                else:
                    user_locs[u] = (g_lat, g_lon)

            sfcs.append(SFC(
                slice_id=n,
                functions=chain,
                user_ids=users,
                e2e_budget_ms=budgets,
                criticality=crit,
                ground_lat=g_lat,
                ground_lon=g_lon,
                user_locations=user_locs,
            ))
        return sfcs

    def risk_parameters(self, sfcs: List[SFC]) -> RiskParameters:
        """
        Draw risk model parameters from the configured ranges.

        Phi[n,n'] is symmetric; we store only (min(n,n'), max(n,n')) keys.
        """
        sens_R: Dict[FuncID, float] = dict(self.config["sensitivity_R"])
        crit_C: Dict[SliceID, float] = {sfc.slice_id: sfc.criticality for sfc in sfcs}

        phi_lo, phi_hi = self.config["policy_phi_range"]
        phi: Dict[Tuple[SliceID, SliceID], float] = {}
        for n in range(self.ns):
            for np_ in range(n + 1, self.ns):
                phi[(n, np_)] = self.rng.uniform(phi_lo, phi_hi)

        mig_D: Dict[FuncID, float] = dict(self.config["migration_cost_D"])

        return RiskParameters(
            sensitivity_R=sens_R,
            criticality_C=crit_C,
            policy_phi=phi,
            migration_cost_D=mig_D,
        )

    def vnf_instances(self, topo: TopologySnapshot) -> Dict[Tuple[FuncID, InstID, SatID], VNFInstance]:
        """
        Generate VNF instance candidates for each (func, inst, satellite) triple.

        Returns a dict keyed by (func_type, instance_id, satellite_id).
        CPU ranges are drawn from per-function-type config keys
        activation_cpu_range and per_user_cpu_range when present,
        falling back to the global defaults [0.5, 2.0] and [0.1, 0.5].
        proc_delay_ms ~ Uniform(vnf_delay_range_ms) if configured, else 0.
        """
        act_ranges  = self.config.get("activation_cpu_range", {})
        user_ranges = self.config.get("per_user_cpu_range", {})
        vnf_d_range = self.config.get("vnf_delay_range_ms", None)

        instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance] = {}
        for f in self.F:
            b_lo, b_hi = act_ranges.get(f,  [0.5, 2.0])
            a_lo, a_hi = user_ranges.get(f, [0.1, 0.5])
            for s in range(topo.num_satellites):
                for i in range(INSTANCES_PER_SAT):
                    b_cpu = float(self.np_rng.uniform(b_lo, b_hi))
                    a_cpu = float(self.np_rng.uniform(a_lo, a_hi))
                    if vnf_d_range is not None:
                        p_delay = float(self.np_rng.uniform(vnf_d_range[0], vnf_d_range[1]))
                    else:
                        p_delay = 0.0
                    instances[(f, i, s)] = VNFInstance(
                        func_type=f,
                        instance_id=i,
                        satellite=s,
                        activation_cpu=b_cpu,
                        per_user_cpu=a_cpu,
                        proc_delay_ms=p_delay,
                    )
        return instances

    def check_feasibility(
        self,
        topo: TopologySnapshot,
        sfcs: List[SFC],
        vnf_instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance],
    ) -> bool:
        """
        Quick feasibility pre-check: verify that at least one satellite has
        enough CPU capacity to host at least one instance of each function
        type with at least one user.

        If this check fails, the instance should be regenerated with a new seed.
        """
        for f in self.F:
            feasible_sat = False
            for s in range(topo.num_satellites):
                for i in range(INSTANCES_PER_SAT):
                    vi = vnf_instances.get((f, i, s))
                    if vi is None:
                        continue
                    # Minimum CPU needed: activation + 1 user
                    min_cpu = vi.activation_cpu + vi.per_user_cpu
                    if topo.cpu_capacity[s] >= min_cpu:
                        feasible_sat = True
                        break
                if feasible_sat:
                    break
            if not feasible_sat:
                return False
        return True


def load_config(config_path: str) -> dict:
    """Load a JSON scenario configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Instance pretty-printer
# ---------------------------------------------------------------------------

def print_instance_info(
    sfcs: "List[SFC]",
    risk_params: "RiskParameters",
    vnf_instances: "Dict[Tuple[FuncID, InstID, SatID], VNFInstance]",
    topo: "TopologySnapshot",
    config: dict,
) -> None:
    """
    Print a detailed breakdown of a generated instance covering:
      - SFC chains (per-slice function chain, users, E2E budget, criticality)
      - Risk parameters (sensitivity R, policy phi, migration costs)
      - VNF instance CPU stats (activation and per-user ranges per function type)
      - Topology layout (orbital planes, ISL degree distribution, CPU capacity)
    """
    W = 68  # total line width
    SEP  = "=" * W
    SEP2 = "-" * W

    def _hdr(title: str) -> None:
        print(f"\n{'─' * W}")
        print(f"  {title}")
        print(f"{'─' * W}")

    def _stat(label: str, value: str, indent: int = 4) -> None:
        pad = " " * indent
        print(f"{pad}{label:<36}{value}")

    print(f"\n{SEP}")
    print(f"  GENERATED INSTANCE DETAILS")
    print(SEP)

    # ── 1. SFC Chains ───────────────────────────────────────────────────
    _hdr("1. SERVICE FUNCTION CHAINS")
    chain_lengths = [len(sfc.functions) for sfc in sfcs]
    budgets_all   = [b for sfc in sfcs for b in sfc.e2e_budget_ms.values()]
    crits_all     = [sfc.criticality for sfc in sfcs]

    _stat("Number of slices:",        str(len(sfcs)))
    _stat("Users per slice:",         str(len(sfcs[0].user_ids)) if sfcs else "0")
    _stat("Chain length range:",      f"{min(chain_lengths)} – {max(chain_lengths)}")
    _stat("E2E budget range (ms):",   f"{min(budgets_all):.1f} – {max(budgets_all):.1f}")
    _stat("Criticality range:",       f"{min(crits_all):.3f} – {max(crits_all):.3f}")

    print()
    print(f"    {'Slice':>5}  {'Chain':<32}  {'Users':>5}  "
          f"{'Budget(ms)':>14}  {'Crit':>6}")
    print(f"    {'-'*5}  {'-'*32}  {'-'*5}  {'-'*14}  {'-'*6}")
    for sfc in sfcs:
        chain_str = " → ".join(sfc.functions)
        budgets   = list(sfc.e2e_budget_ms.values())
        b_lo, b_hi = min(budgets), max(budgets)
        budget_str = f"{b_lo:.1f} – {b_hi:.1f}" if b_lo != b_hi else f"{b_lo:.1f}"
        print(f"    {sfc.slice_id:>5}  {chain_str:<32}  {len(sfc.user_ids):>5}  "
              f"  {budget_str:>13}  {sfc.criticality:>6.3f}")

    # ── 2. Risk Parameters ──────────────────────────────────────────────
    _hdr("2. RISK PARAMETERS")

    # Sensitivity R
    print("    Function sensitivity R[f]:")
    for f_type, r_val in sorted(risk_params.sensitivity_R.items()):
        print(f"        {f_type:<8}  R = {r_val:.4f}")

    # Policy phi
    phi_vals = list(risk_params.policy_phi.values())
    print()
    _stat("Isolation policy φ[n,n']:",
          f"min={min(phi_vals):.4f}  mean={sum(phi_vals)/len(phi_vals):.4f}  max={max(phi_vals):.4f}")

    # Risk weights w = R * phi * C * C'
    weights = []
    slice_ids = [sfc.slice_id for sfc in sfcs]
    f_types   = list(risk_params.sensitivity_R.keys())
    for idx_a, n in enumerate(slice_ids):
        for n_prime in slice_ids[idx_a + 1:]:
            for f in f_types:
                w = risk_params.risk_weight(n, n_prime, f)
                if w > 0:
                    weights.append(w)
    if weights:
        _stat("Risk weight w[n,n',f] range:",
              f"min={min(weights):.4f}  max={max(weights):.4f}")

    # Migration costs
    print()
    print("    Migration disruption cost D[f]:")
    for f_type, d_val in sorted(risk_params.migration_cost_D.items()):
        print(f"        {f_type:<8}  D = {d_val:.4f}")

    # ── 3. VNF Instance CPU Stats ────────────────────────────────────────
    _hdr("3. VNF INSTANCE CPU STATISTICS")
    print(f"    {'Function':<10}  {'Act. CPU (b)':>20}  {'Per-user CPU (a)':>20}")
    print(f"    {'-'*10}  {'-'*20}  {'-'*20}")

    F_types = config.get("function_types", list(risk_params.sensitivity_R.keys()))
    S = topo.num_satellites

    for f_type in F_types:
        act_vals  = [vnf_instances[(f_type, i, s)].activation_cpu
                     for i in range(INSTANCES_PER_SAT) for s in range(S)
                     if (f_type, i, s) in vnf_instances]
        user_vals = [vnf_instances[(f_type, i, s)].per_user_cpu
                     for i in range(INSTANCES_PER_SAT) for s in range(S)
                     if (f_type, i, s) in vnf_instances]
        if not act_vals:
            continue
        act_str  = f"[{min(act_vals):.3f}, {max(act_vals):.3f}]"
        user_str = f"[{min(user_vals):.3f}, {max(user_vals):.3f}]"
        print(f"    {f_type:<10}  {act_str:>20}  {user_str:>20}")

    total_vnfs = len(vnf_instances)
    _stat("\n    Total VNF instance candidates:", str(total_vnfs))

    # ── 4. Topology ──────────────────────────────────────────────────────
    _hdr("4. TOPOLOGY (EPOCH 0 SNAPSHOT)")

    planes         = config.get("orbital_planes", 1)
    sats_per_plane = max(1, S // planes)

    # ISL degree distribution (excluding self-loops)
    degrees = {
        s: len([t for t in topo.isl_neighbors.get(s, []) if t != s])
        for s in range(S)
    }
    deg_vals  = list(degrees.values())
    deg_counts: Dict[int, int] = {}
    for d in deg_vals:
        deg_counts[d] = deg_counts.get(d, 0) + 1

    # ISL delay stats
    delay_vals = [v for (s, t), v in topo.isl_delay_ms.items() if s != t]

    # CPU capacity stats
    cap_vals = list(topo.cpu_capacity.values())

    _stat("Satellites:",              str(S))
    _stat("Orbital planes:",          str(planes))
    _stat("Satellites per plane:",    str(sats_per_plane))

    # ISL edges (undirected)
    isl_edges = sum(
        1 for (s, t) in topo.isl_delay_ms if s < t and s != t
    )
    _stat("ISL links (undirected):",  str(isl_edges))

    print()
    print("    ISL degree distribution (excl. self-loops):")
    for deg in sorted(deg_counts):
        bar = "█" * deg_counts[deg]
        print(f"        degree {deg}: {deg_counts[deg]:>4} satellites  {bar}")

    if delay_vals:
        print()
        _stat("ISL delay range (ms):",
              f"{min(delay_vals):.2f} – {max(delay_vals):.2f}")
        _stat("ISL delay mean (ms):",
              f"{sum(delay_vals)/len(delay_vals):.2f}")

    print()
    _stat("CPU capacity range:",
          f"{min(cap_vals):.2f} – {max(cap_vals):.2f}")
    _stat("CPU capacity mean:",
          f"{sum(cap_vals)/len(cap_vals):.2f}")
    _stat("Total system CPU:",
          f"{sum(cap_vals):.2f}")

    print(f"\n{SEP}\n")