"""
src/instance_generator_matlab.py
=================================
Drop-in replacement for InstanceGenerator that delegates topology
generation to MATLAB via MatlabTopologyBridge, while keeping all
other generation logic (SFCs, risk params, VNF instances) identical.

Usage
-----
Replace the import in experiment.py / main.py:

    # Before:
    from src.instance_generator import InstanceGenerator, load_config

    # After:
    from src.instance_generator_matlab import InstanceGeneratorMatlab as InstanceGenerator, load_config

No other changes needed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import (
    FuncID, InstID, RiskParameters,
    SatID, SFC, SliceID, TopologySnapshot, UserID, VNFInstance,
)
# Re-export everything from original module that callers may need
from .instance_generator import (
    load_config, print_instance_info,
    INSTANCES_PER_SAT,
)

# Import the bridge (lives at the project root)
from matlab_topology_bridge import MatlabTopologyBridge


class InstanceGeneratorMatlab:
    """
    Generates simulation instances using MATLAB SatCom toolkit for topology
    and pure-Python logic for everything else (SFCs, risk params, VNF instances).

    The bridge is created lazily on first call to topology_snapshot() and
    shared across all epochs within the same instance.

    Parameters
    ----------
    config : dict
        Scenario config (same format as InstanceGenerator).
    seed : int
        Random seed (used for SFC / risk / VNF generation; topology comes
        from MATLAB using the same constellation params each time).
    config_path : str, optional
        Path to the config file (needed by MatlabTopologyBridge).
        If not provided, the bridge uses DEFAULT_MATLAB_CFG defaults.
    """

    def __init__(self, config: dict, seed: int, config_path: str = "config/base.json"):
        self.config      = config
        self.seed        = seed
        self.rng         = random.Random(seed)
        self.np_rng      = np.random.default_rng(seed)
        self._config_path = config_path

        # Derived constants (same as original)
        self.S  = config["num_satellites"]
        self.ns = config["num_slices"]
        self.F  = config["function_types"]

        # Bridge is created lazily
        self._bridge: Optional[MatlabTopologyBridge] = None

    # ── Bridge lifecycle ──────────────────────────────────────────────────

    def _get_bridge(self) -> MatlabTopologyBridge:
        if self._bridge is None:
            self._bridge = MatlabTopologyBridge(self._config_path)
        return self._bridge

    def precompute_epochs(self, num_epochs: int, start_epoch: int = 0) -> None:
        """
        Generate all epoch topologies in a single MATLAB call and cache them.

        Call this once before the epoch loop so that topology_snapshot()
        is served from memory for every subsequent epoch — no repeated
        MATLAB launches.

        Parameters
        ----------
        num_epochs : int
            Number of consecutive epochs to pre-generate.
        start_epoch : int
            Index of the first epoch (default 0).
        """
        self._get_bridge().precompute_all_epochs(num_epochs, start_epoch)

    def close(self) -> None:
        """Shut down the MATLAB engine (call when done with this instance)."""
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None

    def __del__(self):
        self.close()

    # ── Public generators (same interface as InstanceGenerator) ───────────

    def topology_snapshot(self, epoch: int = 0) -> TopologySnapshot:
        """
        Delegate topology generation to MATLAB SatCom toolkit.

        The returned TopologySnapshot is structurally identical to the one
        produced by the original Python InstanceGenerator, so all downstream
        modules (SA, MILP, baselines, metrics) work without modification.
        """
        bridge = self._get_bridge()
        topo = bridge.get_topology(epoch)

        # Verify satellite count matches config (MATLAB is authoritative)
        if topo.num_satellites != self.S:
            raise ValueError(
                f"MATLAB returned {topo.num_satellites} satellites "
                f"but config expects {self.S}."
            )
        return topo

    def service_function_chains(self) -> List[SFC]:
        """Identical to original InstanceGenerator — pure Python."""
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

            if ground_stations:
                gs = ground_stations[n % len(ground_stations)]
                g_lat = float(gs.get("lat", 0.0))
                g_lon = float(gs.get("lon", 0.0))
            else:
                g_lat, g_lon = 0.0, 0.0

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
        """Identical to original InstanceGenerator — pure Python."""
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
        """Identical to original InstanceGenerator — pure Python."""
        act_ranges  = self.config.get("activation_cpu_range", {})
        user_ranges = self.config.get("per_user_cpu_range", {})
        d_lo, d_hi  = self.config.get("vnf_delay_range_ms", [1.0, 3.0])

        instances: Dict[Tuple[FuncID, InstID, SatID], VNFInstance] = {}
        for f in self.F:
            b_lo, b_hi = act_ranges.get(f,  [0.5, 2.0])
            a_lo, a_hi = user_ranges.get(f, [0.1, 0.5])
            for s in range(topo.num_satellites):
                for i in range(INSTANCES_PER_SAT):
                    b_cpu = float(self.np_rng.uniform(b_lo, b_hi))
                    a_cpu = float(self.np_rng.uniform(a_lo, a_hi))
                    d_ms  = float(self.np_rng.uniform(d_lo, d_hi))
                    instances[(f, i, s)] = VNFInstance(
                        func_type=f,
                        instance_id=i,
                        satellite=s,
                        activation_cpu=b_cpu,
                        per_user_cpu=a_cpu,
                        proc_delay_ms=d_ms,
                    )
        return instances
