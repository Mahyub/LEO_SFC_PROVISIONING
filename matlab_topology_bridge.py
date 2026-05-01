"""
matlab_topology_bridge.py
=========================
Bridge module: calls MATLAB SatCom toolkit to generate LEO satellite
network topology and returns TopologySnapshot objects compatible with the
existing LEO-SFC pipeline.

Batch epoch precomputation
--------------------------
The key design principle is that MATLAB is launched **once** per simulation
run and generates ALL epoch snapshots in that single call.  This avoids the
30-60 s cold-start overhead on every epoch and ensures temporal consistency
(satellites are propagated forward continuously from the same orbital epoch
rather than being re-initialised from t=0 for each snapshot).

Typical usage::

    bridge = MatlabTopologyBridge("config/base.json")
    bridge.precompute_all_epochs(num_epochs=20)   # one MATLAB call
    for epoch in range(20):
        topo = bridge.get_topology(epoch)          # served from cache

Three interface strategies are supported, tried in priority order:
  1. matlab.engine (MATLAB Engine API for Python) — fastest, in-process
  2. Subprocess / .mat file exchange          — works without Engine license
  3. MATLAB Production Server REST API        — for cloud / server deployments

Set bridge_mode = "engine" | "subprocess" | "rest" to force one.

Dependencies
------------
  Strategy 1: pip install matlabengine  (ships with MATLAB >= R2014b)
  Strategy 2: pip install scipy numpy
  Strategy 3: pip install requests
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Optional third-party imports (guarded) ──────────────────────────────────
try:
    import matlab.engine as _matlab_engine  # Strategy 1
    _HAS_ENGINE = True
except ImportError:
    _HAS_ENGINE = False

try:
    import scipy.io as _sio               # read .mat files (Strategy 2)
    import numpy as _np
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import requests as _requests          # Strategy 3
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from src.types import SatID, TopologySnapshot


# ────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_MATLAB_CFG: Dict[str, Any] = {
    "matlab_script_dir":         "matlab",
    "matlab_exe":                "matlab",
    "mps_url":                   "http://localhost:9910/satcom/generateTopology",
    "bridge_mode":               "auto",
    "walker_total_sats":         100,
    "walker_planes":             10,
    "walker_phasing":            1,
    "orbit_altitude_km":         550.0,
    "orbit_inclination_deg":     53.0,
    "epoch_start_utc":           "2025-01-01 00:00:00",
    "epoch_duration_s":          60.0,
    "isl_model":                 "nearest_k",
    "isl_k":                     4,
    "isl_max_range_km":          2500.0,
    "cpu_capacity_range":        [40, 120],
    "matlab_subprocess_timeout_s": 600,
}


def _load_bridge_config(config_path: str) -> dict:
    """Merge user config with defaults (highest priority: matlab_bridge block)."""
    with open(config_path) as f:
        user_cfg = json.load(f)

    cfg = dict(DEFAULT_MATLAB_CFG)
    cfg.update(user_cfg.get("matlab_bridge", {}))

    for k in ("cpu_capacity_range", "random_seed_base", "isl_delay_range_ms"):
        if k in user_cfg:
            cfg[k] = user_cfg[k]

    if "num_satellites" in user_cfg:
        cfg["num_satellites"]    = user_cfg["num_satellites"]
        cfg["walker_total_sats"] = user_cfg["num_satellites"]
    if "orbital_planes" in user_cfg:
        cfg["orbital_planes"] = user_cfg["orbital_planes"]
        cfg["walker_planes"]  = user_cfg["orbital_planes"]

    return cfg


# ────────────────────────────────────────────────────────────────────────────
# Main bridge class
# ────────────────────────────────────────────────────────────────────────────

class MatlabTopologyBridge:
    """
    Calls MATLAB SatCom toolkit and converts results into TopologySnapshot
    objects for the LEO-SFC pipeline.

    The recommended pattern is to call precompute_all_epochs() once at the
    start of a simulation run so that MATLAB is launched only once for the
    entire epoch series.  get_topology() then serves subsequent requests
    directly from the in-memory cache without any further MATLAB invocations.

    Parameters
    ----------
    config_path : str
        Path to the scenario JSON config file.
    mode : str, optional
        Override bridge mode: "engine" | "subprocess" | "rest".
        Default "auto" tries Engine -> subprocess -> REST.
    """

    def __init__(self, config_path: str, mode: Optional[str] = None):
        self.cfg         = _load_bridge_config(config_path)
        self._mode       = mode or self.cfg.get("bridge_mode", "auto")
        self._engine     = None
        self._script_dir = Path(self.cfg["matlab_script_dir"])

        # Cache: epoch index -> TopologySnapshot (populated by precompute_all_epochs)
        self._cache: Dict[int, TopologySnapshot] = {}

        if self._mode == "auto":
            self._mode = "engine" if _HAS_ENGINE else "subprocess"

        if self._mode == "engine":
            self._start_engine()

        # Auto-load cache from disk if cache_file is configured and exists
        cache_file = self.cfg.get("cache_file", "")
        if cache_file and Path(cache_file).exists():
            self.load_cache(cache_file)

    # ── Strategy 1: MATLAB Engine API ────────────────────────────────────

    def _start_engine(self) -> None:
        if not _HAS_ENGINE:
            raise RuntimeError(
                "matlab.engine not available. "
                "Install with: pip install matlabengine "
                "(requires matching MATLAB version)"
            )
        print("[MatlabBridge] Starting MATLAB engine …")
        t0 = time.perf_counter()
        self._engine = _matlab_engine.start_matlab()
        self._engine.addpath(str(self._script_dir.resolve()), nargout=0)
        print(f"[MatlabBridge] Engine ready in {time.perf_counter()-t0:.1f}s")

    def _call_engine(self, epoch: int) -> dict:
        """Single-epoch engine call (fallback if precompute not used)."""
        params = self._build_matlab_params(epoch)
        adj_mat, delay_mat, pos_mat = self._engine.generate_topology(
            params, float(params["epochOffsetS"]), nargout=3
        )
        return {
            "adj":       _matlab_to_numpy(adj_mat),
            "delays":    _matlab_to_numpy(delay_mat),
            "positions": _matlab_to_numpy(pos_mat),
        }

    def _call_engine_batch(self, num_epochs: int, start_epoch: int) -> dict:
        """Batch engine call: returns 3-D numpy arrays [S×S×E] / [S×3×E]."""
        params = self._build_matlab_params_batch(num_epochs, start_epoch)
        adj_mat, delay_mat, pos_mat = self._engine.generate_topology(
            params, float(params["epochOffsetS"]), nargout=3
        )
        adj      = _matlab_to_numpy(adj_mat)
        delays   = _matlab_to_numpy(delay_mat)
        pos      = _matlab_to_numpy(pos_mat)
        # Engine returns 2-D when numEpochs==1; normalise to 3-D
        if adj.ndim == 2:
            adj    = adj[:, :, _np.newaxis]
            delays = delays[:, :, _np.newaxis]
            pos    = pos[:, _np.newaxis] if pos.ndim == 1 else pos[:, :, _np.newaxis]
        return {"adj": adj, "delays": delays, "positions": pos}

    # ── Strategy 2: Subprocess + .mat file exchange ───────────────────────

    def _call_subprocess(self, epoch: int) -> dict:
        """Single-epoch subprocess call (fallback if precompute not used)."""
        return self._call_subprocess_batch(num_epochs=1, start_epoch=epoch)

    def _call_subprocess_batch(self, num_epochs: int, start_epoch: int) -> dict:
        """
        Batch subprocess call: launches MATLAB once for all epochs.

        Writes params to a JSON file, runs MATLAB as a subprocess, reads back
        the .mat result file which contains 3-D arrays when num_epochs > 1.
        """
        if not _HAS_SCIPY:
            raise RuntimeError("scipy required for subprocess mode: pip install scipy")

        tmpdir = tempfile.mkdtemp(prefix="matlab_topo_")
        try:
            params_file = os.path.join(tmpdir, "params.json")
            result_file = os.path.join(tmpdir, "topology.mat")

            params = self._build_matlab_params_batch(num_epochs, start_epoch)
            params["result_file"] = result_file
            with open(params_file, "w") as f:
                json.dump(params, f)

            script_dir_fwd  = str(self._script_dir.resolve()).replace("\\", "/")
            params_file_fwd = params_file.replace("\\", "/")

            matlab_cmd = (
                f"addpath('{script_dir_fwd}'); "
                f"generate_topology_cli('{params_file_fwd}');"
            )
            cmd = [
                self.cfg.get("matlab_exe", "matlab"),
                "-batch", matlab_cmd,
                "-nosplash", "-nodesktop",
            ]

            timeout_s = int(self.cfg.get("matlab_subprocess_timeout_s", 600))
            print(f"[MatlabBridge] Launching MATLAB subprocess "
                  f"(epochs {start_epoch}–{start_epoch+num_epochs-1}, "
                  f"timeout={timeout_s}s) …")
            print(f"[MatlabBridge] CMD: {' '.join(cmd)}")
            t0 = time.perf_counter()
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout_s
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"MATLAB subprocess timed out after {timeout_s}s.\n"
                    f"Increase 'matlab_subprocess_timeout_s' in your config, "
                    f"or switch to bridge_mode='engine' for much faster calls.\n"
                    f"CMD: {' '.join(cmd)}"
                )
            elapsed = time.perf_counter() - t0

            if result.returncode != 0:
                raise RuntimeError(
                    f"MATLAB subprocess failed (exit {result.returncode}):\n"
                    f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                )
            print(f"[MatlabBridge] MATLAB finished in {elapsed:.1f}s "
                  f"({num_epochs} epoch(s))")

            if not os.path.exists(result_file):
                raise FileNotFoundError(
                    f"MATLAB did not produce {result_file}.\n"
                    f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                )
            mat = _sio.loadmat(result_file)
            adj    = mat["adj_matrix"].astype(float)
            delays = mat["delay_matrix"].astype(float)
            pos    = mat["positions"].astype(float)

            # MATLAB squeezes to 2-D when numEpochs==1 — normalise to 3-D
            if adj.ndim == 2:
                adj    = adj[:, :, _np.newaxis]
                delays = delays[:, :, _np.newaxis]
                pos    = pos[:, :, _np.newaxis] if pos.ndim == 2 else pos

            return {"adj": adj, "delays": delays, "positions": pos}
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Strategy 3: MATLAB Production Server REST API ────────────────────

    def _call_rest(self, epoch: int) -> dict:
        """Single-epoch REST call (fallback if precompute not used)."""
        return self._call_rest_batch(num_epochs=1, start_epoch=epoch)

    def _call_rest_batch(self, num_epochs: int, start_epoch: int) -> dict:
        """Batch REST call: sends numEpochs in params, expects 3-D arrays."""
        if not _HAS_REQUESTS:
            raise RuntimeError("requests required for REST mode: pip install requests")

        url    = self.cfg["mps_url"]
        params = self._build_matlab_params_batch(num_epochs, start_epoch)
        payload = {
            "nargout": 3,
            "rhs":     [params, float(params["epochOffsetS"])],
        }

        print(f"[MatlabBridge] Calling MPS at {url} "
              f"(epochs {start_epoch}–{start_epoch+num_epochs-1}) …")
        t0 = time.perf_counter()
        resp = _requests.post(url, json=payload, timeout=max(60, num_epochs * 10))
        resp.raise_for_status()
        print(f"[MatlabBridge] MPS responded in {time.perf_counter()-t0:.1f}s")

        import numpy as np
        lhs = resp.json()["lhs"]
        adj    = np.array(lhs[0]["mwdata"]).reshape(lhs[0]["mwsize"], order="F")
        delays = np.array(lhs[1]["mwdata"]).reshape(lhs[1]["mwsize"], order="F")
        pos    = np.array(lhs[2]["mwdata"]).reshape(lhs[2]["mwsize"], order="F")

        if adj.ndim == 2:
            adj    = adj[:, :, np.newaxis]
            delays = delays[:, :, np.newaxis]
            pos    = pos[:, :, np.newaxis] if pos.ndim == 2 else pos

        return {"adj": adj, "delays": delays, "positions": pos}

    # ── Shared helpers ────────────────────────────────────────────────────

    def _build_matlab_params(self, epoch: int) -> dict:
        """Build params dict for a single epoch."""
        cfg = self.cfg
        epoch_start_s = epoch * float(cfg.get("epoch_duration_s", 60.0))

        total_sats = int(cfg["walker_total_sats"])
        num_planes = int(cfg["walker_planes"])
        if total_sats % num_planes != 0:
            raise ValueError(
                f"num_satellites ({total_sats}) must be divisible by "
                f"orbital_planes ({num_planes})."
            )

        return {
            "totalSats":      total_sats,
            "numPlanes":      num_planes,
            "phasing":        int(cfg.get("walker_phasing", 1)),
            "altitudeKm":     float(cfg.get("orbit_altitude_km", 550.0)),
            "inclinationDeg": float(cfg.get("orbit_inclination_deg", 53.0)),
            "islModel":       cfg.get("isl_model", "nearest_k"),
            "islK":           int(cfg.get("isl_k", 4)),
            "islMaxRangeKm":  float(cfg.get("isl_max_range_km", 2500.0)),
            "epochStartUTC":  cfg.get("epoch_start_utc", "2025-01-01 00:00:00"),
            "epochOffsetS":   epoch_start_s,
            "epochDurationS": float(cfg.get("epoch_duration_s", 60.0)),
        }

    def _build_matlab_params_batch(self, num_epochs: int, start_epoch: int) -> dict:
        """Build params dict for a batch of epochs starting at start_epoch."""
        params = self._build_matlab_params(start_epoch)
        params["numEpochs"] = num_epochs
        return params

    def _raw_to_topology(self, raw_2d: dict, epoch: int) -> TopologySnapshot:
        """
        Convert 2-D MATLAB arrays (single epoch) into a TopologySnapshot.

        raw_2d keys: "adj" [S×S], "delays" [S×S], "positions" [S×3]
        """
        import numpy as np
        import random

        adj    = raw_2d["adj"]
        delays = raw_2d["delays"]
        S      = adj.shape[0]

        isl_neighbors: Dict[SatID, List[SatID]] = {}
        for s in range(S):
            nbrs = [s]
            for t in range(S):
                if s != t and adj[s, t] > 0.5:
                    nbrs.append(t)
            isl_neighbors[s] = nbrs

        isl_delay_ms: Dict[Tuple[SatID, SatID], float] = {}
        for s in range(S):
            isl_delay_ms[(s, s)] = 0.0
            for t in range(S):
                if s != t and adj[s, t] > 0.5:
                    isl_delay_ms[(s, t)] = float(delays[s, t])

        rng = random.Random(self.cfg.get("random_seed_base", 42) + epoch)
        cap_lo, cap_hi = self.cfg.get("cpu_capacity_range", [40, 120])
        cpu_capacity: Dict[SatID, float] = {
            s: rng.uniform(cap_lo, cap_hi) for s in range(S)
        }

        # Store satellite positions for visibility checks.
        # MATLAB returns positions as [S×3]: (lat_deg, lon_deg, alt_km).
        sat_positions = {}
        pos_arr = raw_2d.get("positions")
        if pos_arr is not None and pos_arr.ndim == 2 and pos_arr.shape == (S, 3):
            for s in range(S):
                sat_positions[s] = (
                    float(pos_arr[s, 0]),  # lat_deg
                    float(pos_arr[s, 1]),  # lon_deg
                    float(pos_arr[s, 2]),  # alt_km
                )

        return TopologySnapshot(
            epoch=epoch,
            num_satellites=S,
            isl_neighbors=isl_neighbors,
            isl_delay_ms=isl_delay_ms,
            cpu_capacity=cpu_capacity,
            sat_positions=sat_positions,
        )

    def _raw_batch_to_topologies(
        self, raw: dict, num_epochs: int, start_epoch: int
    ) -> List[TopologySnapshot]:
        """
        Convert 3-D MATLAB arrays into a list of TopologySnapshot objects.

        raw["adj"]       shape [S, S, E]
        raw["delays"]    shape [S, S, E]
        raw["positions"] shape [S, 3, E]
        """
        adj_3d    = raw["adj"]      # [S, S, E]
        delays_3d = raw["delays"]   # [S, S, E]
        pos_3d    = raw["positions"]

        snapshots: List[TopologySnapshot] = []
        for e in range(num_epochs):
            raw_2d = {
                "adj":       adj_3d[:, :, e],
                "delays":    delays_3d[:, :, e],
                "positions": pos_3d[:, :, e] if pos_3d.ndim == 3 else pos_3d,
            }
            snap = self._raw_to_topology(raw_2d, start_epoch + e)
            snapshots.append(snap)
        return snapshots

    # ── Public API ────────────────────────────────────────────────────────

    def precompute_all_epochs(
        self, num_epochs: int, start_epoch: int = 0
    ) -> None:
        """
        Generate all epochs in a SINGLE MATLAB call and cache the results.

        This is the preferred entry point when running a multi-epoch
        simulation.  After this call, get_topology() is served entirely from
        the in-memory cache — no further MATLAB interaction occurs.

        Parameters
        ----------
        num_epochs : int
            Total number of consecutive epochs to generate.
        start_epoch : int
            Index of the first epoch (default 0).  The time offset sent to
            MATLAB is  start_epoch * epoch_duration_s.
        """
        print(f"[MatlabBridge] Precomputing {num_epochs} epoch(s) "
              f"(start={start_epoch}) in one MATLAB call …")
        t0 = time.perf_counter()

        if self._mode == "engine":
            raw = self._call_engine_batch(num_epochs, start_epoch)
        elif self._mode == "subprocess":
            raw = self._call_subprocess_batch(num_epochs, start_epoch)
        elif self._mode == "rest":
            raw = self._call_rest_batch(num_epochs, start_epoch)
        else:
            raise ValueError(f"Unknown bridge mode: {self._mode!r}")

        snapshots = self._raw_batch_to_topologies(raw, num_epochs, start_epoch)
        for snap in snapshots:
            self._cache[snap.epoch] = snap

        elapsed = time.perf_counter() - t0
        print(f"[MatlabBridge] Cached epochs {start_epoch}–"
              f"{start_epoch+num_epochs-1} in {elapsed:.1f}s total "
              f"({elapsed/num_epochs:.2f}s per epoch)")

        # Auto-save to disk if cache_file is configured
        cache_file = self.cfg.get("cache_file", "")
        if cache_file:
            self.save_cache(cache_file)

    def get_topology(self, epoch: int) -> TopologySnapshot:
        """
        Return the TopologySnapshot for the given epoch.

        Serves from cache if precompute_all_epochs() was called.  Falls back
        to a single-epoch MATLAB call otherwise (each call incurs full
        MATLAB startup cost in subprocess mode — prefer precompute_all_epochs
        for multi-epoch runs).
        """
        if epoch in self._cache:
            return self._cache[epoch]

        # Fallback: single-epoch call (warns the user)
        print(f"[MatlabBridge] WARNING: epoch {epoch} not in cache. "
              f"Calling MATLAB for a single epoch — consider calling "
              f"precompute_all_epochs() before the epoch loop.")

        if self._mode == "engine":
            raw_2d = self._call_engine(epoch)
        elif self._mode == "subprocess":
            raw_2d = self._call_subprocess(epoch)
        elif self._mode == "rest":
            raw_2d = self._call_rest(epoch)
        else:
            raise ValueError(f"Unknown bridge mode: {self._mode!r}")

        # Subprocess / REST already return 3-D; squeeze back to 2-D for
        # _raw_to_topology when called from the single-epoch fallback path.
        adj    = raw_2d["adj"]
        delays = raw_2d["delays"]
        pos    = raw_2d["positions"]
        if adj.ndim == 3:
            raw_2d = {
                "adj":       adj[:, :, 0],
                "delays":    delays[:, :, 0],
                "positions": pos[:, :, 0] if pos.ndim == 3 else pos,
            }

        return self._raw_to_topology(raw_2d, epoch)

    # ── Cache inspection and persistence ─────────────────────────────────

    def cache_info(self) -> dict:
        """
        Print and return a summary of the current in-memory cache.

        Returns
        -------
        dict with keys:
          cached_epochs  — sorted list of epoch indices in cache
          num_epochs     — total cached epoch count
          satellites     — number of satellites (from first cached epoch)
          has_positions  — True if sat_positions are populated
          cache_file     — path configured for auto-save/load ('' if none)
        """
        if not self._cache:
            print("[MatlabBridge] Cache is empty.")
            return {"cached_epochs": [], "num_epochs": 0,
                    "satellites": 0, "has_positions": False,
                    "cache_file": self.cfg.get("cache_file", "")}

        epochs = sorted(self._cache.keys())
        first  = self._cache[epochs[0]]
        info   = {
            "cached_epochs": epochs,
            "num_epochs":    len(epochs),
            "satellites":    first.num_satellites,
            "has_positions": bool(first.sat_positions),
            "cache_file":    self.cfg.get("cache_file", ""),
        }
        print(
            f"[MatlabBridge] Cache: {info['num_epochs']} epoch(s)  "
            f"indices={epochs[0]}–{epochs[-1]}  "
            f"satellites={info['satellites']}  "
            f"positions={'yes' if info['has_positions'] else 'no'}  "
            f"cache_file={info['cache_file'] or '(none)'}"
        )
        return info

    def save_cache(self, path: str) -> None:
        """
        Serialise the in-memory cache to a JSON file on disk.

        The file can be reloaded with load_cache() on the next run,
        avoiding a MATLAB re-launch entirely.  Existing file is overwritten.

        Parameters
        ----------
        path : str
            Destination file path (e.g. "data/topo_cache.json").
        """
        if not self._cache:
            print("[MatlabBridge] save_cache: nothing to save (cache is empty).")
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            str(epoch): _topo_to_dict(snap)
            for epoch, snap in self._cache.items()
        }
        with open(path, "w") as f:
            json.dump(payload, f)

        epochs = sorted(self._cache.keys())
        print(f"[MatlabBridge] Saved {len(self._cache)} epoch(s) "
              f"({epochs[0]}–{epochs[-1]}) → {path}")

    def load_cache(self, path: str) -> None:
        """
        Load a previously saved cache from a JSON file into memory.

        After this call, get_topology() is served from the restored cache
        without any MATLAB interaction.

        Parameters
        ----------
        path : str
            Source file path produced by save_cache().
        """
        with open(path) as f:
            payload = json.load(f)

        loaded = 0
        for epoch_str, snap_dict in payload.items():
            epoch = int(epoch_str)
            self._cache[epoch] = _dict_to_topo(snap_dict)
            loaded += 1

        epochs = sorted(self._cache.keys())
        print(f"[MatlabBridge] Loaded {loaded} epoch(s) "
              f"({epochs[0]}–{epochs[-1]}) ← {path}")

    def close(self) -> None:
        """Shut down the MATLAB engine (if running)."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
            print("[MatlabBridge] MATLAB engine stopped.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ────────────────────────────────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────────────────────────────────

def _matlab_to_numpy(matlab_array) -> "_np.ndarray":
    """Convert a matlab.double / matlab.int32 to a numpy ndarray."""
    import numpy as np
    return np.array(matlab_array)


# ── TopologySnapshot JSON serialisation helpers ──────────────────────────────

def _topo_to_dict(snap: TopologySnapshot) -> dict:
    """
    Serialise a TopologySnapshot to a plain JSON-compatible dict.

    isl_delay_ms tuple keys (s, t) are encoded as "s,t" strings.
    sat_positions tuple values are stored as [lat, lon, alt] lists.
    """
    return {
        "epoch":         snap.epoch,
        "num_satellites": snap.num_satellites,
        "isl_neighbors": {str(k): v for k, v in snap.isl_neighbors.items()},
        "isl_delay_ms":  {f"{s},{t}": d for (s, t), d in snap.isl_delay_ms.items()},
        "cpu_capacity":  {str(k): v for k, v in snap.cpu_capacity.items()},
        "sat_positions": {str(k): list(v) for k, v in snap.sat_positions.items()},
    }


def _dict_to_topo(d: dict) -> TopologySnapshot:
    """Deserialise a TopologySnapshot from the dict produced by _topo_to_dict."""
    return TopologySnapshot(
        epoch          = d["epoch"],
        num_satellites = d["num_satellites"],
        isl_neighbors  = {int(k): v for k, v in d["isl_neighbors"].items()},
        isl_delay_ms   = {
            (int(parts[0]), int(parts[1])): v
            for k, v in d["isl_delay_ms"].items()
            for parts in [k.split(",")]
        },
        cpu_capacity   = {int(k): v for k, v in d["cpu_capacity"].items()},
        sat_positions  = {
            int(k): (v[0], v[1], v[2])
            for k, v in d.get("sat_positions", {}).items()
        },
    )
