"""
src/visibility.py
=================
Geometric satellite visibility model for LEO constellations.

Two responsibilities
--------------------
1. Elevation angle — compute whether a satellite at a known (lat, lon, alt)
   is above the minimum elevation angle as seen from a ground station.

2. Walker-Star position approximation — compute approximate (lat, lon, alt)
   for each satellite in a Walker-Star constellation at a given epoch,
   using full orbital mechanics (circular orbit, J2-free propagation).
   Used by the pure-Python InstanceGenerator so it can apply visibility
   even without MATLAB.

Physical model
--------------
Visibility condition (ITU-R S.1257 / standard LEO link budget):
    A satellite is "visible" from a ground station when its elevation angle
    as seen from the station is >= min_elevation_deg (typically 10°).

Elevation formula (spherical-Earth):
    Given central angle c between sub-satellite point and ground station,
    and orbital altitude h:

        el = atan( (cos(c) - R/(R+h)) / sin(c) )   [radians]

    where R = 6371 km (mean Earth radius).

Walker-Star orbital mechanics:
    Circular orbit (e = 0), so true anomaly = mean anomaly.
    Argument of perigee is 0 (standard Walker convention).
    RAAN of plane p: Omega_p = 2*pi*p / P
    Mean anomaly of satellite q in plane p at t=0:
        M0 = 2*pi*q/t + 2*pi*p*F/T    (Walker phasing)
    where t = T/P (sats per plane), T = total sats, P = planes, F = phasing.
    Current mean anomaly: M(epoch_s) = M0 + n * epoch_s
    where n = sqrt(GM / (R+h)^3) is mean motion.
    ECI->ECEF rotation accounts for Earth's sidereal rotation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .types import SatID, TopologySnapshot, SFC


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM        = 6371.0          # mean Earth radius (km)
GM_EARTH               = 398600.4418     # gravitational parameter (km³/s²)
OMEGA_EARTH            = 7.2921150e-5    # Earth rotation rate (rad/s)
SPEED_OF_LIGHT_KM_PER_MS = 299.792458   # speed of light (km/ms)


# ---------------------------------------------------------------------------
# Elevation angle and access delay
# ---------------------------------------------------------------------------

def compute_access_delay_ms(
    user_lat_deg: float,
    user_lon_deg: float,
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_km: float,
) -> float:
    """
    One-way propagation delay from a ground user to a satellite (milliseconds).

    Uses the spherical-Earth slant-range formula:
        slant = sqrt(R² + (R+h)² - 2·R·(R+h)·cos(c))
    where c is the great-circle central angle between the sub-satellite point
    and the ground station, R is Earth's mean radius, and h is orbit altitude.
    """
    lat_u = math.radians(user_lat_deg)
    lon_u = math.radians(user_lon_deg)
    lat_s = math.radians(sat_lat_deg)
    lon_s = math.radians(sat_lon_deg)

    dlat = lat_s - lat_u
    dlon = lon_s - lon_u
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat_u) * math.cos(lat_s) * math.sin(dlon / 2) ** 2)
    c = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))

    R = EARTH_RADIUS_KM
    h = sat_alt_km
    slant_km = math.sqrt(R ** 2 + (R + h) ** 2 - 2.0 * R * (R + h) * math.cos(c))
    return slant_km / SPEED_OF_LIGHT_KM_PER_MS


def compute_elevation_deg(
    user_lat_deg: float,
    user_lon_deg: float,
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_km: float,
) -> float:
    """
    Return the elevation angle (degrees) from a ground user to a satellite.

    Uses the standard spherical-Earth formula.  Negative values mean the
    satellite is below the geometric horizon.
    """
    lat_u = math.radians(user_lat_deg)
    lon_u = math.radians(user_lon_deg)
    lat_s = math.radians(sat_lat_deg)
    lon_s = math.radians(sat_lon_deg)

    # Haversine great-circle central angle (radians)
    dlat = lat_s - lat_u
    dlon = lon_s - lon_u
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat_u) * math.cos(lat_s) * math.sin(dlon / 2) ** 2)
    c = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))

    if c < 1e-9:
        return 90.0  # satellite directly overhead

    R = EARTH_RADIUS_KM
    h = sat_alt_km
    el_rad = math.atan((math.cos(c) - R / (R + h)) / math.sin(c))
    return math.degrees(el_rad)


# ---------------------------------------------------------------------------
# Walker-Star position computation (pure Python, no MATLAB required)
# ---------------------------------------------------------------------------

def walker_star_positions(
    num_sats: int,
    num_planes: int,
    altitude_km: float,
    inclination_deg: float,
    phasing: int = 1,
    epoch_s: float = 0.0,
) -> List[Tuple[float, float, float]]:
    """
    Compute (lat_deg, lon_deg, alt_km) for each satellite in a Walker-Star
    constellation at the given epoch offset (seconds from reference time).

    Parameters
    ----------
    num_sats : int        Total satellites T.
    num_planes : int      Number of orbital planes P.
    altitude_km : float   Orbit altitude above Earth surface (km).
    inclination_deg : float  Orbit inclination (degrees).
    phasing : int         Walker phasing parameter F (default 1).
    epoch_s : float       Seconds elapsed since reference epoch (default 0).

    Returns
    -------
    List of (lat_deg, lon_deg, alt_km) with length num_sats.
    """
    R       = EARTH_RADIUS_KM
    r_orbit = R + altitude_km                           # orbital radius (km)
    n_mot   = math.sqrt(GM_EARTH / r_orbit ** 3)        # mean motion (rad/s)
    incl    = math.radians(inclination_deg)

    # Greenwich Sidereal Angle at epoch_s (simplified: zero at t=0)
    theta_gst = OMEGA_EARTH * epoch_s

    sats_per_plane = num_sats // num_planes
    positions: List[Tuple[float, float, float]] = []

    for s in range(num_sats):
        p = s // sats_per_plane          # plane index
        q = s %  sats_per_plane          # position within plane

        # RAAN of this plane
        raan = 2.0 * math.pi * p / num_planes

        # Mean anomaly at t = 0  (Walker phasing formula)
        M0 = (2.0 * math.pi * q / sats_per_plane
              + 2.0 * math.pi * p * phasing / num_sats)

        # Current argument of latitude (circular orbit: M = true anomaly)
        u = M0 + n_mot * epoch_s

        # Position in orbital (perifocal) frame — x along periapsis (ω=0)
        x_orb = r_orbit * math.cos(u)
        y_orb = r_orbit * math.sin(u)

        # Rotate to ECI:  R3(-RAAN) · R1(-incl) · [x_orb, y_orb, 0]
        cos_raan = math.cos(raan)
        sin_raan = math.sin(raan)
        cos_incl = math.cos(incl)
        sin_incl = math.sin(incl)

        x_eci = cos_raan * x_orb - sin_raan * cos_incl * y_orb
        y_eci = sin_raan * x_orb + cos_raan * cos_incl * y_orb
        z_eci = sin_incl * y_orb

        # Rotate ECI -> ECEF (Earth's sidereal rotation)
        cos_gst = math.cos(theta_gst)
        sin_gst = math.sin(theta_gst)
        x_ecef =  cos_gst * x_eci + sin_gst * y_eci
        y_ecef = -sin_gst * x_eci + cos_gst * y_eci
        z_ecef =  z_eci

        # ECEF -> geodetic (spherical-Earth approximation)
        r_vec  = math.sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2)
        lat_deg = math.degrees(math.asin(z_ecef / r_vec))
        lon_deg = math.degrees(math.atan2(y_ecef, x_ecef))
        alt_km  = r_vec - R

        positions.append((lat_deg, lon_deg, alt_km))

    return positions


# ---------------------------------------------------------------------------
# Visibility queries
# ---------------------------------------------------------------------------

def visible_satellites(
    topo: "TopologySnapshot",
    user_lat_deg: float,
    user_lon_deg: float,
    min_elevation_deg: float = 10.0,
) -> List["SatID"]:
    """
    Return the list of satellites visible from (user_lat_deg, user_lon_deg).

    If the topology carries no position data (sat_positions is empty), all
    satellites are returned so the rest of the pipeline stays functional.
    """
    if not topo.sat_positions:
        return list(range(topo.num_satellites))

    result: List[int] = []
    for s, (lat, lon, alt) in topo.sat_positions.items():
        el = compute_elevation_deg(user_lat_deg, user_lon_deg, lat, lon, alt)
        if el >= min_elevation_deg:
            result.append(s)
    return result


def all_pairs_sp_delays(topo: "TopologySnapshot") -> Dict[Tuple[int, int], float]:
    """
    All-pairs shortest-path ISL propagation delays (ms) via Dijkstra.

    Returns {(src, dst): delay_ms}.  Self-pairs have delay 0.
    """
    import heapq
    result: Dict[Tuple[int, int], float] = {}
    for src in range(topo.num_satellites):
        dist: Dict[int, float] = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, s = heapq.heappop(pq)
            if d > dist.get(s, float("inf")):
                continue
            for t in topo.neighbors_of(s):
                nd = d + topo.delay(s, t)
                if nd < dist.get(t, float("inf")):
                    dist[t] = nd
                    heapq.heappush(pq, (nd, t))
        for dst, d in dist.items():
            result[(src, dst)] = d
    return result


def precompute_user_visibility(
    topo: "TopologySnapshot",
    sfcs: List["SFC"],
    min_elevation_deg: float = 10.0,
) -> Dict[Tuple[int, int], List["SatID"]]:
    """
    Compute visible satellites for each (sfc_idx, user_id) pair.

    Users within the same slice may be geographically distributed and
    therefore see different satellite sets at any given epoch.

    Returns Dict[(sfc_idx, user_id) -> List[SatID]].  The coverage-gap
    fallback (return all satellites) is applied per-user so the problem
    remains feasible when a specific user has no visible satellite.
    """
    result: Dict[Tuple[int, int], List[int]] = {}
    fallback = list(range(topo.num_satellites))
    for sfc_idx, sfc in enumerate(sfcs):
        for u in sfc.user_ids:
            lat, lon = sfc.user_location(u)
            vis = visible_satellites(topo, lat, lon, min_elevation_deg)
            result[(sfc_idx, u)] = vis if vis else fallback
    return result


def precompute_slice_visibility(
    topo: "TopologySnapshot",
    sfcs: List["SFC"],
    min_elevation_deg: float = 10.0,
) -> Dict[int, List["SatID"]]:
    """
    Compute visible satellites for each SFC using the slice centre location.

    Retained for backward compatibility.  New code should call
    precompute_user_visibility() to get per-user visibility sets.

    Returns Dict[sfc_idx -> List[SatID]].
    """
    result: Dict[int, List[int]] = {}
    for sfc_idx, sfc in enumerate(sfcs):
        lat = getattr(sfc, "ground_lat", 0.0)
        lon = getattr(sfc, "ground_lon", 0.0)
        vis = visible_satellites(topo, lat, lon, min_elevation_deg)
        result[sfc_idx] = vis if vis else list(range(topo.num_satellites))
    return result
