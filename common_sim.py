"""
common_sim.py
==============
Shared low-level utilities used by both the exploring bot
(explore_maze.py) and the localizing/planning bot (localize_and_navigate.py):

  * geometry.json <-> (ring, idx) <-> (x, y) conversions
  * pose reading from MuJoCo sensors
  * exact wall/opening detection via raycasting (mujoco.mj_ray) at the
    four "logical" maze directions (outward / inward / cw / ccw) as seen
    from wherever the bot currently is -- this is genuine sensing (it
    queries the physics geometry, never the abstract maze.links data)
  * a simple proportional differential-drive controller that drives the
    bot from its current pose to a target (x, y) waypoint
"""

from __future__ import annotations
import json
import math
import numpy as np
import mujoco


# --------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------- #
def load_geometry(path="geometry.json") -> dict:
    with open(path) as f:
        return json.load(f)


def angle_to_idx(theta: float, ring: int, geometry: dict) -> int:
    n = geometry["ring_sizes"][ring]
    span = 2 * math.pi / n
    theta = theta % (2 * math.pi)
    return int(theta / span) % n


def xy_to_cell(x: float, y: float, geometry: dict) -> tuple[int, int]:
    row_height = geometry["row_height"]
    r = math.hypot(x, y)
    ring = min(int(r / row_height), geometry["n_rings"] - 1)
    theta = math.atan2(y, x)
    idx = angle_to_idx(theta, ring, geometry)
    return ring, idx


def cell_center_xy(ring: int, idx: int, geometry: dict) -> tuple[float, float]:
    row_height = geometry["row_height"]
    n = geometry["ring_sizes"][ring]
    span = 2 * math.pi / n
    theta_mid = (idx + 0.5) * span
    r_mid = (ring + 0.5) * row_height
    return r_mid * math.cos(theta_mid), r_mid * math.sin(theta_mid)


def cell_mid_angle(ring: int, idx: int, geometry: dict) -> float:
    n = geometry["ring_sizes"][ring]
    span = 2 * math.pi / n
    return (idx + 0.5) * span


# --------------------------------------------------------------------- #
# Pose reading
# --------------------------------------------------------------------- #
def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def read_pose(model, data) -> tuple[float, float, float]:
    """Returns (x, y, yaw) of the bot chassis, as read from the onboard
    pose sensor (stand-in for wheel-encoder/IMU odometry)."""
    pos = data.sensor("s_pos").data
    quat = data.sensor("s_quat").data
    return float(pos[0]), float(pos[1]), quat_to_yaw(quat)


# --------------------------------------------------------------------- #
# Exact raycast-based logical-direction scanning
# --------------------------------------------------------------------- #
# Bot collision/visual geoms all live in geom-group 3 (see build_mujoco_model.py)
# so raycasts used for wall-sensing can cleanly exclude the bot's own body
# (wheels/casters are separate child bodies, so a plain `bodyexclude` alone
# would miss them -- group filtering handles the whole kinematic tree at once).
_RAY_GEOMGROUP = np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8)


def _cast_ray(model, data, origin_xyz, direction_xy, exclude_body_id, max_dist):
    pnt = np.array(origin_xyz, dtype=np.float64)
    vec = np.array([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64)
    vec /= np.linalg.norm(vec)
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, vec, _RAY_GEOMGROUP, 1, exclude_body_id, geomid)
    if dist < 0:
        return max_dist
    return dist


def cw_neighbor(ring, idx, geometry):
    n = geometry["ring_sizes"][ring]
    return None if n <= 1 else (ring, (idx + 1) % n)


def ccw_neighbor(ring, idx, geometry):
    n = geometry["ring_sizes"][ring]
    return None if n <= 1 else (ring, (idx - 1) % n)


def inward_parent(ring, idx, geometry):
    if ring == 0:
        return None
    ratio = geometry["ring_sizes"][ring] // geometry["ring_sizes"][ring - 1]
    return (ring - 1, idx // ratio)


def outward_children(ring, idx, geometry):
    """A cell can have MORE THAN ONE outward child (a ring's cell count
    can multiply by more than 2 going outward -- e.g. the single center
    hub cell is adjacent to every cell in ring 1), so this returns a
    list, not a single neighbor."""
    if ring >= geometry["n_rings"] - 1:
        return []
    ratio = geometry["ring_sizes"][ring + 1] // geometry["ring_sizes"][ring]
    base = idx * ratio
    return [(ring + 1, base + k) for k in range(ratio)]


def all_neighbors(ring, idx, geometry):
    """Every geometric neighbor of a cell as (label, ring, idx) triples.
    Pure index/geometry math -- does not touch the maze graph's links."""
    out = []
    cw = cw_neighbor(ring, idx, geometry)
    if cw:
        out.append(("cw", *cw))
    ccw = ccw_neighbor(ring, idx, geometry)
    if ccw:
        out.append(("ccw", *ccw))
    par = inward_parent(ring, idx, geometry)
    if par:
        out.append(("inward", *par))
    for k, (tr, ti) in enumerate(outward_children(ring, idx, geometry)):
        out.append((f"outward_{k}", tr, ti))
    return out


def _neighbor_boundary_xy(ring, idx, label, tr, ti, geometry):
    """The exact boundary point between cell (ring, idx) and its neighbor
    (tr, ti) reached via `label`. Shared by logical_scan (ray aim point)
    and route_through_cells (waypoint insertion so straight-line legs
    pass cleanly through doorways instead of cutting corners)."""
    row_height = geometry["row_height"]
    theta_mid = cell_mid_angle(ring, idx, geometry)
    span = 2 * math.pi / geometry["ring_sizes"][ring]
    r_mid = (ring + 0.5) * row_height
    if label == "inward":
        r_in = ring * row_height
        return r_in * math.cos(theta_mid), r_in * math.sin(theta_mid)
    if label == "cw":
        a = theta_mid + span / 2
        return r_mid * math.cos(a), r_mid * math.sin(a)
    if label == "ccw":
        a = theta_mid - span / 2
        return r_mid * math.cos(a), r_mid * math.sin(a)
    # outward_k -- aim at that specific child's own mid-angle
    child_angle = cell_mid_angle(tr, ti, geometry)
    r_out = tr * row_height
    return r_out * math.cos(child_angle), r_out * math.sin(child_angle)


def logical_scan(model, data, geometry, ring: int, idx: int, bot_body_id: int) -> dict:
    """Casts a ray from the bot's current position toward the *exact*
    boundary point it would need to cross to reach each geometric
    neighbor of cell (ring, idx), and classifies that neighbor as open
    (passage) or blocked (wall) by comparing the raycast hit distance to
    the (exactly known) distance to that boundary point. This is genuine
    sensing: it queries physics geometry via raycasting and never
    inspects the maze graph's `links`.

    Returns {label: (is_open, target_ring, target_idx)} for every
    geometric neighbor -- cw, ccw, inward (at most one each) and
    outward_0..outward_k (there can be several, e.g. the center hub cell
    borders every cell in ring 1).

    Aiming at each neighbor's own exact boundary point (rather than a
    single generic "outward direction") matters on a polar grid: a
    cell can border several children at once, and a cell's own mid-angle
    can coincide exactly with a child-to-child seam whenever a ring's
    cell count multiplies going outward -- aiming loosely would graze a
    wall instead of cleanly entering one specific opening.
    """
    x, y, _ = read_pose(model, data)
    z = 0.06
    max_dist = 2.5 * geometry["outer_radius"]

    result = {}
    for label, tr, ti in all_neighbors(ring, idx, geometry):
        tx, ty = _neighbor_boundary_xy(ring, idx, label, tr, ti, geometry)
        dx, dy = tx - x, ty - y
        expected_dist = math.hypot(dx, dy)
        hit_dist = _cast_ray(model, data, (x, y, z), (dx, dy), bot_body_id, max_dist)
        is_open = hit_dist > expected_dist * 1.35
        result[label] = (is_open, tr, ti)
    return result


def route_through_cells(cell_path: list, geometry: dict) -> list:
    """Turns a path of (ring, idx) cells into a list of (x, y) waypoints
    that pass CLEANLY through each doorway between consecutive cells,
    instead of cutting straight cell-center-to-cell-center (which can
    clip a wall corner, since an opening isn't necessarily on the
    straight line between two cell centers several rings apart).

    For each hop, inserts a waypoint just past the shared boundary
    (nudged from the boundary point towards the next cell's center) in
    addition to the next cell's own center."""
    if not cell_path:
        return []
    waypoints = [cell_center_xy(*cell_path[0], geometry)]
    for i in range(len(cell_path) - 1):
        ring, idx = cell_path[i]
        tr, ti = cell_path[i + 1]
        label = None
        for lbl, nr, ni in all_neighbors(ring, idx, geometry):
            if (nr, ni) == (tr, ti):
                label = lbl
                break
        next_center = cell_center_xy(tr, ti, geometry)
        if label is not None:
            bx, by = _neighbor_boundary_xy(ring, idx, label, tr, ti, geometry)
            ncx, ncy = next_center
            ddx, ddy = ncx - bx, ncy - by
            d = math.hypot(ddx, ddy) or 1.0
            push = min(0.15, d * 0.5)
            waypoints.append((bx + ddx / d * push, by + ddy / d * push))
        waypoints.append(next_center)
    return waypoints


# --------------------------------------------------------------------- #
# Differential-drive waypoint controller
# --------------------------------------------------------------------- #
def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class WaypointDriver:
    """Simple proportional controller for the planar (slide_x, slide_y,
    yaw) actuated chassis: rotates to face the target, then drives
    forward while correcting heading, decelerating smoothly on approach
    (so it actually converges tightly on the target instead of
    overshooting at a constant speed). Returns world-frame
    (vx, vy, omega) actuator commands each call."""

    def __init__(self, base_speed=0.9, k_heading=4.0, turn_speed=1.8,
                 pos_tol=0.02, heading_tol_deep_turn=0.35, k_dist=3.5):
        self.base_speed = base_speed
        self.k_heading = k_heading
        self.turn_speed = turn_speed
        self.pos_tol = pos_tol
        self.heading_tol_deep_turn = heading_tol_deep_turn
        self.k_dist = k_dist

    def step(self, pose, target_xy):
        x, y, yaw = pose
        dx, dy = target_xy[0] - x, target_xy[1] - y
        dist = math.hypot(dx, dy)
        if dist < self.pos_tol:
            return 0.0, 0.0, 0.0, True
        target_heading = math.atan2(dy, dx)
        heading_err = wrap_angle(target_heading - yaw)
        if abs(heading_err) > self.heading_tol_deep_turn:
            omega = self.turn_speed * np.sign(heading_err)
            return 0.0, 0.0, omega, False
        speed = min(self.base_speed, self.k_dist * dist)
        vx = speed * math.cos(target_heading)
        vy = speed * math.sin(target_heading)
        omega = self.k_heading * heading_err
        return vx, vy, omega, False
