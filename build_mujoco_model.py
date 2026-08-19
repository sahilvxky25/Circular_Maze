"""
build_mujoco_model.py
======================
Turns a `PolarMaze` (see polar_maze.py) into an actual MuJoCo world:
  * circular floor
  * radial walls  (straight boxes, between angularly-adjacent cells)
  * arc walls     (curved boundaries between rings, approximated by a
                    fan of short straight box segments)
  * a small differential-drive bot with:
        - 2 driven wheels (velocity actuators)
        - 2 passive low-friction casters for balance
        - N horizontal rangefinder sensors ("lidar") arranged in a ring
        - a pose sensor (framepos/framequat) standing in for wheel-encoder
          / IMU odometry (a real diff-drive robot integrates this from
          wheel speeds + gyro; we read it directly here to keep the
          low-level *motion controller* simple -- the actual maze
          "localization" task is solved separately, topologically, from
          the rangefinder wall pattern -- see localize_and_navigate.py)

Also writes a small geometry.json describing the polar grid in metric
units so other scripts can convert between (x, y) <-> (ring, idx)
without re-parsing the XML.
"""

from __future__ import annotations
import json
import math

from polar_maze import PolarMaze


def cell_geom(maze: PolarMaze, ring: int, idx: int, row_height: float):
    """Returns (r_in, r_out, theta0, theta1, cx, cy) for a cell, in meters/radians."""
    n = maze.ring_count(ring)
    span = 2 * math.pi / n
    theta0 = idx * span
    theta1 = theta0 + span
    r_in = ring * row_height
    r_out = (ring + 1) * row_height
    r_mid = 0.5 * (r_in + r_out)
    theta_mid = 0.5 * (theta0 + theta1)
    cx, cy = r_mid * math.cos(theta_mid), r_mid * math.sin(theta_mid)
    return r_in, r_out, theta0, theta1, cx, cy


def cell_center(maze: PolarMaze, ring: int, idx: int, row_height: float):
    _, _, _, _, cx, cy = cell_geom(maze, ring, idx, row_height)
    return cx, cy


def _box(name, cx, cy, length, thickness, height, direction_deg):
    """A wall segment: a box whose local +Y axis (the `length` side)
    points along `direction_deg` (degrees, measured like a normal
    math angle from +X axis). Rotating a box by (direction-90) about Z
    achieves this, since an unrotated box's +Y axis points at 90 deg."""
    euler_z = direction_deg - 90.0
    return (
        f'<geom name="{name}" type="box" '
        f'pos="{cx:.5f} {cy:.5f} {height/2:.5f}" '
        f'euler="0 0 {euler_z:.5f}" '
        f'size="{thickness/2:.5f} {length/2:.5f} {height/2:.5f}" '
        f'class="wall"/>'
    )


def build_wall_geoms(maze: PolarMaze, row_height: float, wall_h: float, wall_t: float,
                      arc_segment_len: float = 0.14) -> list[str]:
    geoms = []
    wid = 0

    # ---- radial walls: boundaries between cw-adjacent cells in a ring ----
    for row in maze.grid:
        if len(row) <= 1:
            continue  # single-cell ring (the hub) has no internal radial walls
        for cell in row:
            if cell.cw is cell:
                continue
            if not cell.is_linked(cell.cw):
                r_in, r_out, theta0, theta1, _, _ = cell_geom(maze, cell.ring, cell.idx, row_height)
                boundary_theta = theta1  # shared boundary with cw neighbor
                length = r_out - r_in
                r_mid = 0.5 * (r_in + r_out)
                cx = r_mid * math.cos(boundary_theta)
                cy = r_mid * math.sin(boundary_theta)
                direction_deg = math.degrees(boundary_theta)  # wall runs along the radius
                geoms.append(_box(f"wr_{wid}", cx, cy, length, wall_t, wall_h, direction_deg))
                wid += 1

    # ---- arc walls: boundaries between a cell and its inward parent ----
    for r in range(1, maze.n_rings):
        for child in maze.grid[r]:
            if not child.is_linked(child.inward):
                _, _, theta0, theta1, _, _ = cell_geom(maze, r, child.idx, row_height)
                radius = r * row_height
                geoms += _arc_segments(f"wa_{wid}", radius, theta0, theta1, wall_t, wall_h, arc_segment_len)
                wid += 1

    # ---- outer boundary: always solid ----
    outer_r = maze.n_rings * row_height
    for cell in maze.grid[-1]:
        _, _, theta0, theta1, _, _ = cell_geom(maze, maze.n_rings - 1, cell.idx, row_height)
        geoms += _arc_segments(f"wo_{wid}", outer_r, theta0, theta1, wall_t, wall_h, arc_segment_len)
        wid += 1

    return geoms


def _arc_segments(base_name, radius, theta0, theta1, thickness, height, target_len):
    span = theta1 - theta0
    arc_len = radius * span
    n_seg = max(1, math.ceil(arc_len / target_len))
    seg_span = span / n_seg
    seg_len = radius * seg_span * 1.05  # slight overlap so segments don't gap on curves
    out = []
    for s in range(n_seg):
        t_mid = theta0 + (s + 0.5) * seg_span
        cx, cy = radius * math.cos(t_mid), radius * math.sin(t_mid)
        direction_deg = math.degrees(t_mid) + 90.0  # tangential direction
        out.append(_box(f"{base_name}_{s}", cx, cy, seg_len, thickness, height, direction_deg))
    return out


BOT_XML = """
    <body name="bot" pos="{bx:.5f} {by:.5f} 0.05">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="0.5"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="0.5"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0.05"/>
      <site name="chassis_site" pos="0 0 0" size="0.01"/>
      <geom name="chassis" type="cylinder" size="0.09 0.035" pos="0 0 0" class="bot_body"/>
      <geom name="heading_mark" type="box" size="0.03 0.015 0.005" pos="0.07 0 0.045" class="bot_marker"/>
      <geom name="wheel_left_g" type="cylinder" size="0.045 0.015" pos="0 0.10 -0.005" euler="90 0 0" class="wheel"/>
      <geom name="wheel_right_g" type="cylinder" size="0.045 0.015" pos="0 -0.10 -0.005" euler="90 0 0" class="wheel"/>
{rangefinder_sites}
    </body>
"""


def build_rangefinder_sites(n_rf: int) -> tuple[str, list[float]]:
    """Evenly spaced horizontal rangefinder sites around the bot. Returns
    (xml_snippet, list_of_angles_rad) -- angle 0 = bot's forward (+X)."""
    sites = []
    angles = []
    for k in range(n_rf):
        a = 2 * math.pi * k / n_rf
        angles.append(a)
        zx, zy = math.cos(a), math.sin(a)
        sites.append(
            f'      <site name="rf_{k}" pos="0 0 0.01" '
            f'zaxis="{zx:.5f} {zy:.5f} 0" size="0.005"/>'
        )
    return "\n".join(sites), angles


def build_model_xml(maze: PolarMaze, row_height: float = 0.5, wall_h: float = 0.30,
                     wall_t: float = 0.03, n_rangefinders: int = 16,
                     bot_start_xy: tuple[float, float] = (0.0, 0.0)) -> tuple[str, dict]:
    outer_r = maze.n_rings * row_height
    wall_geoms = build_wall_geoms(maze, row_height, wall_h, wall_t)
    rf_xml, rf_angles = build_rangefinder_sites(n_rangefinders)
    bx, by = bot_start_xy

    rf_sensors = "\n".join(f'    <rangefinder name="s_rf_{k}" site="rf_{k}" cutoff="{outer_r*2.2:.3f}"/>'
                            for k in range(n_rangefinders))

    xml = f"""<mujoco model="polar_maze">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1200" offheight="1200"/>
  </visual>

  <default>
    <default class="wall">
      <geom type="box" rgba="0.55 0.55 0.62 1" contype="1" conaffinity="1"/>
    </default>
    <default class="bot_body">
      <geom rgba="0.85 0.25 0.15 1" contype="2" conaffinity="1" mass="0.9" group="3"/>
    </default>
    <default class="bot_marker">
      <geom rgba="1 1 1 1" contype="0" conaffinity="0" mass="0.001" group="3"/>
    </default>
    <default class="wheel">
      <geom rgba="0.15 0.15 0.15 1" contype="2" conaffinity="1" mass="0.05" group="3"/>
    </default>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.2 0.24" rgb2="0.22 0.24 0.28"
             width="512" height="512"/>
    <material name="floor_mat" texture="grid" texrepeat="30 30" reflectance="0.05"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.65 0.8" rgb2="0.05 0.05 0.1" width="256" height="256"/>
  </asset>

  <worldbody>
    <light name="top_light" pos="0 0 {outer_r*2.5:.2f}" dir="0 0 -1" diffuse="0.9 0.9 0.9" directional="true"/>
    <light name="fill_light" pos="{outer_r:.2f} {outer_r:.2f} {outer_r:.2f}" diffuse="0.35 0.35 0.35" directional="true"/>
    <geom name="floor" type="cylinder" size="{outer_r+wall_t:.4f} 0.02" pos="0 0 -0.02"
          material="floor_mat" contype="1" conaffinity="1"/>

{chr(10).join('    ' + g for g in wall_geoms)}

{BOT_XML.format(bx=bx, by=by, rangefinder_sites=rf_xml)}
  </worldbody>

  <actuator>
    <velocity name="motor_x" joint="slide_x" kv="8.0" ctrlrange="-3 3"/>
    <velocity name="motor_y" joint="slide_y" kv="8.0" ctrlrange="-3 3"/>
    <velocity name="motor_yaw" joint="yaw" kv="3.0" ctrlrange="-6 6"/>
  </actuator>

  <sensor>
{rf_sensors}
    <framepos name="s_pos" objtype="site" objname="chassis_site"/>
    <framequat name="s_quat" objtype="site" objname="chassis_site"/>
  </sensor>
</mujoco>
"""
    geometry = {
        "n_rings": maze.n_rings,
        "row_height": row_height,
        "outer_radius": outer_r,
        "wall_h": wall_h,
        "wall_t": wall_t,
        "ring_sizes": [len(row) for row in maze.grid],
        "n_rangefinders": n_rangefinders,
        "rangefinder_angles": rf_angles,
    }
    return xml, geometry


def build_and_save(maze: PolarMaze, xml_path="maze_world.xml", geom_path="geometry.json", **kwargs):
    start_cell = kwargs.pop("start_cell", None)
    row_height = kwargs.get("row_height", 0.5)
    if start_cell is not None:
        cx, cy = cell_center(maze, *start_cell, row_height)
        kwargs["bot_start_xy"] = (cx, cy)
    xml, geometry = build_model_xml(maze, **kwargs)
    with open(xml_path, "w") as f:
        f.write(xml)
    with open(geom_path, "w") as f:
        json.dump(geometry, f, indent=2)
    return xml_path, geom_path


if __name__ == "__main__":
    import mujoco

    maze = PolarMaze(n_rings=6, seed=1)
    maze.save("maze.json")
    xml_path, geom_path = build_and_save(maze, start_cell=(0, 0))
    print("Wrote", xml_path, "and", geom_path)

    # sanity check: does MuJoCo actually accept the generated model?
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    print("Model compiled & stepped OK. nq =", model.nq, " ngeom =", model.ngeom)
