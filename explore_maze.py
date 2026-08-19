"""
explore_maze.py
================
Places the bot anywhere in the maze and has it autonomously traverse the
ENTIRE maze (iterative depth-first search with physical backtracking),
using only its onboard raycast "lidar" to decide, at each cell, which
neighboring cells are reachable. As it goes, it builds and saves a graph
map of the maze (`discovered_map.json`) purely from what it has sensed --
never from the ground-truth `maze.json` used to build the world.

Run:
    python explore_maze.py                      # visualize in the MuJoCo viewer
    python explore_maze.py --headless            # run fast, no viewer (for testing)
    python explore_maze.py --rings 8 --seed 7
    python explore_maze.py --start-ring 2 --start-idx 3   # start anywhere, not just center
"""

from __future__ import annotations
import argparse
import json
import time

import mujoco
import mujoco.viewer
import numpy as np

from polar_maze import PolarMaze
from build_mujoco_model import build_and_save
from common_sim import (
    load_geometry, read_pose, logical_scan, cell_center_xy, route_through_cells, WaypointDriver,
)


def drive_to(model, data, driver, target_xy, viewer=None, max_seconds=25.0, realtime=True):
    """Runs the physics simulation, commanding actuator velocities each
    step, until the bot reaches target_xy (or times out)."""
    dt = model.opt.timestep
    steps = 0
    max_steps = int(max_seconds / dt)
    while steps < max_steps:
        pose = read_pose(model, data)
        vx, vy, omega, arrived = driver.step(pose, target_xy)
        data.ctrl[0] = vx
        data.ctrl[1] = vy
        data.ctrl[2] = omega
        mujoco.mj_step(model, data)
        steps += 1
        if viewer is not None:
            viewer.sync()
            if realtime:
                time.sleep(dt)
            if not viewer.is_running():
                return False
        if arrived:
            return True
    return False  # timed out -- shouldn't normally happen


def drive_path(model, data, driver, waypoints, viewer=None, max_seconds=25.0, realtime=True):
    """Drives through a list of (x, y) waypoints in order (see
    common_sim.route_through_cells) -- used instead of a single
    straight-line drive_to() for cell-to-cell moves so the bot passes
    cleanly through doorways rather than cutting a corner against a wall."""
    for wp in waypoints:
        ok = drive_to(model, data, driver, wp, viewer, max_seconds, realtime)
        if not ok:
            return False
    return True


def explore(n_rings=6, seed=1, start_cell=(0, 0), headless=False, realtime=True,
            out_map="discovered_map.json", out_xml="maze_world.xml", out_geom="geometry.json"):
    maze = PolarMaze(n_rings=n_rings, seed=seed)
    maze.save("maze.json")  # ground truth, kept only for offline grading/inspection

    xml_path, geom_path = build_and_save(
        maze, xml_path=out_xml, geom_path=out_geom, start_cell=start_cell
    )
    geometry = load_geometry(geom_path)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    bot_body_id = model.body("bot").id
    mujoco.mj_forward(model, data)

    driver = WaypointDriver()
    viewer_cm = None
    viewer = None
    if not headless:
        viewer_cm = mujoco.viewer.launch_passive(model, data)
        viewer = viewer_cm.__enter__()

    try:
        visited = {start_cell}
        stack = [start_cell]
        scanned = {}
        graph = {}  # "r,i" -> list of "r,i" (json-friendly string keys)

        def key(c):
            return f"{c[0]},{c[1]}"

        def add_edge(a, b):
            graph.setdefault(key(a), [])
            graph.setdefault(key(b), [])
            if key(b) not in graph[key(a)]:
                graph[key(a)].append(key(b))
            if key(a) not in graph[key(b)]:
                graph[key(b)].append(key(a))

        t0 = time.time()
        while stack:
            cell = stack[-1]
            if cell not in scanned:
                scan = logical_scan(model, data, geometry, cell[0], cell[1], bot_body_id)
                scanned[cell] = scan
                for label, (is_open, tr, ti) in scan.items():
                    if is_open:
                        add_edge(cell, (tr, ti))

            candidates = [
                (tr, ti) for (_, (is_open, tr, ti)) in scanned[cell].items()
                if is_open and (tr, ti) not in visited
            ]
            if candidates:
                nxt = candidates[0]
                waypoints = route_through_cells([cell, nxt], geometry)[1:]
                ok = drive_path(model, data, driver, waypoints, viewer, realtime=realtime)
                if not ok and viewer is not None and not viewer.is_running():
                    break
                visited.add(nxt)
                stack.append(nxt)
            else:
                stack.pop()
                if stack:
                    waypoints = route_through_cells([cell, stack[-1]], geometry)[1:]
                    ok = drive_path(model, data, driver, waypoints, viewer, realtime=realtime)
                    if not ok and viewer is not None and not viewer.is_running():
                        break

        elapsed = time.time() - t0
        total_cells = sum(geometry["ring_sizes"])
        print(f"Exploration finished in {elapsed:.1f}s (sim). "
              f"Visited {len(visited)}/{total_cells} cells.")

        with open(out_map, "w") as f:
            json.dump({
                "n_rings": n_rings,
                "geometry_file": out_geom,
                "start_cell": list(start_cell),
                "visited_count": len(visited),
                "total_cells": total_cells,
                "graph": graph,
            }, f, indent=2)
        print("Saved discovered map to", out_map)
        return graph, visited, total_cells

    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rings", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start-ring", type=int, default=0)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-realtime", action="store_true", help="run viewer as fast as possible")
    args = ap.parse_args()

    explore(
        n_rings=args.rings, seed=args.seed,
        start_cell=(args.start_ring, args.start_idx),
        headless=args.headless, realtime=not args.no_realtime,
    )
