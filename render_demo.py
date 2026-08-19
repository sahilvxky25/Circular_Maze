"""
render_demo.py
===============
Optional convenience script: renders an offscreen MP4 showing (1) the
bot fully exploring the maze and (2) the bot localizing itself and
driving the shortest path to a destination -- useful if you want a
video to look at without opening the interactive MuJoCo viewer.

If you have a display, you don't need this: just run explore_maze.py /
localize_and_navigate.py without --headless and watch the live
mujoco.viewer window instead.

Run:
    python render_demo.py --rings 6 --seed 1 --dest-ring 0 --dest-idx 0
"""

from __future__ import annotations
import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np

from polar_maze import PolarMaze
from build_mujoco_model import build_and_save, cell_center
from common_sim import load_geometry, read_pose, logical_scan, cell_center_xy, route_through_cells, WaypointDriver
from localize_and_navigate import load_map, localize, shortest_path, parse_key


def make_camera(model):
    cam = mujoco.MjvCamera()
    cam.lookat = [0, 0, 0]
    cam.distance = model.stat.extent * 1.55
    cam.azimuth = 90
    cam.elevation = -90
    return cam


class FrameGrabber:
    def __init__(self, model, data, writer, cam, every_n=25, size=480):
        self.renderer = mujoco.Renderer(model, height=size, width=size)
        self.data = data
        self.writer = writer
        self.cam = cam
        self.every_n = every_n
        self.count = 0

    def sync(self):
        self.count += 1
        if self.count % self.every_n != 0:
            return
        self.renderer.update_scene(self.data, camera=self.cam)
        self.writer.append_data(self.renderer.render())

    def is_running(self):
        return True


def run_demo(n_rings, seed, dest, out_path="demo.mp4", fps=30):
    from explore_maze import drive_path as explore_drive_path  # noqa
    import explore_maze as em

    maze = PolarMaze(n_rings=n_rings, seed=seed)
    maze.save("maze.json")
    xml_path, geom_path = build_and_save(maze, start_cell=(0, 0))
    geometry = load_geometry(geom_path)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    bot_body_id = model.body("bot").id
    mujoco.mj_forward(model, data)
    driver = WaypointDriver(base_speed=1.6, k_dist=5.0)
    cam = make_camera(model)

    writer = imageio.get_writer(out_path, fps=fps)
    grabber = FrameGrabber(model, data, writer, cam)

    print("Rendering full exploration ...")
    visited = {(0, 0)}
    stack = [(0, 0)]
    scanned = {}
    graph = {}

    def key(c):
        return f"{c[0]},{c[1]}"

    def add_edge(a, b):
        graph.setdefault(key(a), [])
        graph.setdefault(key(b), [])
        if key(b) not in graph[key(a)]:
            graph[key(a)].append(key(b))
        if key(a) not in graph[key(b)]:
            graph[key(b)].append(key(a))

    while stack:
        cell = stack[-1]
        if cell not in scanned:
            scan = logical_scan(model, data, geometry, cell[0], cell[1], bot_body_id)
            scanned[cell] = scan
            for label, (is_open, tr, ti) in scan.items():
                if is_open:
                    add_edge(cell, (tr, ti))
        candidates = [(tr, ti) for (_, (is_open, tr, ti)) in scanned[cell].items()
                      if is_open and (tr, ti) not in visited]
        if candidates:
            nxt = candidates[0]
            wps = route_through_cells([cell, nxt], geometry)[1:]
            em.drive_path(model, data, driver, wps, grabber, realtime=False)
            visited.add(nxt)
            stack.append(nxt)
        else:
            stack.pop()
            if stack:
                wps = route_through_cells([cell, stack[-1]], geometry)[1:]
                em.drive_path(model, data, driver, wps, grabber, realtime=False)

    print(f"Exploration done, visited {len(visited)} cells. Rendering localize + navigate ...")

    # respawn the bot somewhere else in the maze for the localization demo
    import random
    all_cells = sorted({parse_key(k) for k in graph.keys()})
    start_cell = random.choice([c for c in all_cells if c != (0, 0)])
    start_xy = cell_center(maze, *start_cell, geometry["row_height"])
    data.qpos[model.joint("slide_x").qposadr[0]] = start_xy[0]
    data.qpos[model.joint("slide_y").qposadr[0]] = start_xy[1]
    mujoco.mj_forward(model, data)
    print("(ground truth, for reference only) bot respawned near", start_cell)

    localized = localize(model, data, geometry, graph, driver, bot_body_id, viewer=grabber, verbose=True)
    path, dist = shortest_path(graph, geometry, localized, dest)
    print(f"Localized to {localized}. Shortest path to {dest}: {len(path)-1} hops, {dist:.2f} m")
    waypoints = route_through_cells(path, geometry)[1:]
    em.drive_path(model, data, driver, waypoints, grabber, realtime=False)

    writer.close()
    print("Saved video to", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rings", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dest-ring", type=int, default=0)
    ap.add_argument("--dest-idx", type=int, default=0)
    ap.add_argument("--out", default="demo.mp4")
    args = ap.parse_args()
    run_demo(args.rings, args.seed, (args.dest_ring, args.dest_idx), out_path=args.out)
