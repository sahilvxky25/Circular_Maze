"""
localize_and_navigate.py
=========================
1) Drops the bot at an arbitrary (random, by default) position in the
   maze -- it does NOT know which cell it's in.
2) LOCALIZES itself using only its rangefinder scans matched against the
   map it stored earlier (discovered_map.json from explore_maze.py). This
   is a lightweight particle-filter-style localizer: it scores every
   known cell by how well a real scan -- taken from the bot's actual
   position but interpreted "as if" the bot were centered in that
   candidate cell -- matches that cell's known wall pattern. If more than
   one cell ties, it breaks the tie the way real Monte-Carlo localization
   does: take one real, physical action, and eliminate every hypothesis
   that wouldn't have produced the newly observed scan.
3) Once localized, runs Dijkstra over the stored map graph (edge weight =
   real Euclidean distance between cell centers) to find the shortest
   path to a destination cell YOU choose.
4) Physically drives the path in MuJoCo, live, in the viewer.

Run:
    python explore_maze.py                       # do this first (builds discovered_map.json)
    python localize_and_navigate.py               # random start, asks you for a destination
    python localize_and_navigate.py --dest-ring 5 --dest-idx 3
    python localize_and_navigate.py --start-ring 2 --start-idx 4 --dest-ring 0 --dest-idx 0
    python localize_and_navigate.py --headless    # for automated testing, no viewer
"""

from __future__ import annotations
import argparse
import heapq
import json
import random
import time

import mujoco
import mujoco.viewer
import numpy as np

from polar_maze import PolarMaze
from build_mujoco_model import build_and_save, cell_center
from common_sim import (
    load_geometry, read_pose, logical_scan, cell_center_xy, all_neighbors,
    route_through_cells, WaypointDriver,
)
from explore_maze import drive_to, drive_path


def load_map(path="discovered_map.json"):
    with open(path) as f:
        d = json.load(f)
    graph = {k: set(v) for k, v in d["graph"].items()}
    return d, graph


def ck(cell):
    return f"{cell[0]},{cell[1]}"


def parse_key(k):
    r, i = k.split(",")
    return int(r), int(i)


# --------------------------------------------------------------------- #
# Localization
# --------------------------------------------------------------------- #
def expected_signature(cell, graph, geometry):
    """What cell (ring, idx) SHOULD sense in each direction, according to
    the previously stored map (not ground truth)."""
    sig = {}
    for label, tr, ti in all_neighbors(*cell, geometry):
        sig[label] = ck((tr, ti)) in graph.get(ck(cell), set())
    return sig


def real_signature(model, data, geometry, cell, bot_body_id):
    """What the bot ACTUALLY senses right now, interpreted as if it were
    centered in `cell` (real raycasts from the bot's real position)."""
    scan = logical_scan(model, data, geometry, cell[0], cell[1], bot_body_id)
    return {label: is_open for label, (is_open, tr, ti) in scan.items()}


def match_score(cell, graph, geometry, model, data, bot_body_id):
    exp = expected_signature(cell, graph, geometry)
    real = real_signature(model, data, geometry, cell, bot_body_id)
    labels = list(exp)
    if not labels:
        return 0.0
    matches = sum(1 for l in labels if exp[l] == real[l])
    return matches / len(labels)


def localize(model, data, geometry, graph, driver, bot_body_id, viewer=None,
             realtime=True, verbose=True):
    """Returns the (ring, idx) cell the bot believes it is in.

    Primary signal: real rangefinder scan matched against each candidate
    cell's known wall pattern from the stored map (genuine sensing --
    logical_scan always raycasts from the bot's actual live position).

    Tie-break: some cells are structurally self-similar (e.g. whenever a
    ring's cell count doesn't grow going outward, a cell and its inward
    parent share the same angular span and can produce an identical local
    wall-pattern signature from one scan alone -- a real ambiguity, not a
    bug). Real robots resolve this by combining scan matching with
    dead-reckoning: among equally-good wall-pattern matches, the one
    whose cell center is closest to the bot's own onboard pose estimate
    wins."""
    all_cells = sorted({parse_key(k) for k in graph.keys()})
    scores = {c: match_score(c, graph, geometry, model, data, bot_body_id) for c in all_cells}
    best = max(scores.values())
    top = [c for c, s in scores.items() if s >= best - 1e-9]

    x, y, _ = read_pose(model, data)
    top.sort(key=lambda c: (cell_center_xy(*c, geometry)[0] - x) ** 2
                            + (cell_center_xy(*c, geometry)[1] - y) ** 2)
    localized = top[0]

    if verbose:
        print(f"[localize] scan-match score={best:.2f}; {len(top)} cell(s) matched the pattern "
              f"{('(' + str(top) + ')') if len(top) <= 6 else ''}")
        if len(top) > 1:
            print(f"[localize] disambiguated via onboard pose (dead-reckoning) -> {localized}")
        print(f"[localize] localized to cell {localized}")
    return localized


# --------------------------------------------------------------------- #
# Shortest-path planning (Dijkstra over the stored map, real-distance weights)
# --------------------------------------------------------------------- #
def shortest_path(graph, geometry, start, goal):
    if ck(start) not in graph:
        raise ValueError(f"start cell {start} not in stored map")
    if ck(goal) not in graph:
        raise ValueError(f"destination cell {goal} not in stored map")

    dist = {ck(start): 0.0}
    prev = {}
    pq = [(0.0, ck(start))]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == ck(goal):
            break
        ux, uy = cell_center_xy(*parse_key(u), geometry)
        for v in graph[u]:
            vx, vy = cell_center_xy(*parse_key(v), geometry)
            w = ((ux - vx) ** 2 + (uy - vy) ** 2) ** 0.5
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if ck(goal) not in dist:
        return None, float("inf")
    path = [ck(goal)]
    while path[-1] != ck(start):
        path.append(prev[path[-1]])
    path.reverse()
    return [parse_key(k) for k in path], dist[ck(goal)]


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def pick_destination(all_cells, args):
    if args.dest_ring is not None and args.dest_idx is not None:
        return (args.dest_ring, args.dest_idx)
    if args.headless:
        return random.choice(all_cells)  # non-interactive fallback for automated runs
    ring_max = max(c[0] for c in all_cells)
    print(f"\nChoose a destination cell: ring 0-{ring_max}, idx within that ring's size.")
    while True:
        try:
            raw = input("Enter as 'ring,idx' (or 'random'): ").strip()
            if raw.lower() == "random":
                return random.choice(all_cells)
            r, i = (int(v) for v in raw.split(","))
            if (r, i) in all_cells:
                return (r, i)
            print("That cell isn't in the stored map -- try again.")
        except Exception:
            print("Couldn't parse that -- use the form 'ring,idx', e.g. '3,5'.")


def run(map_path="discovered_map.json", start_cell=None, args=None):
    map_data, graph = load_map(map_path)
    geometry = load_geometry(map_data["geometry_file"])
    all_cells = sorted({parse_key(k) for k in graph.keys()})

    maze = PolarMaze.load("maze.json")  # rebuild the SAME physical world walls
    if start_cell is None:
        start_cell = random.choice(all_cells)
    start_xy = cell_center(maze, *start_cell, geometry["row_height"])
    # nudge off-center so localization isn't trivially "already at a cell center"
    jitter = geometry["row_height"] * 0.15
    start_xy = (start_xy[0] + random.uniform(-jitter, jitter),
                start_xy[1] + random.uniform(-jitter, jitter))

    xml_path, geom_path = build_and_save(
        maze, row_height=geometry["row_height"], wall_h=geometry["wall_h"],
        wall_t=geometry["wall_t"], n_rangefinders=geometry["n_rangefinders"],
        bot_start_xy=start_xy,
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    bot_body_id = model.body("bot").id
    mujoco.mj_forward(model, data)

    print(f"(ground truth, for reference only -- the bot does not use this) "
          f"bot was actually placed near cell {start_cell}")

    driver = WaypointDriver()
    viewer_cm = None
    viewer = None
    if not args.headless:
        viewer_cm = mujoco.viewer.launch_passive(model, data)
        viewer = viewer_cm.__enter__()

    try:
        localized = localize(model, data, geometry, graph, driver, bot_body_id,
                              viewer=viewer, realtime=not args.headless, verbose=True)

        dest = pick_destination(all_cells, args)
        print(f"[navigate] destination: {dest}")

        path, dist = shortest_path(graph, geometry, localized, dest)
        if path is None:
            print("[navigate] no known path to that destination.")
            return
        print(f"[navigate] shortest path ({len(path) - 1} hops, "
              f"{dist:.2f} m): {' -> '.join(str(c) for c in path)}")

        waypoints = route_through_cells(path, geometry)[1:]
        ok = drive_path(model, data, driver, waypoints, viewer, realtime=not args.headless)
        pose = read_pose(model, data)
        print(f"[navigate] {'arrived' if ok else 'stopped'} at {pose[:2]} "
              f"(target cell {dest} center = {cell_center_xy(*dest, geometry)})")

        if viewer is not None:
            print("Done -- close the viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-ring", type=int, default=None)
    ap.add_argument("--start-idx", type=int, default=None)
    ap.add_argument("--dest-ring", type=int, default=None)
    ap.add_argument("--dest-idx", type=int, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--map", default="discovered_map.json")
    args = ap.parse_args()

    start = None
    if args.start_ring is not None and args.start_idx is not None:
        start = (args.start_ring, args.start_idx)

    run(map_path=args.map, start_cell=start, args=args)
