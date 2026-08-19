# Circular Maze Robot — MuJoCo

A polar/circular maze, generated fresh each run, that a simulated robot
(1) fully explores and maps from scratch using only onboard sensing, then
(2) can be dropped into blind, localize itself in, and drive the shortest
path to any destination you choose — all simulated and visualized live in
MuJoCo.

## Files

| File | Purpose |
|---|---|
| `polar_maze.py` | Generates the circular maze itself: concentric rings of cells carved into a perfect maze (randomized recursive backtracker on a polar grid). Pure graph logic, no MuJoCo. |
| `build_mujoco_model.py` | Converts a `PolarMaze` into an actual MJCF world: floor, radial + curved walls, and a small robot (chassis + decorative wheels + a ring of rangefinder "lidar" sensors + pose sensor). |
| `common_sim.py` | Shared utilities: `(ring, idx) <-> (x, y)` conversions, exact-raycast wall/opening sensing (`logical_scan`), and the waypoint-following drive controller. |
| `explore_maze.py` | **Phase 1.** Bot starts anywhere, does an iterative DFS with physical backtracking, sensing its way around, and saves everything it has learned to `discovered_map.json`. |
| `localize_and_navigate.py` | **Phase 2.** Bot is dropped somewhere new (it doesn't know where), figures out which cell it's in by matching real sensor scans against the stored map, then plans and drives the shortest path (Dijkstra) to a destination you pick. |
| `render_demo.py` | Optional: renders an offscreen MP4 of the whole thing, for machines without a display. |

## Quick start

```bash
pip install mujoco numpy

# 1. Explore and map the maze (opens a live MuJoCo viewer window)
python explore_maze.py --rings 6 --seed 1

# 2. Drop the bot somewhere new; it localizes itself and drives to a
#    destination you type in when prompted (or pass --dest-ring/--dest-idx)
python localize_and_navigate.py
```

Both scripts accept `--headless` to run without opening a viewer window
(useful for servers / testing) and `--start-ring/--start-idx` to control
where the bot begins.

This directory already ships with a pre-explored 6-ring maze
(`maze.json`, `geometry.json`, `discovered_map.json`) and a sample video
(`demo_4ring.mp4`) so you can try `localize_and_navigate.py` immediately
without waiting for exploration first.

## How each part works

**Maze generation** (`polar_maze.py`): rings are laid out so cells stay
roughly square as the circle gets bigger (each ring's cell count is
chosen from the previous ring's), then a randomized DFS carves a perfect
maze (spanning tree — exactly one path between any two cells) over the
whole polar grid graph.

**The world** (`build_mujoco_model.py`): every un-linked boundary between
two cells becomes a wall — straight boxes for the radial (angle)
boundaries, a fan of short boxes approximating an arc for the ring
boundaries. The robot is a small chassis that slides/rotates in the
plane (`slide_x`, `slide_y`, `yaw` joints + velocity actuators) — this
keeps it a real MuJoCo rigid body with mass, inertia, and wall collision,
while avoiding the tip-over instability that true small-scale wheel
contact physics runs into. It carries a ring of `rangefinder` sensors and
a pose sensor (standing in for wheel-encoder/IMU odometry).

**Sensing** (`common_sim.logical_scan`): rather than reading the maze's
wall data directly, the bot casts real rays (`mujoco.mj_ray`) from its
actual position toward the exact point where each neighboring cell's
shared boundary would be, and checks whether the ray reaches
significantly past that point (open) or stops right at it (wall). This
is genuine sensing — it never touches the maze generator's internal
`links` — and it's what both exploration and localization are built on.

**Exploration** (`explore_maze.py`): classic DFS-with-backtracking,
except every step is a real physical drive through the simulator. At
each new cell it senses which neighbors are open, drives to an unvisited
one, and backtracks (physically driving back) when it hits a dead end.
Since the maze is a spanning tree, this is guaranteed to visit every
cell. Discovered edges are written to `discovered_map.json`.

**Localization** (`localize_and_navigate.py`): for every cell in the
stored map, it checks whether a real scan taken from the bot's actual
position — but interpreted "as if" the bot were centered in that
candidate cell — matches that cell's known wall pattern. Cells that are
structurally self-similar (e.g. a ring that doesn't grow going outward
makes a cell and its inward parent look identical from one scan) can tie;
those are broken using the bot's own onboard pose estimate (dead
reckoning) — exactly how real robots combine odometry with feature/scan
matching.

**Path planning**: Dijkstra over the stored map graph, edge weights =
real Euclidean distance between cell centers, from the localized cell to
whatever destination cell you choose.
