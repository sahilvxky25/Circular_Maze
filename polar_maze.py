"""
polar_maze.py
==============
Generates a *perfect* circular (polar) maze: concentric rings, each split
into an increasing number of cells the further out you go, connected by
a randomized recursive-backtracker so there is exactly one path between
any two cells (no loops, fully connected, fully solvable).

The maze is stored as an abstract graph (rings/cells/links) with no
notion of MuJoCo yet -- `build_mujoco_model.py` turns this graph into an
actual MJCF world with walls.

Usage:
    maze = PolarMaze(n_rings=6, seed=42)
    maze.save("maze.json")
    maze2 = PolarMaze.load("maze.json")
"""

from __future__ import annotations
import json
import math
import random
from dataclasses import dataclass, field


@dataclass(eq=False)  # identity-based equality/hash -- cells are graph nodes
class Cell:
    ring: int
    idx: int                      # index within its ring
    cw: "Cell" = None             # clockwise neighbor (same ring)
    ccw: "Cell" = None            # counter-clockwise neighbor (same ring)
    inward: "Cell" = None         # single parent cell one ring closer to center
    outward: list = field(default_factory=list)  # child cells one ring further out
    links: set = field(default_factory=set)       # neighbors with NO wall between them

    def link(self, other: "Cell"):
        self.links.add(other)
        other.links.add(self)

    def is_linked(self, other: "Cell") -> bool:
        return other in self.links

    def neighbors(self):
        n = []
        if self.cw is not None:
            n.append(self.cw)
        if self.ccw is not None and self.ccw is not self.cw:
            n.append(self.ccw)
        if self.inward is not None:
            n.append(self.inward)
        n.extend(self.outward)
        return n

    def key(self):
        return (self.ring, self.idx)


class PolarMaze:
    """A circular maze on a polar grid of `n_rings` rings."""

    def __init__(self, n_rings: int = 6, seed: int | None = None, build: bool = True):
        self.n_rings = n_rings
        self.seed = seed
        self.grid: list[list[Cell]] = []
        if build:
            self._make_grid()
            self._carve()

    # ------------------------------------------------------------------ #
    # Grid construction
    # ------------------------------------------------------------------ #
    def _make_grid(self):
        """Builds the polar grid: ring 0 is a single hub cell at the
        center, and each subsequent ring roughly doubles cell count as
        needed to keep cells approximately square (classic 'Theta'
        polar-grid algorithm, Jamis Buck)."""
        n_rings = self.n_rings
        row_height = 1.0 / n_rings
        grid = [[Cell(0, 0)]]
        for r in range(1, n_rings):
            radius = r / n_rings
            circumference = 2 * math.pi * radius
            prev_count = len(grid[r - 1])
            estimated_cell_width = circumference / prev_count
            ratio = max(1, round(estimated_cell_width / row_height))
            cell_count = prev_count * ratio
            grid.append([Cell(r, i) for i in range(cell_count)])

        # link clockwise / counter-clockwise neighbors within each ring
        for row in grid:
            n = len(row)
            for i, cell in enumerate(row):
                cell.cw = row[(i + 1) % n]
                cell.ccw = row[(i - 1) % n]

        # link inward / outward neighbors between adjacent rings
        for r in range(1, n_rings):
            row, prev_row = grid[r], grid[r - 1]
            ratio = len(row) // len(prev_row)
            for i, cell in enumerate(row):
                parent = prev_row[i // ratio]
                cell.inward = parent
                parent.outward.append(cell)

        self.grid = grid

    def _carve(self):
        """Randomized recursive backtracker (iterative, stack-based) over
        the polar grid graph -> produces a perfect maze (spanning tree)."""
        rng = random.Random(self.seed)
        cells = [c for row in self.grid for c in row]
        start = rng.choice(cells)
        visited = {start}
        stack = [start]
        while stack:
            current = stack[-1]
            unvisited = [n for n in current.neighbors() if n not in visited]
            if not unvisited:
                stack.pop()
                continue
            nxt = rng.choice(unvisited)
            current.link(nxt)
            visited.add(nxt)
            stack.append(nxt)

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def cell(self, ring: int, idx: int) -> Cell:
        row = self.grid[ring]
        return row[idx % len(row)]

    def all_cells(self):
        return [c for row in self.grid for c in row]

    def ring_count(self, ring: int) -> int:
        return len(self.grid[ring])

    def random_cell(self, rng: random.Random | None = None) -> Cell:
        rng = rng or random
        return rng.choice(self.all_cells())

    # ------------------------------------------------------------------ #
    # Serialization -- store the maze topology (rings/cells/links) so the
    # exploring bot's discovered map, and the later localizer/planner,
    # can be checked / cross-used without rebuilding MuJoCo.
    # ------------------------------------------------------------------ #
    def to_dict(self):
        return {
            "n_rings": self.n_rings,
            "seed": self.seed,
            "ring_sizes": [len(row) for row in self.grid],
            "links": sorted(
                {
                    tuple(sorted((c.key(), o.key())))
                    for c in self.all_cells()
                    for o in c.links
                }
            ),
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PolarMaze":
        with open(path) as f:
            d = json.load(f)
        maze = cls(n_rings=d["n_rings"], seed=d.get("seed"), build=False)
        maze.grid = [[Cell(r, i) for i in range(size)] for r, size in enumerate(d["ring_sizes"])]
        # relink cw/ccw/inward/outward exactly like _make_grid does
        for row in maze.grid:
            n = len(row)
            for i, cell in enumerate(row):
                cell.cw = row[(i + 1) % n]
                cell.ccw = row[(i - 1) % n]
        for r in range(1, maze.n_rings):
            row, prev_row = maze.grid[r], maze.grid[r - 1]
            ratio = len(row) // len(prev_row)
            for i, cell in enumerate(row):
                parent = prev_row[i // ratio]
                cell.inward = parent
                parent.outward.append(cell)
        for (r1, i1), (r2, i2) in d["links"]:
            maze.cell(r1, i1).link(maze.cell(r2, i2))
        return maze


if __name__ == "__main__":
    m = PolarMaze(n_rings=6, seed=1)
    print(f"Generated maze with {sum(len(r) for r in m.grid)} cells across {m.n_rings} rings")
    print("Ring sizes:", [len(r) for r in m.grid])
    m.save("maze.json")
    m2 = PolarMaze.load("maze.json")
    print("Reloaded OK, links:", len(m2.to_dict()["links"]))
