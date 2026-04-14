# src/path_planning.py
# ═══════════════════════════════════════════════════════════════
# Path Planning Module
# ═══════════════════════════════════════════════════════════════
# Implements three classic search algorithms:
#
#   A*        — heuristic search (fastest, industry standard)
#   Dijkstra  — uniform-cost search (optimal, no heuristic)
#   BFS       — breadth-first search (unweighted, simplest)
#
# All three return the shortest collision-free path from
# start → goal on the 2D grid.
#
# Industry context
# ─────────────────
#   • Autonomous vehicles use A* / D* / RRT variants for global
#     route planning before feeding a local planner.
#   • Warehouse robots (Amazon Kiva, Locus Robotics) use
#     multi-agent A* for coordinated navigation.
#   • Drone delivery systems use 3D variants of these algorithms.
# ═══════════════════════════════════════════════════════════════

import heapq
from collections import deque
from typing import List, Tuple, Optional, Dict
import numpy as np

# 4-directional movement (cardinal only)
MOVES_4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# 8-directional movement (diagonal allowed)
MOVES_8 = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    ( 1, 0), ( 1,-1), (0,-1), (-1,-1),
]


def _heuristic(a: Tuple[int, int], b: Tuple[int, int], method: str = "manhattan") -> float:
    """
    Admissible heuristic functions.
    Manhattan — used with 4-directional movement.
    Euclidean — used with 8-directional movement.
    Chebyshev — used with 8-directional movement (diagonal cost = 1).
    """
    if method == "manhattan":
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    elif method == "euclidean":
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    elif method == "chebyshev":
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    return 0.0


def _reconstruct(came_from: Dict, current: Tuple) -> List[Tuple[int, int]]:
    """Trace back the path from goal to start."""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    path.reverse()
    return path


# ── A* ────────────────────────────────────────────────────────
def astar(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    diagonal: bool = False,
) -> Tuple[Optional[List[Tuple[int, int]]], dict]:
    """
    A* path planning.

    Parameters
    ----------
    grid     : 2D numpy array (1 = obstacle, 0 = free)
    start    : (row, col) start cell
    goal     : (row, col) goal cell
    diagonal : allow diagonal movement

    Returns
    -------
    path  : list of (row, col) from start to goal, or None
    stats : { "nodes_explored": int, "path_length": int }
    """
    moves     = MOVES_8 if diagonal else MOVES_4
    heuristic = "chebyshev" if diagonal else "manhattan"
    rows, cols = grid.shape

    # open_set: (f_score, g_score, node)
    open_set  = []
    heapq.heappush(open_set, (0.0, 0.0, start))
    came_from = {}
    g_score   = {start: 0.0}
    explored  = 0

    while open_set:
        _, g, current = heapq.heappop(open_set)
        explored += 1

        if current == goal:
            path = _reconstruct(came_from, current)
            return path, {"nodes_explored": explored, "path_length": len(path)}

        for dr, dc in moves:
            nr, nc     = current[0] + dr, current[1] + dc
            neighbour  = (nr, nc)

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue

            # Diagonal moves cost √2, cardinal = 1
            step_cost  = 1.414 if (dr != 0 and dc != 0) else 1.0
            new_g      = g_score[current] + step_cost

            if new_g < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour]   = new_g
                f = new_g + _heuristic(neighbour, goal, heuristic)
                heapq.heappush(open_set, (f, new_g, neighbour))

    return None, {"nodes_explored": explored, "path_length": 0}


# ── Dijkstra ──────────────────────────────────────────────────
def dijkstra(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    diagonal: bool = False,
) -> Tuple[Optional[List[Tuple[int, int]]], dict]:
    """
    Dijkstra's algorithm — A* with heuristic = 0.
    Guaranteed optimal path. Slower than A* on large maps.
    """
    moves      = MOVES_8 if diagonal else MOVES_4
    rows, cols = grid.shape

    dist      = {start: 0.0}
    came_from = {}
    pq        = [(0.0, start)]
    explored  = 0

    while pq:
        d, current = heapq.heappop(pq)
        explored  += 1

        if current == goal:
            path = _reconstruct(came_from, current)
            return path, {"nodes_explored": explored, "path_length": len(path)}

        if d > dist.get(current, float("inf")):
            continue

        for dr, dc in moves:
            nr, nc    = current[0] + dr, current[1] + dc
            neighbour = (nr, nc)

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue

            step_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
            new_d     = dist[current] + step_cost

            if new_d < dist.get(neighbour, float("inf")):
                dist[neighbour]      = new_d
                came_from[neighbour] = current
                heapq.heappush(pq, (new_d, neighbour))

    return None, {"nodes_explored": explored, "path_length": 0}


# ── BFS ───────────────────────────────────────────────────────
def bfs(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    diagonal: bool = False,
) -> Tuple[Optional[List[Tuple[int, int]]], dict]:
    """
    Breadth-First Search — finds shortest path by hop count.
    Treats all moves as equal cost (no weights).
    """
    moves      = MOVES_8 if diagonal else MOVES_4
    rows, cols = grid.shape

    queue     = deque([start])
    came_from = {start: None}
    explored  = 0

    while queue:
        current = queue.popleft()
        explored += 1

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, {"nodes_explored": explored, "path_length": len(path)}

        for dr, dc in moves:
            nr, nc    = current[0] + dr, current[1] + dc
            neighbour = (nr, nc)

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue
            if neighbour not in came_from:
                came_from[neighbour] = current
                queue.append(neighbour)

    return None, {"nodes_explored": explored, "path_length": 0}


# ── Planner wrapper ───────────────────────────────────────────
class PathPlanner:
    """
    High-level planner that selects an algorithm and handles
    dynamic re-planning when obstacles change.
    """

    ALGORITHMS = {"astar": astar, "dijkstra": dijkstra, "bfs": bfs}

    def __init__(self, algorithm: str = "astar", diagonal: bool = False):
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {list(self.ALGORITHMS)}")
        self.algorithm = algorithm
        self.diagonal  = diagonal
        self.last_plan: Optional[List[Tuple[int, int]]] = None
        self.stats:     dict = {}

    def plan(
        self,
        obstacle_map: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Compute a path from start to goal on obstacle_map.
        Stores and returns the path.
        """
        fn   = self.ALGORITHMS[self.algorithm]
        path, self.stats = fn(obstacle_map, start, goal, self.diagonal)
        self.last_plan   = path

        if path:
            print(f"  [{self.algorithm.upper()}] Path found: "
                  f"{len(path)} steps, "
                  f"{self.stats['nodes_explored']} nodes explored")
        else:
            print(f"  [{self.algorithm.upper()}] No path found — "
                  f"{self.stats['nodes_explored']} nodes explored")

        return path

    def replan(
        self,
        obstacle_map: np.ndarray,
        current_pos: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Re-plan from current_pos (used after dynamic obstacle appears)."""
        return self.plan(obstacle_map, current_pos, goal)

    def next_step(
        self,
        current_pos: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """
        Return the movement delta (dr, dc) to reach the next
        waypoint in the current plan.
        """
        if not self.last_plan or len(self.last_plan) < 2:
            return None

        # Find current position in path
        try:
            idx = self.last_plan.index(current_pos)
        except ValueError:
            # Current position not in plan — re-plan needed
            return None

        if idx + 1 >= len(self.last_plan):
            return None

        next_wp = self.last_plan[idx + 1]
        dr = next_wp[0] - current_pos[0]
        dc = next_wp[1] - current_pos[1]
        return (dr, dc)
