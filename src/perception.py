# src/perception.py
# ═══════════════════════════════════════════════════════════════
# Perception Module — Sensor & Object Detection
# ═══════════════════════════════════════════════════════════════
# Simulates the sensors an autonomous agent uses to perceive its
# environment:
#
#   LiDAR scan  — 8-direction distance rays
#   Camera FOV  — forward-facing 5×5 patch
#   Proximity   — cardinal + diagonal occupancy checks
#
# In a real system these would come from physical sensors.
# Here we derive them directly from the grid map, which is
# mathematically equivalent to a perfect LiDAR/camera reading.
# ═══════════════════════════════════════════════════════════════

import numpy as np
from typing import Dict, List, Tuple

# 8-direction ray cast offsets (N, NE, E, SE, S, SW, W, NW)
DIRECTIONS_8 = [
    (-1,  0), (-1,  1), (0,  1), (1,  1),
    ( 1,  0), ( 1, -1), (0, -1), (-1, -1),
]
DIR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Cardinal directions only (for movement)
DIRECTIONS_4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR4_NAMES   = ["N", "E", "S", "W"]


class LiDARSensor:
    """
    Simulated LiDAR — fires rays in 8 directions and returns
    the distance (in grid cells) to the nearest obstacle.

    Parameters
    ----------
    max_range : int — maximum ray length in cells
    """

    def __init__(self, max_range: int = 8):
        self.max_range = max_range

    def scan(
        self,
        grid: np.ndarray,
        agent_pos: List[int],
    ) -> Dict[str, float]:
        """
        Cast 8 rays from agent_pos and return distances.

        Returns
        -------
        dict  { "N": dist, "NE": dist, …, "NW": dist }
        """
        rows, cols  = grid.shape
        distances   = {}

        for (dr, dc), name in zip(DIRECTIONS_8, DIR_NAMES):
            dist = 0
            r, c = agent_pos
            hit  = False

            for _ in range(self.max_range):
                r += dr
                c += dc
                dist += 1

                if r < 0 or r >= rows or c < 0 or c >= cols:
                    hit = True
                    break
                if grid[r, c] == 1:   # obstacle
                    hit = True
                    break

            distances[name] = float(dist) if hit else float(self.max_range)

        return distances


class CameraSensor:
    """
    Simulated forward-facing camera — returns a small patch
    of the grid centred on the agent (agent's local view).

    Mimics a mono camera with a narrow field of view.
    """

    def __init__(self, fov_rows: int = 5, fov_cols: int = 5):
        self.fov_rows = fov_rows
        self.fov_cols = fov_cols

    def observe(
        self,
        grid: np.ndarray,
        agent_pos: List[int],
    ) -> np.ndarray:
        """
        Return a (fov_rows × fov_cols) patch centred on the agent.
        Out-of-bounds cells are treated as obstacles (value 1).
        """
        rows, cols = grid.shape
        half_r = self.fov_rows // 2
        half_c = self.fov_cols // 2
        patch  = np.ones((self.fov_rows, self.fov_cols), dtype=int)

        for i, dr in enumerate(range(-half_r, half_r + 1)):
            for j, dc in enumerate(range(-half_c, half_c + 1)):
                r = agent_pos[0] + dr
                c = agent_pos[1] + dc
                if 0 <= r < rows and 0 <= c < cols:
                    patch[i, j] = grid[r, c]

        return patch


class ObjectDetector:
    """
    Detects objects in the agent's surroundings and classifies them.

    Object classes (simulated):
      - WALL       : static boundary / building
      - OBSTACLE   : static interior obstacle
      - DYNAMIC    : moving obstacle (pedestrian / vehicle)
      - FREE_SPACE : navigable cell
      - GOAL       : destination marker
    """

    # Map grid values → class labels
    LABEL_MAP = {
        0: "FREE_SPACE",
        1: "OBSTACLE",
        2: "DYNAMIC_OBJECT",
        3: "AGENT",
        4: "GOAL",
        5: "TRAIL",
    }

    def detect(
        self,
        patch: np.ndarray,
    ) -> List[dict]:
        """
        Run detection on a camera patch.

        Returns a list of detected objects, each with:
          { "label": str, "position": (row, col), "distance": float }
        where (row, col) is relative to the patch centre.
        """
        detections = []
        centre     = (patch.shape[0] // 2, patch.shape[1] // 2)

        for r in range(patch.shape[0]):
            for c in range(patch.shape[1]):
                val   = patch[r, c]
                label = self.LABEL_MAP.get(val, "UNKNOWN")
                if label in ("OBSTACLE", "DYNAMIC_OBJECT", "GOAL"):
                    dr   = r - centre[0]
                    dc   = c - centre[1]
                    dist = float(np.sqrt(dr ** 2 + dc ** 2))
                    detections.append({
                        "label":    label,
                        "position": (dr, dc),
                        "distance": round(dist, 2),
                    })

        return detections


class ProximitySensor:
    """
    Simple 8-cell proximity check — returns which neighbouring
    cells are occupied.  Used by the collision-avoidance layer.
    """

    def check(
        self,
        grid: np.ndarray,
        agent_pos: List[int],
    ) -> Dict[str, bool]:
        """
        Returns a dict: { "N": blocked, "NE": blocked, … }
        True  = that direction is blocked
        False = that direction is free
        """
        rows, cols  = grid.shape
        blocked     = {}

        for (dr, dc), name in zip(DIRECTIONS_8, DIR_NAMES):
            r = agent_pos[0] + dr
            c = agent_pos[1] + dc

            if r < 0 or r >= rows or c < 0 or c >= cols:
                blocked[name] = True
            else:
                blocked[name] = (grid[r, c] == 1)

        return blocked


class PerceptionSystem:
    """
    Aggregates all sensors into a single perception reading.
    This is what the navigation controller queries each timestep.
    """

    def __init__(self, lidar_range: int = 8):
        self.lidar    = LiDARSensor(max_range=lidar_range)
        self.camera   = CameraSensor(fov_rows=7, fov_cols=7)
        self.detector = ObjectDetector()
        self.prox     = ProximitySensor()

    def perceive(
        self,
        grid: np.ndarray,
        agent_pos: List[int],
    ) -> dict:
        """
        Full perception cycle.

        Returns
        -------
        {
          "lidar"      : { dir: distance, … },
          "camera"     : np.ndarray patch,
          "detections" : [ { label, position, distance }, … ],
          "proximity"  : { dir: blocked, … },
        }
        """
        patch = self.camera.observe(grid, agent_pos)
        return {
            "lidar":      self.lidar.scan(grid, agent_pos),
            "camera":     patch,
            "detections": self.detector.detect(patch),
            "proximity":  self.prox.check(grid, agent_pos),
        }
