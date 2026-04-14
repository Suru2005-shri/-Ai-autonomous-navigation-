# simulation/environment.py
# ═══════════════════════════════════════════════════════════════
# 2D Grid World Simulation Environment
# ═══════════════════════════════════════════════════════════════
import numpy as np
import random
from typing import List, Tuple, Optional

FREE     = 0
OBSTACLE = 1
DYNAMIC  = 2
AGENT    = 3
GOAL     = 4
TRAIL    = 5


class DynamicObstacle:
    def __init__(self, start, waypoints):
        self.position  = list(start)
        self.waypoints = waypoints
        self.wp_index  = 0
        self.direction = 1

    def step(self, grid):
        if not self.waypoints:
            return
        target = self.waypoints[self.wp_index]
        dy = int(np.sign(target[0] - self.position[0]))
        dx = int(np.sign(target[1] - self.position[1]))
        nr, nc = self.position[0] + dy, self.position[1] + dx
        rows, cols = grid.shape
        if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr, nc] not in (OBSTACLE, AGENT, GOAL)):
            self.position = [nr, nc]
        if [int(target[0]), int(target[1])] == self.position:
            self.wp_index += self.direction
            if self.wp_index >= len(self.waypoints):
                self.direction = -1
                self.wp_index  = max(0, len(self.waypoints) - 2)
            elif self.wp_index < 0:
                self.direction = 1
                self.wp_index  = min(1, len(self.waypoints) - 1)


class Environment:
    def __init__(self, rows=20, cols=20, obstacle_ratio=0.18,
                 n_dynamic=4, seed=42):
        self.rows  = rows
        self.cols  = cols
        self.seed  = seed
        self.rng   = random.Random(seed)
        np.random.seed(seed)

        self.grid          = np.zeros((rows, cols), dtype=int)
        self.agent_pos     = [1, 1]
        self.goal_pos      = (rows - 2, cols - 2)
        self.dynamic_obs   = []
        self.trail         = []
        self.step_count    = 0
        self.reached_goal  = False
        self.collision     = False

        self._build_map(obstacle_ratio, n_dynamic)

    def _build_map(self, obstacle_ratio, n_dynamic):
        self.grid[:] = FREE

        # Border walls
        self.grid[0, :]  = OBSTACLE
        self.grid[-1, :] = OBSTACLE
        self.grid[:, 0]  = OBSTACLE
        self.grid[:, -1] = OBSTACLE

        # Scattered interior obstacles
        for r in range(2, self.rows - 2):
            for c in range(2, self.cols - 2):
                if self.rng.random() < obstacle_ratio * 0.55:
                    self.grid[r, c] = OBSTACLE

        # One horizontal wall with a gap
        mid_r = self.rows // 2
        for c in range(2, self.cols - 2):
            self.grid[mid_r, c] = OBSTACLE
        gap_c = self.rng.randint(3, self.cols - 5)
        for gc in range(gap_c, min(gap_c + 3, self.cols - 2)):
            self.grid[mid_r, gc] = FREE

        # One vertical wall with a gap
        mid_c = self.cols // 2
        for r in range(2, self.rows - 2):
            if self.grid[r, mid_c] == FREE:
                self.grid[r, mid_c] = OBSTACLE
        gap_r = self.rng.randint(3, self.rows - 5)
        for gr in range(gap_r, min(gap_r + 3, self.rows - 2)):
            self.grid[gr, mid_c] = FREE

        # Guaranteed corridors: left inner column + bottom inner row
        for r in range(1, self.rows - 1):
            self.grid[r, 1] = FREE
        for c in range(1, self.cols - 1):
            self.grid[self.rows - 2, c] = FREE

        # Clear start and goal areas
        self.agent_pos = [1, 1]
        self.goal_pos  = (self.rows - 2, self.cols - 2)
        for dr in range(3):
            for dc in range(3):
                self.grid[1 + dr, 1 + dc] = FREE
                self.grid[self.rows - 2 - dr, self.cols - 2 - dc] = FREE

        # Dynamic obstacles
        self.dynamic_obs = []
        placed, attempts = 0, 0
        while placed < n_dynamic and attempts < 300:
            r = self.rng.randint(3, self.rows - 4)
            c = self.rng.randint(3, self.cols - 4)
            if (self.grid[r, c] == FREE and
                    abs(r - 1) + abs(c - 1) > 4 and
                    abs(r - (self.rows - 2)) + abs(c - (self.cols - 2)) > 4):
                wps = [(r, c)]
                for _ in range(4):
                    nr = max(2, min(self.rows - 3, wps[-1][0] + self.rng.choice([-1, 0, 1])))
                    nc = max(2, min(self.cols - 3, wps[-1][1] + self.rng.choice([-1, 0, 1])))
                    wps.append((nr, nc))
                self.dynamic_obs.append(DynamicObstacle((r, c), wps))
                placed += 1
            attempts += 1

    def render(self):
        view = self.grid.copy()
        for (r, c) in self.trail:
            if view[r, c] == FREE:
                view[r, c] = TRAIL
        for dyn in self.dynamic_obs:
            r, c = dyn.position
            if view[r, c] not in (AGENT, GOAL):
                view[r, c] = DYNAMIC
        view[self.goal_pos[0],  self.goal_pos[1]]  = GOAL
        view[self.agent_pos[0], self.agent_pos[1]] = AGENT
        return view

    def get_obstacle_map(self):
        obs = (self.grid == OBSTACLE).astype(int)
        for dyn in self.dynamic_obs:
            r, c = dyn.position
            obs[r, c] = 1
        return obs

    def get_state(self):
        return {
            "agent":        tuple(self.agent_pos),
            "goal":         self.goal_pos,
            "step":         self.step_count,
            "reached_goal": self.reached_goal,
            "collision":    self.collision,
        }

    def move_agent(self, direction):
        dr, dc = direction
        new_r  = self.agent_pos[0] + dr
        new_c  = self.agent_pos[1] + dc
        if (new_r < 0 or new_r >= self.rows or
                new_c < 0 or new_c >= self.cols or
                self.grid[new_r, new_c] == OBSTACLE):
            self.collision = True
            return {"status": "collision", "reason": "static"}
        for dyn in self.dynamic_obs:
            if [new_r, new_c] == dyn.position:
                self.collision = True
                return {"status": "collision", "reason": "dynamic"}
        self.trail.append((self.agent_pos[0], self.agent_pos[1]))
        self.agent_pos = [new_r, new_c]
        self.step_count += 1
        if (new_r, new_c) == self.goal_pos:
            self.reached_goal = True
            return {"status": "goal"}
        if self.step_count % 2 == 0:
            for dyn in self.dynamic_obs:
                dyn.step(self.grid)
        return {"status": "moved"}

    def reset(self):
        self.agent_pos    = [1, 1]
        self.trail        = []
        self.step_count   = 0
        self.reached_goal = False
        self.collision    = False
