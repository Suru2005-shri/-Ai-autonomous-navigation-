# src/navigation.py
# ═══════════════════════════════════════════════════════════════
# Navigation Controller — Decision Making & Obstacle Avoidance
# ═══════════════════════════════════════════════════════════════
# This module acts as the "brain" of the autonomous agent.
# It combines:
#   • Global path from the planner (macro route)
#   • Local sensor readings from perception (micro avoidance)
#   • Reactive behaviour when blocked (re-plan or dodge)
#
# Decision hierarchy
# ──────────────────
#   1. Is the goal reached?  → stop
#   2. Is a collision imminent? → emergency avoid
#   3. Is the planned path clear? → follow it
#   4. Is the path blocked? → re-plan
#   5. No path available? → rotate / wait
# ═══════════════════════════════════════════════════════════════

import numpy as np
from typing import Optional, Tuple, List, Dict
from src.path_planning import PathPlanner
from src.perception    import PerceptionSystem

# Cardinal direction deltas
DIRECTION_MAP: Dict[str, Tuple[int, int]] = {
    "N": (-1,  0),
    "E": ( 0,  1),
    "S": ( 1,  0),
    "W": ( 0, -1),
}

# Priority order when avoiding obstacles (try these rotations)
AVOIDANCE_ORDER = ["N", "E", "S", "W", "NE", "SE", "SW", "NW"]
DIAG_MAP: Dict[str, Tuple[int, int]] = {
    "NE": (-1,  1), "SE": ( 1,  1),
    "SW": ( 1, -1), "NW": (-1, -1),
}
ALL_MOVES = {**DIRECTION_MAP, **DIAG_MAP}


class NavigationController:
    """
    Main navigation controller.

    Parameters
    ----------
    algorithm  : path-planning algorithm ('astar', 'dijkstra', 'bfs')
    replan_interval : re-plan every N steps even if no collision
    max_steps   : max steps before the run is declared failed
    """

    def __init__(
        self,
        algorithm:       str = "astar",
        replan_interval: int = 10,
        max_steps:       int = 500,
    ):
        self.planner    = PathPlanner(algorithm=algorithm, diagonal=False)
        self.perception = PerceptionSystem(lidar_range=8)

        self.replan_interval = replan_interval
        self.max_steps       = max_steps

        # Metrics
        self.replan_count    = 0
        self.avoid_count     = 0
        self.step_log:  List[dict] = []

    # ── Main planning call ───────────────────────────────────
    def plan_route(self, env) -> bool:
        """
        Compute an initial route from the environment.
        Returns True if a path was found.
        """
        obs_map = env.get_obstacle_map()
        start   = tuple(env.agent_pos)
        goal    = env.goal_pos

        path = self.planner.plan(obs_map, start, goal)
        return path is not None

    # ── Single-step decision ─────────────────────────────────
    def decide(self, env) -> dict:
        """
        Decide the next move given current environment state.

        Returns
        -------
        {
          "action"   : (dr, dc) move delta,
          "decision" : str description,
          "perception": perception dict,
        }
        """
        state      = env.get_state()
        agent_pos  = state["agent"]
        goal       = state["goal"]
        step       = state["step"]

        # Build full obstacle map (static + dynamic)
        obs_map    = env.get_obstacle_map()

        # Perception reading
        percept    = self.perception.perceive(obs_map, list(agent_pos))
        proximity  = percept["proximity"]

        # ── Re-plan if interval reached or path broken ───────
        if step % self.replan_interval == 0:
            self.planner.plan(obs_map, agent_pos, goal)
            self.replan_count += 1

        # ── Emergency avoidance ───────────────────────────────
        # If LiDAR reads very close obstacle ahead, override path
        lidar = percept["lidar"]
        if lidar.get("N", 9) <= 1 and lidar.get("S", 9) <= 1:
            # Completely boxed in north/south — try east
            return self._reactive_move(obs_map, agent_pos, percept,
                                       "Emergency avoid (boxed)")

        # ── Follow planned path ───────────────────────────────
        delta = self.planner.next_step(agent_pos)

        if delta is not None:
            dr, dc     = delta
            nr, nc     = agent_pos[0] + dr, agent_pos[1] + dc
            rows, cols = obs_map.shape

            # Check if next planned step is now blocked
            if (0 <= nr < rows and 0 <= nc < cols and obs_map[nr, nc] == 0):
                return {
                    "action":    delta,
                    "decision":  "follow_plan",
                    "perception": percept,
                }
            else:
                # Blocked — re-plan immediately
                self.planner.plan(obs_map, agent_pos, goal)
                self.replan_count += 1

                delta2 = self.planner.next_step(agent_pos)
                if delta2:
                    return {
                        "action":    delta2,
                        "decision":  "replan_follow",
                        "perception": percept,
                    }

        # ── Fallback: reactive greedy move toward goal ────────
        return self._reactive_move(obs_map, agent_pos, percept, "reactive_greedy")

    # ── Reactive / greedy fallback ───────────────────────────
    def _reactive_move(
        self,
        obs_map:   np.ndarray,
        agent_pos: Tuple[int, int],
        percept:   dict,
        reason:    str,
    ) -> dict:
        """
        Greedy fallback: among free directions, pick the one
        that gets closest to goal (Euclidean distance).
        """
        goal    = (obs_map.shape[0] - 2, obs_map.shape[1] - 2)
        rows, cols = obs_map.shape
        best_d  = float("inf")
        best_mv = (0, 0)

        for name, (dr, dc) in ALL_MOVES.items():
            nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if obs_map[nr, nc] == 1:
                continue
            d = np.sqrt((nr - goal[0]) ** 2 + (nc - goal[1]) ** 2)
            if d < best_d:
                best_d  = d
                best_mv = (dr, dc)

        self.avoid_count += 1
        return {
            "action":    best_mv,
            "decision":  reason,
            "perception": percept,
        }

    # ── Full run loop ────────────────────────────────────────
    def run(self, env, verbose: bool = True) -> dict:
        """
        Execute the full navigation run.

        Returns a summary dict with metrics.
        """
        self.replan_count = 0
        self.avoid_count  = 0
        self.step_log     = []

        env.reset()
        self.plan_route(env)

        if verbose:
            print(f"\n  Start : {tuple(env.agent_pos)}")
            print(f"  Goal  : {env.goal_pos}")

        frames: List[np.ndarray] = []

        for step in range(self.max_steps):
            state = env.get_state()
            if state["reached_goal"] or state["collision"]:
                break

            decision = self.decide(env)
            action   = decision["action"]

            # Record frame
            frames.append(env.render().copy())

            # Execute move
            result = env.move_agent(action)

            self.step_log.append({
                "step":     step,
                "pos":      tuple(env.agent_pos),
                "decision": decision["decision"],
                "result":   result["status"],
            })

            if result["status"] == "goal":
                frames.append(env.render().copy())
                break
            elif result["status"] == "collision":
                if verbose:
                    print(f"  ✗ Collision at step {step} pos {tuple(env.agent_pos)}")
                break

        state   = env.get_state()
        success = state["reached_goal"]

        summary = {
            "success":      success,
            "steps":        state["step"],
            "replan_count": self.replan_count,
            "avoid_count":  self.avoid_count,
            "frames":       frames,
            "trail":        list(env.trail),
            "path_planned": self.planner.last_plan,
        }

        if verbose:
            status = "✓ REACHED GOAL" if success else "✗ FAILED"
            print(f"\n  {status} in {state['step']} steps  "
                  f"| replans: {self.replan_count}  "
                  f"| avoidances: {self.avoid_count}")

        return summary
