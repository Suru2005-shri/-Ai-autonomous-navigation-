#!/usr/bin/env python3
# main.py
# ═══════════════════════════════════════════════════════════════
#  AI-Based Autonomous Navigation System
#  Main Pipeline Runner
# ═══════════════════════════════════════════════════════════════
#
#  Usage
#  ─────
#    python main.py                    # default A* navigation
#    python main.py --algo dijkstra    # use Dijkstra planner
#    python main.py --algo bfs         # use BFS planner
#    python main.py --compare          # run all 3 + comparison chart
#    python main.py --grid 30          # 30×30 grid
#    python main.py --no-anim          # skip animation (faster)
#
# ═══════════════════════════════════════════════════════════════

import argparse
import sys
import time
import os
import json

sys.path.insert(0, os.path.dirname(__file__))


def run_single(algo: str, grid_size: int, animate: bool, verbose: bool = True) -> dict:
    """Run one complete navigation trial with a given algorithm."""
    from simulation.environment import Environment
    from src.navigation          import NavigationController
    from src.perception          import PerceptionSystem
    from src.visualizer          import (
        plot_environment,
        plot_planned_path,
        plot_navigation_result,
        plot_lidar_scan,
        plot_decision_log,
        save_animation_frames,
    )

    print(f"\n{'='*60}")
    print(f"  Algorithm : {algo.upper()}")
    print(f"  Grid      : {grid_size}×{grid_size}")
    print(f"{'='*60}")

    # ── Build environment ────────────────────────────────────
    env = Environment(
        rows=grid_size, cols=grid_size,
        obstacle_ratio=0.18,
        n_dynamic=4,
        seed=42,
    )

    # ── Plot initial map ─────────────────────────────────────
    print("\n[1] ENVIRONMENT")
    plot_environment(env.render(), title=f"Environment Map ({grid_size}×{grid_size})")

    # ── Plan initial route ───────────────────────────────────
    print("\n[2] PATH PLANNING")
    from src.path_planning import PathPlanner
    planner   = PathPlanner(algorithm=algo, diagonal=False)
    obs_map   = env.get_obstacle_map()
    init_path = planner.plan(obs_map, tuple(env.agent_pos), env.goal_pos)
    plot_planned_path(
        env.render(),
        init_path,
        tuple(env.agent_pos),
        env.goal_pos,
        algo=algo.upper(),
    )

    # ── LiDAR scan snapshot ──────────────────────────────────
    print("\n[3] PERCEPTION SNAPSHOT")
    perc = PerceptionSystem(lidar_range=8)
    percept = perc.perceive(obs_map, list(env.agent_pos))
    plot_lidar_scan(percept["lidar"], step=0)
    dets = percept["detections"]
    print(f"  Detected objects: {len(dets)}")
    for d in dets[:5]:
        print(f"    {d['label']:20s}  dist={d['distance']:.2f}  pos={d['position']}")

    # ── Full navigation run ──────────────────────────────────
    print("\n[4] NAVIGATION RUN")
    t0 = time.time()
    controller = NavigationController(
        algorithm=algo,
        replan_interval=8,
        max_steps=400,
    )
    summary = controller.run(env, verbose=verbose)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.2f}s")

    # ── Result charts ────────────────────────────────────────
    print("\n[5] VISUALISATION")
    final_grid = env.render()
    plot_navigation_result(final_grid, init_path, summary)

    if summary["step_log"] if hasattr(summary, "step_log") else controller.step_log:
        plot_decision_log(controller.step_log)

    if animate and summary["frames"]:
        save_animation_frames(summary["frames"])

    # ── Save metrics JSON ────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    metrics = {
        "algorithm":    algo,
        "grid_size":    grid_size,
        "success":      summary["success"],
        "steps":        summary["steps"],
        "replan_count": summary["replan_count"],
        "avoid_count":  summary["avoid_count"],
        "nodes_explored": planner.stats.get("nodes_explored", 0),
        "path_length":    planner.stats.get("path_length",    0),
        "elapsed_s":      round(elapsed, 3),
    }
    with open(f"outputs/metrics_{algo}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓ Metrics saved → outputs/metrics_{algo}.json")

    return metrics


def run_comparison(grid_size: int) -> None:
    """Run all three algorithms and generate comparison chart."""
    from src.visualizer import plot_algorithm_comparison

    print("\n" + "="*60)
    print("  ALGORITHM COMPARISON: A* vs Dijkstra vs BFS")
    print("="*60)

    results = {}
    for algo in ["astar", "dijkstra", "bfs"]:
        metrics = run_single(algo, grid_size, animate=False, verbose=False)
        results[algo.upper()] = {
            "nodes_explored": metrics["nodes_explored"],
            "path_length":    metrics["path_length"],
            "steps":          metrics["steps"],
        }

    plot_algorithm_comparison(results)

    print("\n  ── Comparison Summary ──────────────────────────────")
    print(f"  {'Algorithm':<12} {'Nodes':>8} {'Path Len':>10} {'Steps':>8}")
    print(f"  {'-'*42}")
    for algo, r in results.items():
        print(f"  {algo:<12} {r['nodes_explored']:>8} {r['path_length']:>10} {r['steps']:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Autonomous Navigation System"
    )
    parser.add_argument("--algo",    type=str, default="astar",
                        choices=["astar", "dijkstra", "bfs"],
                        help="Path planning algorithm (default: astar)")
    parser.add_argument("--grid",    type=int, default=20,
                        help="Grid size NxN (default: 20)")
    parser.add_argument("--compare", action="store_true",
                        help="Run all 3 algorithms and compare")
    parser.add_argument("--no-anim", action="store_true",
                        help="Skip animation generation")
    args = parser.parse_args()

    print("=" * 60)
    print("  AI-Based Autonomous Navigation System")
    print("=" * 60)

    if args.compare:
        run_comparison(args.grid)
    else:
        run_single(
            algo=args.algo,
            grid_size=args.grid,
            animate=not args.no_anim,
        )

    print("\n" + "="*60)
    print("  ✓ Complete — check outputs/images/ for all charts")
    print("="*60)


if __name__ == "__main__":
    main()
