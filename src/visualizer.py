# src/visualizer.py
# ═══════════════════════════════════════════════════════════════
# Visualisation Module
# ═══════════════════════════════════════════════════════════════
# Renders the simulation as publication-quality charts and
# animations saved to outputs/images/.
#
# Charts generated
# ─────────────────
#  1. environment_map.png     — initial grid with obstacles
#  2. planned_path.png        — A* path overlaid on grid
#  3. navigation_result.png   — final trail + agent + goal
#  4. lidar_scan.png          — polar LiDAR reading
#  5. algorithm_comparison.png— A* vs Dijkstra vs BFS stats
#  6. animation frames        — PNG sequence for GIF creation
# ═══════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors  as mcolors
import matplotlib.animation as animation
import numpy as np
import os
from typing import Optional, List, Tuple

IMG_DIR = "outputs/images"
os.makedirs(IMG_DIR, exist_ok=True)

# ── Colour palette for grid cells ────────────────────────────
# 0=FREE, 1=OBSTACLE, 2=DYNAMIC, 3=AGENT, 4=GOAL, 5=TRAIL
CELL_COLORS = {
    0: "#F8F9FA",   # free space — light grey
    1: "#343A40",   # obstacle   — dark grey
    2: "#F77F00",   # dynamic    — amber
    3: "#0077B6",   # agent      — blue
    4: "#2DC653",   # goal       — green
    5: "#ADE8F4",   # trail      — light blue
}

CMAP_LIST = [
    CELL_COLORS[i] for i in range(6)
]


def _make_cmap():
    """Create a discrete colourmap for the 6 cell types."""
    return mcolors.ListedColormap(CMAP_LIST)


def _cell_legend():
    return [
        mpatches.Patch(color=CELL_COLORS[0], label="Free space"),
        mpatches.Patch(color=CELL_COLORS[1], label="Static obstacle"),
        mpatches.Patch(color=CELL_COLORS[2], label="Dynamic obstacle"),
        mpatches.Patch(color=CELL_COLORS[3], label="Agent (robot)"),
        mpatches.Patch(color=CELL_COLORS[4], label="Goal"),
        mpatches.Patch(color=CELL_COLORS[5], label="Trail"),
    ]


# ─────────────────────────────────────────────────────────────
# 1. Environment map
# ─────────────────────────────────────────────────────────────
def plot_environment(grid: np.ndarray, title: str = "Environment Map") -> str:
    """Render the raw environment grid."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap=_make_cmap(), vmin=0, vmax=5, interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    ax.legend(handles=_cell_legend(), loc="upper right",
              fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path = f"{IMG_DIR}/environment_map.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path}")
    return path


# ─────────────────────────────────────────────────────────────
# 2. Planned path overlay
# ─────────────────────────────────────────────────────────────
def plot_planned_path(
    grid:  np.ndarray,
    path:  Optional[List[Tuple[int, int]]],
    start: Tuple[int, int],
    goal:  Tuple[int, int],
    algo:  str = "A*",
) -> str:
    """Show the grid with the planned path drawn as a coloured line."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap=_make_cmap(), vmin=0, vmax=5, interpolation="nearest")

    if path:
        rows_p = [p[0] for p in path]
        cols_p = [p[1] for p in path]
        ax.plot(cols_p, rows_p, color="#E63946", lw=2.5,
                label=f"{algo} path ({len(path)} steps)", zorder=5)
        # Start and goal markers
        ax.plot(start[1], start[0], "o", ms=12, color="#0077B6",
                zorder=6, label="Start")
        ax.plot(goal[1],  goal[0],  "*", ms=15, color="#2DC653",
                zorder=6, label="Goal")

    ax.set_title(f"Planned Path — {algo} Algorithm", fontsize=14, fontweight="bold")
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    path_out = f"{IMG_DIR}/planned_path.png"
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path_out}")
    return path_out


# ─────────────────────────────────────────────────────────────
# 3. Navigation result (final frame)
# ─────────────────────────────────────────────────────────────
def plot_navigation_result(
    final_grid: np.ndarray,
    planned_path: Optional[List[Tuple[int, int]]],
    summary: dict,
) -> str:
    """Show the final state with trail + planned path overlay."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(final_grid, cmap=_make_cmap(), vmin=0, vmax=5,
              interpolation="nearest")

    # Planned path ghost
    if planned_path:
        pr = [p[0] for p in planned_path]
        pc = [p[1] for p in planned_path]
        ax.plot(pc, pr, color="#E63946", lw=1.5, alpha=0.5,
                linestyle="--", label="Planned path")

    status = "SUCCESS ✓" if summary["success"] else "FAILED ✗"
    color  = "#2DC653"   if summary["success"] else "#E63946"
    ax.set_title(
        f"Navigation Result — {status}   "
        f"({summary['steps']} steps | {summary['replan_count']} replans)",
        fontsize=13, fontweight="bold", color=color,
    )
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    ax.legend(handles=_cell_legend() + [
        mpatches.Patch(color="#E63946", label="Planned path", alpha=0.5)
    ], loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path_out = f"{IMG_DIR}/navigation_result.png"
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path_out}")
    return path_out


# ─────────────────────────────────────────────────────────────
# 4. LiDAR polar scan
# ─────────────────────────────────────────────────────────────
def plot_lidar_scan(lidar_data: dict, step: int = 0) -> str:
    """
    Polar plot of LiDAR distances.
    Shows the agent's 8-direction sensor reading at a given step.
    """
    dir_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    angles    = np.linspace(0, 2 * np.pi, 9)[:-1]   # 8 directions
    distances = [lidar_data.get(d, 8) for d in dir_order]

    # Close the polygon
    angles_plot    = np.append(angles,    angles[0])
    distances_plot = np.append(distances, distances[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.plot(angles_plot, distances_plot, color="#0077B6", lw=2)
    ax.fill(angles_plot, distances_plot, color="#ADE8F4", alpha=0.4)

    ax.set_xticks(angles)
    ax.set_xticklabels(dir_order, fontsize=11)
    ax.set_title(f"LiDAR Scan — Step {step}", fontsize=13,
                 fontweight="bold", pad=20)
    ax.set_rmax(8)
    plt.tight_layout()
    path_out = f"{IMG_DIR}/lidar_scan.png"
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path_out}")
    return path_out


# ─────────────────────────────────────────────────────────────
# 5. Algorithm comparison bar chart
# ─────────────────────────────────────────────────────────────
def plot_algorithm_comparison(results: dict) -> str:
    """
    Side-by-side comparison of A*, Dijkstra, BFS:
    nodes explored, path length, steps taken.
    """
    algos  = list(results.keys())
    n      = len(algos)
    x      = np.arange(n)
    width  = 0.25

    metrics = {
        "Nodes Explored": [results[a].get("nodes_explored", 0) for a in algos],
        "Path Length":    [results[a].get("path_length",    0) for a in algos],
        "Steps Taken":    [results[a].get("steps",          0) for a in algos],
    }

    colors = ["#0077B6", "#2DC653", "#F77F00"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, vals) in enumerate(metrics.items()):
        ax.bar(x + i * width, vals, width, label=label, color=colors[i],
               alpha=0.85, edgecolor="white")

    ax.set_title("Algorithm Comparison: A* vs Dijkstra vs BFS",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(algos, fontsize=12)
    ax.set_ylabel("Value")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path_out = f"{IMG_DIR}/algorithm_comparison.png"
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path_out}")
    return path_out


# ─────────────────────────────────────────────────────────────
# 6. Step-by-step animation (saved as PNG sequence + GIF)
# ─────────────────────────────────────────────────────────────
def save_animation_frames(
    frames: List[np.ndarray],
    max_frames: int = 60,
) -> str:
    """
    Save every N-th frame as a PNG and stitch into an animated GIF.
    """
    os.makedirs(f"{IMG_DIR}/frames", exist_ok=True)
    step    = max(1, len(frames) // max_frames)
    sampled = frames[::step]

    cmap = _make_cmap()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")

    saved_paths = []
    for i, frame in enumerate(sampled):
        ax.clear()
        ax.imshow(frame, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
        ax.set_title(f"Navigation Step {i * step}", fontsize=11)
        ax.axis("off")
        fp = f"{IMG_DIR}/frames/frame_{i:04d}.png"
        fig.savefig(fp, dpi=80, bbox_inches="tight")
        saved_paths.append(fp)

    plt.close()

    # Stitch GIF with matplotlib's animation
    try:
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.axis("off")
        imgs = []
        for frame in sampled:
            im = ax2.imshow(frame, cmap=cmap, vmin=0, vmax=5,
                            interpolation="nearest", animated=True)
            imgs.append([im])

        ani = animation.ArtistAnimation(fig2, imgs, interval=150, blit=True)
        gif_path = f"{IMG_DIR}/navigation_animation.gif"
        ani.save(gif_path, writer="pillow", fps=8)
        plt.close()
        print(f"  ✓ Saved animation → {gif_path}")
        return gif_path
    except Exception as e:
        print(f"  ⚠ GIF creation skipped: {e}")
        plt.close()
        return saved_paths[0] if saved_paths else ""


# ─────────────────────────────────────────────────────────────
# 7. Decision log chart
# ─────────────────────────────────────────────────────────────
def plot_decision_log(step_log: List[dict]) -> str:
    """
    Pie + bar chart showing breakdown of decision types taken
    throughout the run (follow_plan / replan_follow / reactive).
    """
    from collections import Counter
    counts = Counter(entry["decision"] for entry in step_log)

    labels = list(counts.keys())
    values = list(counts.values())
    colors_pie = ["#0077B6", "#2DC653", "#F77F00", "#E63946", "#9B5DE5"][:len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie
    axes[0].pie(values, labels=labels, colors=colors_pie,
                autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Decision Type Distribution", fontsize=12)

    # Bar — decisions over time
    decisions_over_time = [entry["decision"] for entry in step_log]
    unique_decisions    = list(set(decisions_over_time))
    color_map           = dict(zip(unique_decisions, colors_pie))

    for i, d in enumerate(decisions_over_time):
        axes[1].bar(i, 1, color=color_map[d], alpha=0.8, width=1)

    # Legend
    patches = [mpatches.Patch(color=color_map[d], label=d)
               for d in unique_decisions]
    axes[1].legend(handles=patches, fontsize=8, loc="upper right")
    axes[1].set_title("Decisions Over Time (each bar = 1 step)", fontsize=12)
    axes[1].set_xlabel("Step"); axes[1].set_yticks([])
    axes[1].set_xlim(0, len(step_log))

    plt.tight_layout()
    path_out = f"{IMG_DIR}/decision_log.png"
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path_out}")
    return path_out
