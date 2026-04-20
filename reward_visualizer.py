import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from my_custom_markers import CustomMarkers

 
def reward_goal(
    x: np.ndarray,
    y: np.ndarray,
    goal: Tuple[float, float],
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Inverse-distance reward toward the goal.
    reward_goal = 1 / (d(robot, goal) + epsilon)
 
    Args:
        x, y:    Meshgrid arrays of robot positions relative to origin (robot = 0,0).
        goal:    (gx, gy) goal position in the same frame.
        epsilon: Small constant to avoid division by zero when robot is on the goal.
 
    Returns:
        Array of the same shape as x/y with reward values in (0, 1/epsilon].
    """
    gx, gy = goal
    dist = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
    return -dist
 
 
def reward_obstacle(
    x: np.ndarray,
    y: np.ndarray,
    obstacle: Tuple[float, float],
    radius: float = 1.0,
    penalty: float = -10.0,
) -> np.ndarray:
    """
    Hard-boundary penalty: zero outside the obstacle radius, `penalty` inside.
 
    Args:
        x, y:     Meshgrid arrays of robot positions relative to origin.
        obstacle: (ox, oy) obstacle centre position.
        radius:   Collision radius around the obstacle.
        penalty:  Reward value assigned inside the radius (should be negative).
 
    Returns:
        Array of the same shape as x/y with 0.0 outside and `penalty` inside.
    """
    ox, oy = obstacle
    dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
    return np.where(dist <= radius, penalty, 0.0)


def reward_obstacle_reverse_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    obstacle: Tuple[float, float],
    radius: float = 1.0,
    error_bias: list = None,           # Fix 3: avoid mutable default
) -> np.ndarray:
    """
    Soft penalty: zero outside the obstacle radius, negative gradient inside.
    Args:
        x, y:        Meshgrid arrays of robot positions relative to origin.
        obstacle:    (ox, oy) obstacle centre position.
        radius:      Collision radius around the obstacle.
        error_bias:  Per-axis scaling factors [sx, sy] for elliptical distance.
    Returns:
        Array of the same shape as x/y with 0.0 outside and a negative
        gradient value inside (reaches -1.0 at the obstacle centre).
    """
    if error_bias is None:
        error_bias = [1.0, 1.0]           # Fix 3: safe default

    ox, oy = obstacle
    diff = np.array([(ox - x) * error_bias[0],
                     (oy - y) * error_bias[1]])  # Fix 2: explicit per-element scaling

    d_act = np.linalg.norm(diff, axis=0)          # Fix 2: per-element norm over axis 0

    threshold = np.linalg.norm([i * radius for i in error_bias])  # scalar — correct

    condition = d_act >= threshold
    reward = (d_act / np.where(threshold == 0, 1e-9, threshold)) - 1.0

    return np.where(condition, 0.0, reward)        # Fix 1: now correctly indented




 
def combined_reward(
    x: np.ndarray,
    y: np.ndarray,
    goal: Tuple[float, float],
    obstacle: Tuple[float, float],
    c1: float = 1.0,
    c2: float = 1.0,
    epsilon: float = 1e-6,
    obs_radius: float = 1.0,
    obs_penalty: float = -10.0,
) -> np.ndarray:
    """
    r = c1 * reward_goal + c2 * reward_obstacle
    """
    rg = reward_goal(x, y, goal, epsilon=epsilon)
    #ro = reward_obstacle(x, y, obstacle, radius=obs_radius, penalty=obs_penalty)
    ro = reward_obstacle_reverse_ellipse(x, y, obstacle, error_bias=[2.0,1.0], radius=0.5)
    return c1 * rg + c2 * ro
 
 
def plot_reward_heatmap(
    goal: Tuple[float, float] = (3.0, 2.0),
    obstacle: Tuple[float, float] = (-2.0, 1.0),
    c1: float = 1.0,
    c2: float = 1.0,
    epsilon: float = 1e-6,
    obs_radius: float = 1.0,
    obs_penalty: float = -10.0,
    grid_range: float = 6.0,
    resolution: int = 1000,
    clip_high: float = 0.0,
    figsize: Tuple[float, float] = (7, 6),
) -> plt.Figure:
    """
    Plot a heatmap of the combined reward landscape.
 
    The robot sits at the origin (0, 0). Every point (x, y) on the grid
    represents a hypothetical robot position; the goal and obstacle are fixed
    in this frame.
 
    Args:
        goal:        (gx, gy) goal coordinates.
        obstacle:    (ox, oy) obstacle centre coordinates.
        c1, c2:      Weighting coefficients.
        epsilon:     Numerical stability for inverse-distance reward.
        obs_radius:  Hard-penalty radius around the obstacle.
        obs_penalty: Reward value inside the obstacle (negative).
        grid_range:  Half-width of the square grid (metres).
        resolution:  Number of grid points per axis.
        clip_high:   Upper clip value for the reward (prevents the 1/r spike
                     near the goal from washing out the rest of the colormap).
        figsize:     Matplotlib figure size.
 
    Returns:
        The matplotlib Figure object.
    """
    lin = np.linspace(-grid_range, grid_range, resolution)
    x, y = np.meshgrid(lin, lin)
 
    r = combined_reward(
        x, y,
        goal=goal,
        obstacle=obstacle,
        c1=c1,
        c2=c2,
        epsilon=epsilon,
        obs_radius=obs_radius,
        obs_penalty=obs_penalty,
    )
 
    # Clip to keep the colormap readable (the 1/r peak is very sharp)
    r_clipped = np.clip(r, obs_penalty, clip_high)
 
    fig, ax = plt.subplots(figsize=figsize)
 
    im = ax.imshow(
        r_clipped,
        extent=[-grid_range, grid_range, -grid_range, grid_range],
        origin="lower",
        cmap="plasma",
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Combined reward (clipped)", fontsize=11)

    marker_custom = CustomMarkers()
 
    # Markers
    ax.plot(*obstacle, "ko", markersize=10, label=f"Obstacle {obstacle}")
    ax.plot(*goal,     "kX", markersize=14, label=f"Goal {goal}", markeredgecolor="black", markerfacecolor="black")
    ax.plot(0, 0, marker=marker_custom.rr100_marker, markersize=20, markeredgecolor="black", label="Robot (origin)")
 
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.set_title(
        f"Reward heatmap  |  r = {c1}·reward_goal + {c2}·reward_obstacle\n"
        f"goal={goal} | obstacle={obstacle} | obs_radius={obs_radius}",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig
 
 
if __name__ == "__main__":
    fig = plot_reward_heatmap(
        goal=(-1.6, 0.5),
        obstacle=(-0.8, 0.0),
        c1=1.0,
        c2=5.0,
        obs_radius=0.25,
        obs_penalty=-5.0,
        clip_high=0.0,
    )
    #plt.savefig("/mnt/user-data/outputs/reward_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()