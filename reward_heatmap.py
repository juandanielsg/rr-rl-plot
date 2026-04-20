import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from matplotlib.patches import Ellipse
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

    Args:
        x, y:    Meshgrid arrays of robot positions relative to origin.
        goal:    (gx, gy) goal position in the same frame.
        epsilon: Small constant to avoid division by zero when robot is on the goal.

    Returns:
        Array of the same shape as x/y with reward values.
    """
    gx, gy = goal
    dist = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
    return -dist

def get_reward_hourglass(
    x: np.ndarray,
    y: np.ndarray,
    goal: np.ndarray,
    error_bias: np.ndarray,
    squeeze: float = 1.0,
) -> np.ndarray:
    """
    Goal reward with an hourglass-shaped iso-reward contour.

    The hourglass effect squeezes the y-displacement toward zero whenever
    |dx| > |dy| (i.e. the robot is more to the side of the goal than above
    or below it). This narrows the reward field laterally, encouraging the
    robot to approach the goal along the vertical axis.

    Args:
        x, y:       Meshgrid (or scalar) robot positions.
        goal:       [gx, gy] goal position.
        error_bias: [bx, by] per-axis scaling applied after the warp.
        squeeze:    Strength of the hourglass squeeze (default 1.0).

    Returns:
        Array of the same shape as x/y with reward values <= 0.
        Zero only at the goal; more negative further away.
    """
    gx, gy = goal

    # Displacement from robot to goal
    dx = gx - x
    dy = gy - y

    # Hourglass warp: when the robot is wider than it is tall relative to
    # the goal (|dx| > |dy|), pinch dy toward zero proportionally.
    delta = np.minimum(np.abs(dy) - np.abs(dx), 0.0)   # <= 0 when |dx| > |dy|
    dx_warped = dx - np.sign(dx) * delta * squeeze       # squeeze y inward

    # Stack into (2, ...) for vectorised norm, then apply per-axis bias
    diff = np.array([dy, dx_warped]) * np.asarray(error_bias).reshape(2, *([1] * np.ndim(y)))

    reward = -np.linalg.norm(diff, axis=0)
    return reward



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
    error_bias: list = None,
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
        error_bias = [1.0, 1.0]

    ox, oy = obstacle
    diff = np.array([
        (ox - x) * error_bias[0],
        (oy - y) * error_bias[1],
    ])

    #d_act = np.linalg.norm(diff, axis=0)
    d_act = np.sqrt(diff[0] ** 4 + diff[1] ** 4)
    threshold = np.linalg.norm([i * radius for i in error_bias])

    condition = d_act >= threshold
    reward = (d_act / np.where(threshold == 0, 1e-9, threshold)) - 1.0
    return np.where(condition, 0.0, reward)


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
    error_bias: list = None,
) -> np.ndarray:
    """
    r = c1 * reward_goal + c2 * reward_obstacle
    """
    if error_bias is None:
        error_bias = [2.0, 1.0]

    #rg = reward_goal(x, y, goal, epsilon=epsilon)
    rg = get_reward_hourglass(x, y, goal, error_bias=np.array([1.0, 3.0]))
    ro = reward_obstacle_reverse_ellipse(x, y, obstacle, error_bias=error_bias, radius=obs_radius)
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
    resolution: int = 300,
    clip_high: float = 0.0,
    figsize: Tuple[float, float] = (9, 8),
    error_bias: list = None,
) -> plt.Figure:
    """
    Plot an interactive heatmap of the combined reward landscape.

    The robot sits at the origin (0, 0). Every point (x, y) on the grid
    represents a hypothetical robot position; the goal and obstacle are fixed
    in this frame.

    Interactive controls:
        - Sliders for c1, c2, obs_radius, bias x, bias y, clip high.
        - Left-click on the heatmap to reposition the obstacle.

    Args:
        goal:        (gx, gy) goal coordinates.
        obstacle:    (ox, oy) obstacle centre coordinates.
        c1, c2:      Weighting coefficients.
        epsilon:     Numerical stability for inverse-distance reward.
        obs_radius:  Collision radius around the obstacle.
        obs_penalty: Reward value inside the obstacle (negative).
        grid_range:  Half-width of the square grid (metres).
        resolution:  Number of grid points per axis (lower = more responsive).
        clip_high:   Upper clip value for the reward.
        figsize:     Matplotlib figure size.
        error_bias:  Per-axis ellipse scaling [bx, by].

    Returns:
        The matplotlib Figure object.
    """
    if error_bias is None:
        error_bias = [2.0, 1.0]

    # ------------------------------------------------------------------ layout
    fig = plt.figure(figsize=figsize)

    # Main axes and colourbar
    ax      = fig.add_axes([0.08, 0.40, 0.80, 0.55])
    cbar_ax = fig.add_axes([0.90, 0.40, 0.02, 0.55])

    # Slider axes  [left, bottom, width, height]
    ax_c1     = fig.add_axes([0.12, 0.30, 0.35, 0.025])
    ax_c2     = fig.add_axes([0.12, 0.25, 0.35, 0.025])
    ax_radius = fig.add_axes([0.12, 0.20, 0.35, 0.025])
    ax_bx     = fig.add_axes([0.58, 0.30, 0.32, 0.025])
    ax_by     = fig.add_axes([0.58, 0.25, 0.32, 0.025])
    ax_clip   = fig.add_axes([0.58, 0.20, 0.32, 0.025])

    sl_c1     = mwidgets.Slider(ax_c1,     'c1',         0.0,  5.0,  valinit=c1,               valstep=0.05)
    sl_c2     = mwidgets.Slider(ax_c2,     'c2',         0.0,  5.0,  valinit=c2,               valstep=0.05)
    sl_radius = mwidgets.Slider(ax_radius, 'obs radius', 0.1,  3.0,  valinit=obs_radius,       valstep=0.05)
    sl_bx     = mwidgets.Slider(ax_bx,     'bias x',     0.2,  4.0,  valinit=error_bias[0],    valstep=0.1)
    sl_by     = mwidgets.Slider(ax_by,     'bias y',     0.2,  4.0,  valinit=error_bias[1],    valstep=0.1)
    sl_clip   = mwidgets.Slider(ax_clip,   'clip high', -5.0,  5.0,  valinit=clip_high,        valstep=0.1)

    # Section labels
    fig.text(0.12, 0.335, 'Weights & obstacle size', fontsize=8, color='grey')
    fig.text(0.58, 0.335, 'Ellipse bias & colormap clip', fontsize=8, color='grey')
    fig.text(0.08, 0.115,
             'Tip: left-click on the heatmap to reposition the obstacle.',
             fontsize=8, color='grey', style='italic')

    # --------------------------------------------------------- mutable state
    state = {
        'obstacle':    list(obstacle),
        'obs_ellipse': None,
        'obs_marker':  None,
    }

    # ------------------------------------------------------------------ grid
    lin = np.linspace(-grid_range, grid_range, resolution)
    X, Y = np.meshgrid(lin, lin)

    # --------------------------------------------------------- draw function
    def _redraw():
        _c1     = sl_c1.val
        _c2     = sl_c2.val
        _radius = sl_radius.val
        _bx     = sl_bx.val
        _by     = sl_by.val
        _clip   = sl_clip.val
        _obs    = state['obstacle']

        R = combined_reward(
            X, Y,
            goal=goal,
            obstacle=tuple(_obs),
            c1=_c1,
            c2=_c2,
            epsilon=epsilon,
            obs_radius=_radius,
            obs_penalty=obs_penalty,
            error_bias=[_bx, _by],
        )
        # Ensure clip bounds are valid
        _lo = obs_penalty
        _hi = _clip if _clip > _lo else _lo + 1e-3
        R_clipped = np.clip(R, _lo, _hi)

        ax.images[0].set_data(R_clipped)
        ax.images[0].set_clim(R_clipped.min(), R_clipped.max())

        # Rebuild obstacle ellipse
        if state['obs_ellipse'] is not None:
            state['obs_ellipse'].remove()

        # Convert bias + radius to half-axes in data units:
        #   threshold = norm([bx*r, by*r])  →  half-axis_x = threshold/bx, etc.
        thr = np.linalg.norm([_bx * _radius, _by * _radius])
        half_x = thr / _bx
        half_y = thr / _by
        ell = Ellipse(
            xy=_obs,
            width=2 * half_x,
            height=2 * half_y,
            angle=0,
            edgecolor='none',
            facecolor='none',
            linewidth=1.5,
            linestyle='--',
            zorder=5,
        )
        ax.add_patch(ell)
        state['obs_ellipse'] = ell

        # Move obstacle marker
        state['obs_marker'].set_data([_obs[0]], [_obs[1]])

        ax.set_title(
            f"Reward heatmap  |  r = {_c1:.2f}·reward_goal + {_c2:.2f}·reward_obstacle\n"
            f"goal={goal}  |  obstacle=({_obs[0]:.2f}, {_obs[1]:.2f})  |  "
            f"radius={_radius:.2f}  |  bias=[{_bx:.1f}, {_by:.1f}]",
            fontsize=10,
        )
        fig.canvas.draw_idle()

    # --------------------------------------------------------- initial render
    R0 = combined_reward(
        X, Y,
        goal=goal,
        obstacle=tuple(state['obstacle']),
        c1=c1, c2=c2,
        epsilon=epsilon,
        obs_radius=obs_radius,
        obs_penalty=obs_penalty,
        error_bias=error_bias,
    )
    _lo0 = obs_penalty
    _hi0 = clip_high if clip_high > _lo0 else _lo0 + 1e-3
    R0c = np.clip(R0, _lo0, _hi0)

    im = ax.imshow(
        R0c,
        extent=[-grid_range, grid_range, -grid_range, grid_range],
        origin='lower',
        cmap='jet',
        interpolation='bilinear',
        vmin=R0c.min(),
        vmax=R0c.max(),
    )
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Combined reward (clipped)', fontsize=9)

    marker_custom = CustomMarkers()

    obs_marker, = ax.plot(
        *state['obstacle'], 'wo',
        markersize=10, markeredgecolor='black',
        label=f'Obstacle', zorder=6,
    )
    ax.plot(
        *goal, 'wX',
        markersize=14, markeredgecolor='black', markerfacecolor='white',
        label=f'Goal {goal}', zorder=6,
    )
    ax.plot(
        0, 0,
        marker=marker_custom.rr100_marker,
        markersize=20, markeredgecolor='black',
        label='Robot (origin)', zorder=6,
    )

    state['obs_marker'] = obs_marker

    # Draw initial ellipse
    thr0   = np.linalg.norm([error_bias[0] * obs_radius, error_bias[1] * obs_radius])
    half_x0 = thr0 / error_bias[0]
    half_y0 = thr0 / error_bias[1]
    ell0 = Ellipse(
        xy=state['obstacle'],
        width=2 * half_x0,
        height=2 * half_y0,
        angle=0,
        edgecolor='none', facecolor='none',
        linewidth=1.5, linestyle='--', zorder=5,
    )
    ax.add_patch(ell0)
    state['obs_ellipse'] = ell0

    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(
        f"Reward heatmap  |  r = {c1:.2f}·reward_goal + {c2:.2f}·reward_obstacle\n"
        f"goal={goal}  |  obstacle={obstacle}  |  "
        f"radius={obs_radius:.2f}  |  bias={error_bias}",
        fontsize=10,
    )

    # -------------------------------------------- slider callbacks
    for sl in (sl_c1, sl_c2, sl_radius, sl_bx, sl_by, sl_clip):
        sl.on_changed(lambda _: _redraw())

    # -------------------------------------------- click callback
    def _on_click(event):
        # Only respond to left-clicks inside the heatmap axes
        if event.inaxes is not ax or event.button != 1:
            return
        state['obstacle'] = [event.xdata, event.ydata]
        _redraw()

    fig.canvas.mpl_connect('button_press_event', _on_click)

    plt.tight_layout(rect=[0, 0.13, 1, 1])
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
        error_bias=[2.0, 1.0],
    )
    plt.show()
