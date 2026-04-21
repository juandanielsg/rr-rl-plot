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



def create_reward_bowtie(
    x: np.ndarray,
    y: np.ndarray,
    goal: np.ndarray,
    error_bias: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Goal reward with a bowtie-shaped iso-reward contour.

    Implements: reward = -(abs(dx) * dy^4/dx^4 + abs(dx/3))

    The bowtie shape narrows to a point at the goal along the x-axis and
    widens along the y-axis, creating two symmetric lobes. Points directly
    above/below the goal are penalised more heavily than points to the side
    at the same distance.

    Args:
        x, y:       Meshgrid (or scalar) robot positions.
        goal:       [gx, gy] goal position.
        error_bias: [bx, by] per-axis scaling applied to the displacement.
        epsilon:    Small constant to avoid division by zero when dx == 0.

    Returns:
        Array of the same shape as x/y with reward values <= 0.
        Zero only at the goal; more negative further away.
    """
    gx, gy = goal

    dx = gx - x
    dy = gy - y

    bx, by = np.asarray(error_bias).ravel()[:2]
    dx_s = dx * bx
    dy_s = dy * by

    reward = -(np.abs(dy_s) * dx_s ** 16 / (dy_s ** 16 + epsilon) + np.abs(dy_s / 3.0))
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
    d_act = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
    threshold = np.linalg.norm([i * radius for i in error_bias])

    condition = d_act >= threshold
    reward = (d_act / np.where(threshold == 0, 1e-9, threshold)) - 1.0
    #return np.where(condition, 0.0, reward)
    return np.where(condition, 0.0, -1.0)


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
    rg = create_reward_bowtie(x, y, goal, error_bias=np.array([1.0, 3.0]))
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
    figsize: Tuple[float, float] = (16, 8),
    error_bias: list = None,
    linear_step: float = 0.2,
    angular_step: float = 0.1,
) -> plt.Figure:
    """
    Plot two interactive heatmaps side-by-side: bowtie (left) and hourglass (right).

    Both subplots share sliders, obstacle position, and goal position. The scene
    is expressed in the car's reference frame (car always at origin). WASD keys
    drive the car: W/S translate the world along the car's forward axis; A/D
    rotate the world around the origin (steering). Left-clicking on either
    heatmap repositions the obstacle.

    Args:
        goal:         (gx, gy) initial goal coordinates in car frame.
        obstacle:     (ox, oy) initial obstacle centre in car frame.
        c1, c2:       Weighting coefficients for goal and obstacle rewards.
        epsilon:      Numerical stability for bowtie reward (dx == 0 guard).
        obs_radius:   Collision radius around the obstacle.
        obs_penalty:  Clip floor for the reward colormap.
        grid_range:   Half-width of the square grid (metres).
        resolution:   Number of grid points per axis (lower = more responsive).
        clip_high:    Upper clip value for the reward colormap.
        figsize:      Matplotlib figure size.
        error_bias:   Per-axis obstacle-ellipse scaling [bx, by].
        linear_step:  Translation per W/S key press (metres).
        angular_step: Rotation per A/D key press (radians).

    Returns:
        The matplotlib Figure object.
    """
    if error_bias is None:
        error_bias = [2.0, 1.0]

    GOAL_BIAS = np.array([1.0, 3.0])

    # ------------------------------------------------------------------ layout
    fig = plt.figure(figsize=figsize)

    # Two heatmap axes + individual colorbars
    ax_l      = fig.add_axes([0.04,  0.40, 0.41,  0.55])
    cbar_ax_l = fig.add_axes([0.455, 0.40, 0.015, 0.55])
    ax_r      = fig.add_axes([0.52,  0.40, 0.41,  0.55])
    cbar_ax_r = fig.add_axes([0.935, 0.40, 0.015, 0.55])

    # Shared slider axes  [left, bottom, width, height]
    ax_c1     = fig.add_axes([0.12, 0.30, 0.35, 0.025])
    ax_c2     = fig.add_axes([0.12, 0.25, 0.35, 0.025])
    ax_radius = fig.add_axes([0.12, 0.20, 0.35, 0.025])
    ax_bx     = fig.add_axes([0.58, 0.30, 0.32, 0.025])
    ax_by     = fig.add_axes([0.58, 0.25, 0.32, 0.025])
    ax_clip   = fig.add_axes([0.58, 0.20, 0.32, 0.025])

    sl_c1     = mwidgets.Slider(ax_c1,     'c1',         0.0,  5.0, valinit=c1,            valstep=0.05)
    sl_c2     = mwidgets.Slider(ax_c2,     'c2',         0.0,  5.0, valinit=c2,            valstep=0.05)
    sl_radius = mwidgets.Slider(ax_radius, 'obs radius', 0.1,  3.0, valinit=obs_radius,    valstep=0.05)
    sl_bx     = mwidgets.Slider(ax_bx,     'bias x',     0.2,  4.0, valinit=error_bias[0], valstep=0.1)
    sl_by     = mwidgets.Slider(ax_by,     'bias y',     0.2,  4.0, valinit=error_bias[1], valstep=0.1)
    sl_clip   = mwidgets.Slider(ax_clip,   'clip high', -5.0,  5.0, valinit=clip_high,     valstep=0.1)

    fig.text(0.12, 0.335, 'Weights & obstacle size',      fontsize=8, color='grey')
    fig.text(0.58, 0.335, 'Ellipse bias & colormap clip', fontsize=8, color='grey')
    fig.text(0.04, 0.115,
             'W/S: drive forward/back  |  A/D: steer left/right  |  '
             'Left-click: reposition obstacle',
             fontsize=8, color='grey', style='italic')

    # --------------------------------------------------------- mutable state
    state = {
        'goal':          list(goal),
        'obstacle':      list(obstacle),
        'theta':         0.0,           # cumulative CCW world rotation (radians)
        'goal_marker_l': None,
        'goal_marker_r': None,
        'obs_ellipse_l': None,
        'obs_marker_l':  None,
        'obs_ellipse_r': None,
        'obs_marker_r':  None,
    }

    # ------------------------------------------------------------------ grid
    lin = np.linspace(-grid_range, grid_range, resolution)
    X, Y = np.meshgrid(lin, lin)

    # --------------------------------------------------- reward computation
    def _compute(reward_fn):
        _c1     = sl_c1.val
        _c2     = sl_c2.val
        _bx     = sl_bx.val
        _by     = sl_by.val
        _radius = sl_radius.val
        _clip   = sl_clip.val

        # Counter-rotate the grid by -theta so the reward shape (hourglass /
        # bowtie) stays aligned with the car's heading rather than the world axes.
        # R(-theta): x' = cos(t)*x + sin(t)*y,  y' = -sin(t)*x + cos(t)*y
        t = state['theta']
        c, s = np.cos(t), np.sin(t)
        X_r =  c * X + s * Y
        Y_r = -s * X + c * Y
        gx, gy = state['goal']
        goal_r = (c * gx + s * gy, -s * gx + c * gy)

        rg = reward_fn(X_r, Y_r, goal_r, error_bias=GOAL_BIAS)
        ro = reward_obstacle_reverse_ellipse(X, Y, tuple(state['obstacle']),
                                             error_bias=[_bx, _by], radius=_radius)
        R  = _c1 * rg + _c2 * ro
        _lo = obs_penalty
        _hi = _clip if _clip > _lo else _lo + 1e-3
        return np.clip(R, _lo, _hi)

    # ---------------------------------------- per-axes overlay update
    def _update_overlay(ax, ell_key, obs_mkr_key):
        _bx     = sl_bx.val
        _by     = sl_by.val
        _radius = sl_radius.val
        _obs    = state['obstacle']

        if state[ell_key] is not None:
            state[ell_key].remove()
        thr = np.linalg.norm([_bx * _radius, _by * _radius])
        ell = Ellipse(
            xy=_obs,
            width=2 * thr / _bx,
            height=2 * thr / _by,
            angle=0,
            edgecolor='none', facecolor='none',
            linewidth=1.5, linestyle='--', zorder=5,
        )
        ax.add_patch(ell)
        state[ell_key] = ell
        state[obs_mkr_key].set_data([_obs[0]], [_obs[1]])

    # --------------------------------------------------------- redraw both
    def _redraw():
        _c1     = sl_c1.val
        _c2     = sl_c2.val
        _bx     = sl_bx.val
        _by     = sl_by.val
        _radius = sl_radius.val
        _obs    = state['obstacle']
        _goal   = state['goal']

        for ax_, im_, ell_key, obs_mkr_key, goal_mkr_key, reward_fn, label in [
            (ax_l, im_l, 'obs_ellipse_l', 'obs_marker_l', 'goal_marker_l', create_reward_bowtie, 'Bowtie'),
            (ax_r, im_r, 'obs_ellipse_r', 'obs_marker_r', 'goal_marker_r', get_reward_hourglass, 'Hourglass'),
        ]:
            R = _compute(reward_fn)
            im_.set_data(R)
            im_.set_clim(R.min(), R.max())
            _update_overlay(ax_, ell_key, obs_mkr_key)
            state[goal_mkr_key].set_data([_goal[0]], [_goal[1]])
            ax_.set_title(
                f"{label}  |  r = {_c1:.2f}·rg + {_c2:.2f}·ro\n"
                f"goal=({_goal[0]:.2f},{_goal[1]:.2f})  |  "
                f"obs=({_obs[0]:.2f},{_obs[1]:.2f})  |  "
                f"r={_radius:.2f}  bias=[{_bx:.1f},{_by:.1f}]",
                fontsize=9,
            )
        fig.canvas.draw_idle()

    # --------------------------------------------------------- initial render
    _lo0 = obs_penalty
    _hi0 = clip_high if clip_high > _lo0 else _lo0 + 1e-3

    def _init_R(reward_fn):
        rg = reward_fn(X, Y, tuple(state['goal']), error_bias=GOAL_BIAS)
        ro = reward_obstacle_reverse_ellipse(X, Y, tuple(state['obstacle']),
                                             error_bias=error_bias, radius=obs_radius)
        return np.clip(c1 * rg + c2 * ro, _lo0, _hi0)

    R0_l = _init_R(create_reward_bowtie)
    R0_r = _init_R(get_reward_hourglass)

    im_l = ax_l.imshow(
        R0_l,
        extent=[-grid_range, grid_range, -grid_range, grid_range],
        origin='lower', cmap='jet', interpolation='bilinear',
        vmin=R0_l.min(), vmax=R0_l.max(),
    )
    im_r = ax_r.imshow(
        R0_r,
        extent=[-grid_range, grid_range, -grid_range, grid_range],
        origin='lower', cmap='jet', interpolation='bilinear',
        vmin=R0_r.min(), vmax=R0_r.max(),
    )

    cbar_l = fig.colorbar(im_l, cax=cbar_ax_l)
    cbar_l.set_label('Reward (clipped)', fontsize=8)
    cbar_r = fig.colorbar(im_r, cax=cbar_ax_r)
    cbar_r.set_label('Reward (clipped)', fontsize=8)

    marker_custom = CustomMarkers()

    thr0    = np.linalg.norm([error_bias[0] * obs_radius, error_bias[1] * obs_radius])
    half_x0 = thr0 / error_bias[0]
    half_y0 = thr0 / error_bias[1]

    for ax_, ell_key, obs_mkr_key, goal_mkr_key in [
        (ax_l, 'obs_ellipse_l', 'obs_marker_l', 'goal_marker_l'),
        (ax_r, 'obs_ellipse_r', 'obs_marker_r', 'goal_marker_r'),
    ]:
        obs_m, = ax_.plot(
            *state['obstacle'], 'wo',
            markersize=10, markeredgecolor='black', label='Obstacle', zorder=6,
        )
        goal_m, = ax_.plot(
            *state['goal'], 'wX',
            markersize=14, markeredgecolor='black', markerfacecolor='white',
            label='Goal', zorder=6,
        )
        ax_.plot(
            0, 0,
            marker=marker_custom.rr100_marker,
            markersize=20, markeredgecolor='black',
            label='Robot (origin)', zorder=6,
        )
        state[obs_mkr_key]  = obs_m
        state[goal_mkr_key] = goal_m

        ell0 = Ellipse(
            xy=state['obstacle'],
            width=2 * half_x0, height=2 * half_y0,
            angle=0, edgecolor='none', facecolor='none',
            linewidth=1.5, linestyle='--', zorder=5,
        )
        ax_.add_patch(ell0)
        state[ell_key] = ell0

        ax_.set_xlim(-grid_range, grid_range)
        ax_.set_ylim(-grid_range, grid_range)
        ax_.set_xlabel('x (m)', fontsize=11)
        ax_.set_ylabel('y (m)', fontsize=11)
        ax_.set_aspect('equal')
        ax_.legend(loc='upper left', fontsize=9)

    ax_l.set_title(
        f"Bowtie  |  r = {c1:.2f}·rg + {c2:.2f}·ro\n"
        f"goal={goal}  |  obstacle={obstacle}  |  radius={obs_radius:.2f}  |  bias={error_bias}",
        fontsize=9,
    )
    ax_r.set_title(
        f"Hourglass  |  r = {c1:.2f}·rg + {c2:.2f}·ro\n"
        f"goal={goal}  |  obstacle={obstacle}  |  radius={obs_radius:.2f}  |  bias={error_bias}",
        fontsize=9,
    )

    # -------------------------------------------- slider callbacks
    for sl in (sl_c1, sl_c2, sl_radius, sl_bx, sl_by, sl_clip):
        sl.on_changed(lambda _: _redraw())

    # -------------------------------------------- click callback (both axes)
    def _on_click(event):
        if event.inaxes not in (ax_l, ax_r) or event.button != 1:
            return
        state['obstacle'] = [event.xdata, event.ydata]
        _redraw()

    # -------------------------------------------- keyboard callback (WASD)
    def _rotate(pos, angle):
        c, s = np.cos(angle), np.sin(angle)
        x, y = pos
        return [c * x - s * y, s * x + c * y]

    def _on_key(event):
        key = event.key
        if key == 'w':
            state['goal'][1]     -= linear_step
            state['obstacle'][1] -= linear_step
        elif key == 'x':
            state['goal'][1]     += linear_step
            state['obstacle'][1] += linear_step
        elif key == 'a':
            # car turns left (CCW) → world rotates CW = rotate by -angular_step
            state['theta']   -= angular_step
            state['goal']     = _rotate(state['goal'],     -angular_step)
            state['obstacle'] = _rotate(state['obstacle'], -angular_step)
        elif key == 'd':
            # car turns right (CW) → world rotates CCW = rotate by +angular_step
            state['theta']   += angular_step
            state['goal']     = _rotate(state['goal'],     angular_step)
            state['obstacle'] = _rotate(state['obstacle'], angular_step)
        else:
            return
        _redraw()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event',    _on_key)

    plt.tight_layout(rect=[0, 0.13, 1, 1])
    return fig


if __name__ == "__main__":
    fig = plot_reward_heatmap(
        goal=(-0.0, 2.0),
        obstacle=(-0.8, 0.0),
        c1=1.0,
        c2=5.0,
        obs_radius=0.25,
        obs_penalty=-5.0,
        clip_high=0.0,
        figsize=(16, 8),
        error_bias=[2.0, 1.0],
    )
    plt.show()
