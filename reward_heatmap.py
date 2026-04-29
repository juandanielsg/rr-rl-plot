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

    # Hourglass warp: when the robot is more lateral than forward relative to
    # the goal (|dy| > |dx|), pinch dy toward zero proportionally.
    delta    = np.minimum(np.abs(dx) - np.abs(dy), 0.0)  # <= 0 when |dy| > |dx|
    dy_warped = dy - np.sign(dy) * delta * squeeze        # squeeze lateral inward

    # Stack into (2, ...) for vectorised norm, then apply per-axis bias
    # bias[0] scales forward (dx), bias[1] scales lateral (dy_warped)
    diff = np.array([dx, dy_warped]) * np.asarray(error_bias).reshape(2, *([1] * np.ndim(y)))

    reward = -np.linalg.norm(diff, axis=0)
    return reward



def create_reward_bowtie_old(
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

    first_term  = np.sqrt(dx**2 / 3 + epsilon) - epsilon**2
    second_term = 3 * np.maximum(0, np.abs(dy) - np.abs(dx))
    reward = -(first_term + second_term)
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
    error_shape: list = None,
    exponents: list = [0.8, 0.8],
    penalty: float = None,
) -> np.ndarray:
    """
    Soft (or hard) penalty inside an axis-scaled super-ellipse around the obstacle.

    Two independent parameter sets mirror the RL environment:
      - error_shape  defines the boundary condition (which ellipse triggers the penalty).
      - error_bias   defines the reward gradient inside (how steeply it ramps to -1).
    Decoupling them lets you widen the "danger zone" without changing the gradient
    profile, or vice-versa.  When error_shape is omitted it defaults to error_bias,
    recovering the original single-parameter behaviour.

    Args:
        x, y:        Meshgrid arrays of robot positions relative to origin.
        obstacle:    (ox, oy) obstacle centre position.
        radius:      Collision / influence radius (obstacle_distance_threshold).
        error_bias:  Per-axis reward-gradient scaling [s0, s1].
        error_shape: Per-axis boundary scaling [s0, s1]; defaults to error_bias.
        exponents:   Per-axis super-ellipse exponents [e0, e1]; default [2, 2].
        penalty:     If set, return this flat value inside instead of a gradient.

    Returns:
        Array of the same shape as x/y: 0.0 outside, gradient (or flat penalty) inside.
    """
    if error_bias is None:
        error_bias = [1.0, 1.0]
    if error_shape is None:
        error_shape = error_bias
    if exponents is None:
        exponents = [2, 2]

    ox, oy = obstacle
    dx = np.abs(ox - x)
    dy = np.abs(oy - y)
    ex, ey = exponents

    # Boundary condition — determined by error_shape
    d_act     = np.sqrt((error_shape[0] * dx) ** ex + (error_shape[1] * dy) ** ey)
    threshold = np.sqrt((error_shape[0] * radius) ** ex + (error_shape[1] * radius) ** ey)

    condition = d_act >= threshold

    if penalty is not None:
        return np.where(condition, 0.0, penalty)

    # Gradient reward — determined by error_bias
    rew_act   = np.sqrt((error_bias[0] * dx) ** ex + (error_bias[1] * dy) ** ey)
    rew_thresh = np.sqrt((error_bias[0] * radius) ** ex + (error_bias[1] * radius) ** ey)

    reward = np.clip(rew_act / np.where(rew_thresh == 0, 1e-9, rew_thresh) - 1.0, -1.0, 0.0)
    return np.where(condition, 0.0, reward)


def reward_obstacle_circulation(
    x: np.ndarray,
    y: np.ndarray,
    obstacle: Tuple[float, float],
    goal: Tuple[float, float],
    radius: float = 1.0,
    influence: float = 2.0,
    sigma: float = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Breaks local minima by penalising positions where moving away from the
    obstacle conflicts with moving toward the goal.

    Signal: (dot(outward_from_obs, toward_goal) - 1) / 2, in [-1, 0].
      -1  robot is directly behind the obstacle (outward opposes goal direction).
       0  robot has cleared to the goal side (outward aligns with goal direction).

    Weight: logistic soft-start at the obstacle surface multiplied by a
    Gaussian outward decay — fully smooth and differentiable, no hard edges.
    The Gaussian naturally reaches near-zero at `influence`, so no outer mask
    is needed (and none is applied, avoiding a gradient discontinuity there).

    Args:
        x, y:      Meshgrid arrays of robot positions.
        obstacle:  (ox, oy) obstacle centre.
        goal:      (gx, gy) goal position.
        radius:    Obstacle hard radius (weight → 0 inside).
        influence: Radial reach; sets the default Gaussian sigma.
        sigma:     Explicit Gaussian width; defaults to half the annulus width.
        eps:       Numerical stability constant.

    Returns:
        Array of values in [-1, 0].
    """
    ox, oy = obstacle
    gx, gy = goal
    dx, dy = x - ox, y - oy
    d_obs = np.sqrt(dx**2 + dy**2) + eps

    # Outward unit vector: obstacle → robot
    out_x, out_y = dx / d_obs, dy / d_obs

    # Goal unit vector: robot → goal
    d_goal = np.sqrt((x - gx)**2 + (y - gy)**2) + eps
    goal_x, goal_y = (gx - x) / d_goal, (gy - y) / d_goal

    # Cosine similarity in [-1, 1], mapped to [-1, 0]
    alignment   = out_x * goal_x + out_y * goal_y
    circ_signal = (alignment - 1.0) / 2.0

    # Logistic soft-start at the surface × Gaussian outward decay.
    # The Gaussian tail replaces any hard outer cutoff.
    _sigma       = sigma if sigma is not None else max((influence - radius) * 0.5, 0.1)
    surface_dist = d_obs - radius
    inner_blend  = 1.0 / (1.0 + np.exp(-surface_dist / (_sigma * 0.3)))
    outer_decay  = np.exp(-np.maximum(surface_dist, 0.0)**2 / (2.0 * _sigma**2))

    return circ_signal * inner_blend * outer_decay


def combined_reward(
    x: np.ndarray,
    y: np.ndarray,
    goal: Tuple[float, float],
    obstacle: Tuple[float, float],
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 1.0,
    epsilon: float = 1e-6,
    obs_radius: float = 1.0,
    obs_penalty: float = -10.0,
    obs_influence: float = 2.0,
    error_bias: list = None,
) -> np.ndarray:
    """
    r = c1 * reward_goal + c2 * reward_obstacle + c3 * reward_circulation
    """
    if error_bias is None:
        error_bias = [2.0, 1.0]

    rg = create_reward_bowtie(x, y, goal, error_bias=np.array([1.0, 3.0]))
    ro = reward_obstacle_reverse_ellipse(x, y, obstacle, error_bias=error_bias, radius=obs_radius)
    rc = reward_obstacle_circulation(x, y, obstacle, goal, radius=obs_radius, influence=obs_influence)
    return c1 * rg + c2 * ro + c3 * rc


def plot_reward_heatmap(
    goal: Tuple[float, float] = (3.0, 2.0),
    obstacle: Tuple[float, float] = (-2.0, 1.0),
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 1.0,
    epsilon: float = 1e-6,
    obs_radius: float = 1.0,
    obs_influence: float = 2.0,
    obs_penalty: float = -10.0,
    grid_range: float = 6.0,
    resolution: int = 300,
    clip_high: float = 0.0,
    figsize: Tuple[float, float] = (16, 8),
    error_shape: list = None,
    error_bias: list = None,
    linear_step: float = 0.2,
    angular_step: float = 0.1,
) -> plt.Figure:
    """
    Plot two interactive heatmaps side-by-side: bowtie (left) and hourglass (right).

    Both subplots share sliders, obstacle position, and goal position. The scene
    is expressed in the car's reference frame (car always at origin). W/X keys
    translate the world along the car's forward axis; A/D rotate the world
    around the origin (steering). Left-clicking on either
    heatmap repositions the obstacle.

    Args:
        goal:         (gx, gy) initial goal coordinates in car frame.
        obstacle:     (ox, oy) initial obstacle centre in car frame.
        c1, c2, c3:   Weighting coefficients for goal, obstacle, and circulation rewards.
        epsilon:      Numerical stability for bowtie reward (dx == 0 guard).
        obs_radius:   Collision radius around the obstacle.
        obs_influence: Radial reach of the circulation reward beyond the obstacle surface.
        obs_penalty:  Clip floor for the reward colormap.
        grid_range:   Half-width of the square grid (metres).
        resolution:   Number of grid points per axis (lower = more responsive).
        clip_high:    Upper clip value for the reward colormap.
        figsize:      Matplotlib figure size.
        error_bias:   Per-axis obstacle-ellipse scaling [bx, by].
        linear_step:  Translation per W/X key press (metres).
        angular_step: Rotation per A/D key press (radians).

    Returns:
        The matplotlib Figure object.
    """
    if error_bias is None:
        error_bias = [2.0, 1.0]
    if error_shape is None:
        error_shape = list(error_bias)

    GOAL_BIAS = np.array([1.0, 2.0])

    # ------------------------------------------------------------------ layout
    fig = plt.figure(figsize=figsize)

    # Two heatmap axes + individual colorbars
    ax_l      = fig.add_axes([0.04,  0.40, 0.41,  0.55])
    cbar_ax_l = fig.add_axes([0.455, 0.40, 0.015, 0.55])
    ax_r      = fig.add_axes([0.52,  0.40, 0.41,  0.55])
    cbar_ax_r = fig.add_axes([0.935, 0.40, 0.015, 0.55])

    # Shared slider axes  [left, bottom, width, height]
    ax_c1        = fig.add_axes([0.12, 0.30, 0.35, 0.025])
    ax_c2        = fig.add_axes([0.12, 0.25, 0.35, 0.025])
    ax_c3        = fig.add_axes([0.12, 0.20, 0.35, 0.025])
    ax_radius    = fig.add_axes([0.12, 0.15, 0.35, 0.025])
    ax_sx        = fig.add_axes([0.58, 0.30, 0.32, 0.025])   # error_shape[0]
    ax_sy        = fig.add_axes([0.58, 0.25, 0.32, 0.025])   # error_shape[1]
    ax_influence = fig.add_axes([0.58, 0.20, 0.32, 0.025])
    ax_clip      = fig.add_axes([0.58, 0.15, 0.32, 0.025])
    ax_ebx       = fig.add_axes([0.12, 0.10, 0.35, 0.025])   # error_bias[0]
    ax_eby       = fig.add_axes([0.58, 0.10, 0.32, 0.025])   # error_bias[1]
    ax_toggle    = fig.add_axes([0.12, 0.05, 0.22, 0.038])   # bias-ellipse toggle

    sl_c1        = mwidgets.Slider(ax_c1,        'c1',           0.0,  5.0, valinit=c1,              valstep=0.05)
    sl_c2        = mwidgets.Slider(ax_c2,        'c2',           0.0,  5.0, valinit=c2,              valstep=0.05)
    sl_c3        = mwidgets.Slider(ax_c3,        'c3',           0.0,  5.0, valinit=c3,              valstep=0.05)
    sl_radius    = mwidgets.Slider(ax_radius,    'obs radius',   0.1,  3.0, valinit=obs_radius,      valstep=0.05)
    sl_sx        = mwidgets.Slider(ax_sx,        'shape x',      0.2,  4.0, valinit=error_shape[0],  valstep=0.1)
    sl_sy        = mwidgets.Slider(ax_sy,        'shape y',      0.2,  4.0, valinit=error_shape[1],  valstep=0.1)
    sl_influence = mwidgets.Slider(ax_influence, 'circ influence', 0.1, 5.0, valinit=obs_influence,  valstep=0.1)
    sl_clip      = mwidgets.Slider(ax_clip,      'clip high',   -5.0,  5.0, valinit=clip_high,       valstep=0.1)
    sl_ebx       = mwidgets.Slider(ax_ebx,       'bias x',       0.2,  4.0, valinit=error_bias[0],   valstep=0.1)
    sl_eby       = mwidgets.Slider(ax_eby,       'bias y',       0.2,  4.0, valinit=error_bias[1],   valstep=0.1)
    chk_bias     = mwidgets.CheckButtons(ax_toggle, ['Show bias ellipse'], [True])

    fig.text(0.12, 0.335, 'Weights & obstacle size',              fontsize=8, color='grey')
    fig.text(0.58, 0.335, 'Ellipse shape, circulation & clip',    fontsize=8, color='grey')
    fig.text(0.12, 0.135, 'Gradient bias (reward shape inside)',  fontsize=8, color='grey')
    fig.text(0.04, 0.03,
             'W/S: drive forward/back  |  A/D: steer left/right  |  '
             'Left-click: reposition obstacle',
             fontsize=8, color='grey', style='italic')

    # --------------------------------------------------------- mutable state
    state = {
        'goal':           list(goal),
        'obstacle':       list(obstacle),
        'theta':          0.0,
        'goal_marker_l':  None,
        'goal_marker_r':  None,
        'obs_ellipse_l':  None,
        'bias_ellipse_l': None,
        'obs_marker_l':   None,
        'obs_ellipse_r':  None,
        'bias_ellipse_r': None,
        'obs_marker_r':   None,
    }

    # ------------------------------------------------------------------ grid
    lin = np.linspace(-grid_range, grid_range, resolution)
    X, Y = np.meshgrid(lin, lin)

    # --------------------------------------------------- reward computation
    def _compute(reward_fn):
        _c1        = sl_c1.val
        _c2        = sl_c2.val
        _c3        = sl_c3.val
        _sx        = sl_sx.val
        _sy        = sl_sy.val
        _ebx       = sl_ebx.val
        _eby       = sl_eby.val
        _radius    = sl_radius.val
        _influence = sl_influence.val
        _clip      = sl_clip.val

        t = state['theta']
        c, s = np.cos(t), np.sin(t)
        X_r =  c * X + s * Y
        Y_r = -s * X + c * Y
        gx, gy = state['goal']
        goal_r = (c * gx + s * gy, -s * gx + c * gy)

        rg = reward_fn(X_r, Y_r, goal_r, error_bias=GOAL_BIAS)
        ro = reward_obstacle_reverse_ellipse(X, Y, tuple(state['obstacle']),
                                             error_shape=[_sx, _sy],
                                             error_bias=[_ebx, _eby],
                                             radius=_radius)
        rc = reward_obstacle_circulation(X, Y, tuple(state['obstacle']), tuple(state['goal']),
                                         radius=_radius, influence=_influence)
        R  = _c1 * rg + _c2 * ro + _c3 * rc
        _lo = obs_penalty
        _hi = _clip if _clip > _lo else _lo + 1e-3
        return np.clip(R, _lo, _hi)

    # ---------------------------------------- per-axes overlay update
    def _ellipse_semi_axes(s0, s1, r):
        """Semi-axes of the ellipse boundary for scaling [s0, s1] and radius r."""
        thr = np.linalg.norm([s0 * r, s1 * r])
        return thr / s0, thr / s1   # (half_x, half_y)

    def _update_overlay(ax, ell_key, bias_ell_key, obs_mkr_key):
        _sx     = sl_sx.val
        _sy     = sl_sy.val
        _ebx    = sl_ebx.val
        _eby    = sl_eby.val
        _radius = sl_radius.val
        _obs    = state['obstacle']

        # --- shape ellipse (boundary — always visible) ---
        if state[ell_key] is not None:
            state[ell_key].remove()
        hx, hy = _ellipse_semi_axes(_sx, _sy, _radius)
        ell = Ellipse(
            xy=_obs, width=2 * hx, height=2 * hy, angle=0,
            edgecolor='white', facecolor='none', linewidth=1.5, linestyle='--', zorder=5,
        )
        ax.add_patch(ell)
        state[ell_key] = ell

        # --- bias ellipse (gradient shape — toggleable) ---
        if state[bias_ell_key] is not None:
            state[bias_ell_key].remove()
        bhx, bhy = _ellipse_semi_axes(_ebx, _eby, _radius)
        bias_ell = Ellipse(
            xy=_obs, width=2 * bhx, height=2 * bhy, angle=0,
            edgecolor='yellow', facecolor='none', linewidth=1.2, linestyle=':', zorder=5,
            visible=chk_bias.get_status()[0],
        )
        ax.add_patch(bias_ell)
        state[bias_ell_key] = bias_ell

        state[obs_mkr_key].set_data([_obs[0]], [_obs[1]])

    # --------------------------------------------------------- redraw both
    def _redraw():
        _c1     = sl_c1.val
        _c2     = sl_c2.val
        _c3     = sl_c3.val
        _sx     = sl_sx.val
        _sy     = sl_sy.val
        _ebx    = sl_ebx.val
        _eby    = sl_eby.val
        _radius = sl_radius.val
        _obs    = state['obstacle']
        _goal   = state['goal']

        mid = resolution // 2
        for ax_, im_, ell_key, bias_ell_key, obs_mkr_key, goal_mkr_key, reward_fn, label in [
            (ax_l, im_l, 'obs_ellipse_l', 'bias_ellipse_l', 'obs_marker_l', 'goal_marker_l', create_reward_bowtie, 'Bowtie'),
            (ax_r, im_r, 'obs_ellipse_r', 'bias_ellipse_r', 'obs_marker_r', 'goal_marker_r', get_reward_hourglass, 'Hourglass'),
        ]:
            R = _compute(reward_fn)
            r_robot = R[mid, mid]
            im_.set_data(R)
            im_.set_clim(R.min(), R.max())
            _update_overlay(ax_, ell_key, bias_ell_key, obs_mkr_key)
            state[goal_mkr_key].set_data([_goal[0]], [_goal[1]])
            ax_.set_title(
                f"{label}  |  r = {_c1:.2f}·rg + {_c2:.2f}·ro + {_c3:.2f}·rc  |  r(robot) = {r_robot:.3f}\n"
                f"goal=({_goal[0]:.2f},{_goal[1]:.2f})  |  obs=({_obs[0]:.2f},{_obs[1]:.2f})  |  "
                f"r={_radius:.2f}  shape=[{_sx:.1f},{_sy:.1f}]  bias=[{_ebx:.1f},{_eby:.1f}]",
                fontsize=9,
            )
        fig.canvas.draw_idle()

    # --------------------------------------------------------- initial render
    _lo0 = obs_penalty
    _hi0 = clip_high if clip_high > _lo0 else _lo0 + 1e-3

    def _init_R(reward_fn):
        rg = reward_fn(X, Y, tuple(state['goal']), error_bias=GOAL_BIAS)
        ro = reward_obstacle_reverse_ellipse(X, Y, tuple(state['obstacle']),
                                             error_shape=error_shape,
                                             error_bias=error_bias,
                                             radius=obs_radius)
        rc = reward_obstacle_circulation(X, Y, tuple(state['obstacle']), tuple(state['goal']),
                                         radius=obs_radius, influence=obs_influence)
        return np.clip(c1 * rg + c2 * ro + c3 * rc, _lo0, _hi0)

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

    # Initial semi-axes for shape and bias ellipses
    _s_thr0  = np.linalg.norm([error_shape[0] * obs_radius, error_shape[1] * obs_radius])
    s_half_x = _s_thr0 / error_shape[0]
    s_half_y = _s_thr0 / error_shape[1]
    _b_thr0  = np.linalg.norm([error_bias[0] * obs_radius, error_bias[1] * obs_radius])
    b_half_x = _b_thr0 / error_bias[0]
    b_half_y = _b_thr0 / error_bias[1]

    for ax_, ell_key, bias_ell_key, obs_mkr_key, goal_mkr_key in [
        (ax_l, 'obs_ellipse_l', 'bias_ellipse_l', 'obs_marker_l', 'goal_marker_l'),
        (ax_r, 'obs_ellipse_r', 'bias_ellipse_r', 'obs_marker_r', 'goal_marker_r'),
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

        # shape ellipse — boundary, always visible
        ell0 = Ellipse(
            xy=state['obstacle'],
            width=2 * s_half_x, height=2 * s_half_y,
            angle=0, edgecolor='white', facecolor='none',
            linewidth=1.5, linestyle='--', zorder=5,
        )
        ax_.add_patch(ell0)
        state[ell_key] = ell0

        # bias ellipse — gradient shape, toggleable
        bias_ell0 = Ellipse(
            xy=state['obstacle'],
            width=2 * b_half_x, height=2 * b_half_y,
            angle=0, edgecolor='yellow', facecolor='none',
            linewidth=1.2, linestyle=':', zorder=5,
            visible=chk_bias.get_status()[0],
        )
        ax_.add_patch(bias_ell0)
        state[bias_ell_key] = bias_ell0

        ax_.set_xlim(-grid_range, grid_range)
        ax_.set_ylim(-grid_range, grid_range)
        ax_.set_xlabel('x — forward (m)', fontsize=11)
        ax_.set_ylabel('y — left (m)', fontsize=11)
        ax_.set_aspect('equal')
        ax_.legend(loc='upper left', fontsize=9)

    _redraw()

    # -------------------------------------------- slider callbacks
    for sl in (sl_c1, sl_c2, sl_c3, sl_radius, sl_sx, sl_sy, sl_influence, sl_clip, sl_ebx, sl_eby):
        sl.on_changed(lambda _: _redraw())
    chk_bias.on_clicked(lambda _: _redraw())

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

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


if __name__ == "__main__":
    fig = plot_reward_heatmap(
        goal=(2.0, 0.0),
        obstacle=(0.8, 0.4),
        c1=1.0,
        c2=5.0,
        c3=2.0,
        obs_radius=0.25,
        obs_influence=1.5,
        obs_penalty=-5.0,
        clip_high=0.0,
        figsize=(16, 8),
        error_shape=[2.0, 1.0],
        error_bias=[2.0, 1.0],
    )
    plt.show()
