import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from matplotlib.patches import Ellipse
from typing import Tuple
from my_custom_markers import CustomMarkers


def get_reward_hourglass(
    x: np.ndarray,
    y: np.ndarray,
    goal: np.ndarray,
    error_bias: np.ndarray,
    squeeze: float = 1.0,
) -> np.ndarray:
    gx, gy = goal
    dx = gx - x
    dy = gy - y
    delta     = np.minimum(np.abs(dx) - np.abs(dy), 0.0)
    dy_warped = dy - np.sign(dy) * delta * squeeze
    diff      = np.array([dx, dy_warped]) * np.asarray(error_bias).reshape(2, *([1] * np.ndim(y)))
    return -np.linalg.norm(diff, axis=0)


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

    d_act     = np.sqrt((error_shape[0] * dx) ** ex + (error_shape[1] * dy) ** ey)
    threshold = np.sqrt((error_shape[0] * radius) ** ex + (error_shape[1] * radius) ** ey)
    condition = d_act >= threshold

    if penalty is not None:
        return np.where(condition, 0.0, penalty)

    rew_act    = np.sqrt((error_bias[0] * dx) ** ex + (error_bias[1] * dy) ** ey)
    rew_thresh = np.sqrt((error_bias[0] * radius) ** ex + (error_bias[1] * radius) ** ey)
    reward     = np.clip(rew_act / np.where(rew_thresh == 0, 1e-9, rew_thresh) - 1.0, -1.0, 0.0)
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
    ox, oy = obstacle
    gx, gy = goal
    dx, dy = x - ox, y - oy
    d_obs = np.sqrt(dx**2 + dy**2) + eps

    out_x, out_y = dx / d_obs, dy / d_obs

    d_goal = np.sqrt((x - gx)**2 + (y - gy)**2) + eps
    goal_x, goal_y = (gx - x) / d_goal, (gy - y) / d_goal

    alignment   = out_x * goal_x + out_y * goal_y
    circ_signal = (alignment - 1.0) / 2.0

    _sigma       = sigma if sigma is not None else max((influence - radius) * 0.5, 0.1)
    surface_dist = d_obs - radius
    inner_blend  = 1.0 / (1.0 + np.exp(-surface_dist / (_sigma * 0.3)))
    outer_decay  = np.exp(-np.maximum(surface_dist, 0.0)**2 / (2.0 * _sigma**2))

    return circ_signal * inner_blend * outer_decay


def plot_reward_heatmap_hourglass(
    goal: Tuple[float, float] = (3.0, 2.0),
    obstacle: Tuple[float, float] = (-2.0, 1.0),
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 1.0,
    obs_radius: float = 1.0,
    obs_influence: float = 2.0,
    obs_penalty: float = -10.0,
    grid_range: float = 6.0,
    resolution: int = 300,
    clip_high: float = 0.0,
    figsize: Tuple[float, float] = (10, 8),
    error_shape: list = None,
    error_bias: list = None,
    linear_step: float = 0.2,
    angular_step: float = 0.1,
) -> plt.Figure:
    """
    Interactive hourglass reward heatmap.

    The scene is expressed in the car's reference frame (car always at origin).
    W/X translate the world along the car's forward axis; A/D rotate the world
    around the origin (steering). Left-click to reposition the obstacle.
    """
    if error_bias is None:
        error_bias = [2.0, 1.0]
    if error_shape is None:
        error_shape = list(error_bias)

    GOAL_BIAS = np.array([1.0, 2.0])

    # ------------------------------------------------------------------ layout
    fig = plt.figure(figsize=figsize)

    ax      = fig.add_axes([0.05, 0.44, 0.83, 0.51])
    cbar_ax = fig.add_axes([0.90, 0.44, 0.015, 0.51])

    # Three columns: x positions and shared width
    C1, C2, C3, CW = 0.04, 0.36, 0.68, 0.26
    # Four rows (top → bottom), slider height 0.018, pitch 0.040
    R1, R2, R3, R4, SH = 0.350, 0.310, 0.270, 0.230, 0.018

    # Col 1 — Reward weights (3 sliders)
    ax_c1     = fig.add_axes([C1, R1, CW, SH])
    ax_c2     = fig.add_axes([C1, R2, CW, SH])
    ax_c3     = fig.add_axes([C1, R3, CW, SH])

    # Col 2 — Obstacle shape (4 sliders)
    ax_radius = fig.add_axes([C2, R1, CW, SH])
    ax_sx     = fig.add_axes([C2, R2, CW, SH])
    ax_sy     = fig.add_axes([C2, R3, CW, SH])
    ax_exp    = fig.add_axes([C2, R4, CW, SH])

    # Col 3 — Gradient & clipping (4 sliders)
    ax_ebx       = fig.add_axes([C3, R1, CW, SH])
    ax_eby       = fig.add_axes([C3, R2, CW, SH])
    ax_influence = fig.add_axes([C3, R3, CW, SH])
    ax_clip      = fig.add_axes([C3, R4, CW, SH])

    # Toggles — bottom strip
    ax_toggle       = fig.add_axes([C1, 0.105, 0.22, 0.038])
    ax_toggle_shape = fig.add_axes([C2, 0.105, 0.22, 0.038])

    sl_c1        = mwidgets.Slider(ax_c1,        'c1',             0.0, 5.0, valinit=c1,             valstep=0.05)
    sl_c2        = mwidgets.Slider(ax_c2,        'c2',             0.0, 5.0, valinit=c2,             valstep=0.05)
    sl_c3        = mwidgets.Slider(ax_c3,        'c3',             0.0, 5.0, valinit=c3,             valstep=0.05)
    sl_radius    = mwidgets.Slider(ax_radius,    'obs radius',     0.1, 3.0, valinit=obs_radius,     valstep=0.05)
    sl_sx        = mwidgets.Slider(ax_sx,        'shape x',        0.2, 4.0, valinit=error_shape[0], valstep=0.1)
    sl_sy        = mwidgets.Slider(ax_sy,        'shape y',        0.2, 4.0, valinit=error_shape[1], valstep=0.1)
    sl_exp       = mwidgets.Slider(ax_exp,       'exponents',      0.2, 4.0, valinit=0.8,            valstep=0.05)
    sl_ebx       = mwidgets.Slider(ax_ebx,       'bias x',         0.2, 4.0, valinit=error_bias[0],  valstep=0.1)
    sl_eby       = mwidgets.Slider(ax_eby,       'bias y',         0.2, 4.0, valinit=error_bias[1],  valstep=0.1)
    sl_influence = mwidgets.Slider(ax_influence, 'influence', 0.1, 5.0, valinit=obs_influence,  valstep=0.1)
    sl_clip      = mwidgets.Slider(ax_clip,      'clip',     -5.0, 5.0, valinit=clip_high,      valstep=0.1)
    chk_bias     = mwidgets.CheckButtons(ax_toggle,       ['Show bias ellipse'],  [True])
    chk_shape    = mwidgets.CheckButtons(ax_toggle_shape, ['Show shape ellipse'], [True])

    # Group labels
    fig.text(C1, 0.385, 'Reward weights',      fontsize=8, color='grey', fontweight='bold')
    fig.text(C2, 0.385, 'Obstacle shape',       fontsize=8, color='grey', fontweight='bold')
    fig.text(C3, 0.385, 'Gradient & clipping',  fontsize=8, color='grey', fontweight='bold')
    fig.text(C1, 0.03,
             'W/X: drive forward/back  |  A/D: steer left/right  |  '
             'Left-click: reposition obstacle',
             fontsize=8, color='grey', style='italic')

    # --------------------------------------------------------- mutable state
    state = {
        'goal':         list(goal),
        'obstacle':     list(obstacle),
        'theta':        0.0,
        'goal_marker':  None,
        'obs_ellipse':  None,
        'bias_ellipse': None,
        'obs_marker':   None,
    }

    # ------------------------------------------------------------------ grid
    lin = np.linspace(-grid_range, grid_range, resolution)
    X, Y = np.meshgrid(lin, lin)

    # --------------------------------------------------- reward computation
    def _compute():
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

        rg = get_reward_hourglass(X_r, Y_r, goal_r, error_bias=GOAL_BIAS)
        _exp = sl_exp.val
        ro = reward_obstacle_reverse_ellipse(X, Y, tuple(state['obstacle']),
                                             error_shape=[_sx, _sy],
                                             error_bias=[_ebx, _eby],
                                             radius=_radius,
                                             exponents=[_exp, _exp])
        rc = reward_obstacle_circulation(X, Y, tuple(state['obstacle']), tuple(state['goal']),
                                         radius=_radius, influence=_influence)
        R   = _c1 * rg + _c2 * ro + _c3 * rc
        _lo = obs_penalty
        _hi = _clip if _clip > _lo else _lo + 1e-3
        return np.clip(R, _lo, _hi)

    # ------------------------------------------------ overlay update helper
    def _ellipse_semi_axes(s0, s1, r):
        thr = np.linalg.norm([s0 * r, s1 * r])
        return thr / s0, thr / s1

    def _update_overlay():
        _sx     = sl_sx.val
        _sy     = sl_sy.val
        _ebx    = sl_ebx.val
        _eby    = sl_eby.val
        _radius = sl_radius.val
        _obs    = state['obstacle']

        if state['obs_ellipse'] is not None:
            state['obs_ellipse'].remove()
        hx, hy = _ellipse_semi_axes(_sx, _sy, _radius)
        ell = Ellipse(
            xy=(_obs[1], _obs[0]), width=2 * hy, height=2 * hx, angle=0,
            edgecolor='white', facecolor='none', linewidth=1.5, linestyle='--', zorder=5,
            visible=chk_shape.get_status()[0],
        )
        ax.add_patch(ell)
        state['obs_ellipse'] = ell

        if state['bias_ellipse'] is not None:
            state['bias_ellipse'].remove()
        bhx, bhy = _ellipse_semi_axes(_ebx, _eby, _radius)
        bias_ell = Ellipse(
            xy=(_obs[1], _obs[0]), width=2 * bhy, height=2 * bhx, angle=0,
            edgecolor='yellow', facecolor='none', linewidth=1.2, linestyle=':', zorder=5,
            visible=chk_bias.get_status()[0],
        )
        ax.add_patch(bias_ell)
        state['bias_ellipse'] = bias_ell

        state['obs_marker'].set_data([_obs[1]], [_obs[0]])

    # --------------------------------------------------------- redraw
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

        R = _compute()
        mid = resolution // 2
        r_robot = R[mid, mid]
        im.set_data(R.T)
        im.set_clim(R.min(), R.max())
        _update_overlay()
        state['goal_marker'].set_data([_goal[1]], [_goal[0]])
        ax.set_title(
            f"Hourglass  |  r = {_c1:.2f}·rg + {_c2:.2f}·ro + {_c3:.2f}·rc  |  r(robot) = {r_robot:.3f}\n"
            f"goal=({_goal[0]:.2f},{_goal[1]:.2f})  |  obs=({_obs[0]:.2f},{_obs[1]:.2f})  |  "
            f"r={_radius:.2f}  shape=[{_sx:.1f},{_sy:.1f}]  bias=[{_ebx:.1f},{_eby:.1f}]",
            fontsize=9,
        )
        fig.canvas.draw_idle()

    # --------------------------------------------------------- initial render
    _lo0 = obs_penalty
    _hi0 = clip_high if clip_high > _lo0 else _lo0 + 1e-3

    rg0 = get_reward_hourglass(X, Y, tuple(state['goal']), error_bias=GOAL_BIAS)
    ro0 = reward_obstacle_reverse_ellipse(X, Y, tuple(state['obstacle']),
                                          error_shape=error_shape,
                                          error_bias=error_bias,
                                          radius=obs_radius)
    rc0 = reward_obstacle_circulation(X, Y, tuple(state['obstacle']), tuple(state['goal']),
                                      radius=obs_radius, influence=obs_influence)
    R0  = np.clip(c1 * rg0 + c2 * ro0 + c3 * rc0, _lo0, _hi0)

    im = ax.imshow(
        R0.T,
        extent=[-grid_range, grid_range, -grid_range, grid_range],
        origin='lower', cmap='jet', interpolation='bilinear',
        vmin=R0.min(), vmax=R0.max(),
    )
    fig.colorbar(im, cax=cbar_ax).set_label('Reward (clipped)', fontsize=8)

    marker_custom = CustomMarkers()

    _s_thr0  = np.linalg.norm([error_shape[0] * obs_radius, error_shape[1] * obs_radius])
    s_half_x = _s_thr0 / error_shape[0]
    s_half_y = _s_thr0 / error_shape[1]
    _b_thr0  = np.linalg.norm([error_bias[0] * obs_radius, error_bias[1] * obs_radius])
    b_half_x = _b_thr0 / error_bias[0]
    b_half_y = _b_thr0 / error_bias[1]

    obs_m, = ax.plot(
        state['obstacle'][1], state['obstacle'][0], 'wo',
        markersize=10, markeredgecolor='black', label='Obstacle', zorder=6,
    )
    goal_m, = ax.plot(
        state['goal'][1], state['goal'][0], 'wX',
        markersize=14, markeredgecolor='black', markerfacecolor='white',
        label='Goal', zorder=6,
    )
    ax.plot(
        0, 0,
        marker=marker_custom.rr100_marker,
        markersize=20, markeredgecolor='black',
        label='Robot (origin)', zorder=6,
    )
    state['obs_marker']  = obs_m
    state['goal_marker'] = goal_m

    ell0 = Ellipse(
        xy=(state['obstacle'][1], state['obstacle'][0]),
        width=2 * s_half_y, height=2 * s_half_x,
        angle=0, edgecolor='white', facecolor='none',
        linewidth=1.5, linestyle='--', zorder=5,
        visible=chk_shape.get_status()[0],
    )
    ax.add_patch(ell0)
    state['obs_ellipse'] = ell0

    bias_ell0 = Ellipse(
        xy=(state['obstacle'][1], state['obstacle'][0]),
        width=2 * b_half_y, height=2 * b_half_x,
        angle=0, edgecolor='yellow', facecolor='none',
        linewidth=1.2, linestyle=':', zorder=5,
        visible=chk_bias.get_status()[0],
    )
    ax.add_patch(bias_ell0)
    state['bias_ellipse'] = bias_ell0

    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.invert_xaxis()
    ax.set_xlabel('y — left (m)', fontsize=11)
    ax.set_ylabel('x — forward (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)

    _redraw()

    # -------------------------------------------- slider callbacks
    for sl in (sl_c1, sl_c2, sl_c3, sl_radius, sl_sx, sl_sy, sl_influence, sl_clip, sl_ebx, sl_eby, sl_exp):
        sl.on_changed(lambda _: _redraw())
    chk_bias.on_clicked(lambda _: _redraw())
    chk_shape.on_clicked(lambda _: _redraw())

    # -------------------------------------------- click callback
    def _on_click(event):
        if event.inaxes is not ax or event.button != 1:
            return
        state['obstacle'] = [event.ydata, event.xdata]
        _redraw()

    # -------------------------------------------- keyboard callback
    def _rotate(pos, angle):
        c, s = np.cos(angle), np.sin(angle)
        x, y = pos
        return [c * x - s * y, s * x + c * y]

    def _on_key(event):
        key = event.key
        if key == 'w':
            state['goal'][0]     -= linear_step
            state['obstacle'][0] -= linear_step
        elif key == 'x':
            state['goal'][0]     += linear_step
            state['obstacle'][0] += linear_step
        elif key == 'a':
            state['theta']   -= angular_step
            state['goal']     = _rotate(state['goal'],     -angular_step)
            state['obstacle'] = _rotate(state['obstacle'], -angular_step)
        elif key == 'd':
            state['theta']   += angular_step
            state['goal']     = _rotate(state['goal'],     angular_step)
            state['obstacle'] = _rotate(state['obstacle'], angular_step)
        else:
            return
        _redraw()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event',    _on_key)

    return fig


if __name__ == "__main__":
    fig = plot_reward_heatmap_hourglass(
        goal=(2.0, 0.0),
        obstacle=(0.8, 0.4),
        c1=1.0,
        c2=5.0,
        c3=2.0,
        obs_radius=0.25,
        obs_influence=1.5,
        obs_penalty=-5.0,
        clip_high=0.0,
        figsize=(10, 8),
        error_shape=[2.0, 1.0],
        error_bias=[2.0, 1.0],
    )
    plt.show()
