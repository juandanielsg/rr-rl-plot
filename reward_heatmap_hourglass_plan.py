import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from matplotlib.patches import Ellipse, Circle
import matplotlib.transforms as mtransforms
from typing import Tuple
from my_custom_markers import CustomMarkers

ROBOT_WIDTH = 0.65
ROBOT_LENGTH = 0.90
WHEELBASE = 0.55          # RR100: front axle x=0.30, rear axle x=-0.25
OBSTACLE_RADIUS = 0.20
LAT_ACCEL_LIMIT   = 0.02       # max lateral acceleration (m/s²) for velocity-dependent steering
STEER_ACCEL_LIMIT = np.pi / 6  # max steering acceleration (rad/s²)
LIN_ACCEL_LIMIT   = 1.0        # max linear acceleration (m/s²)
MAX_LIN_VEL       = 1.0        # max linear speed (m/s)
RATE              = 40         # planning steps per second


def get_reward_hourglass(
    x: np.ndarray,
    y: np.ndarray,
    goal: np.ndarray,
    error_bias: np.ndarray,
    gradient: float = 1.5,
    theta: float = 0.0,
) -> np.ndarray:
    

    if theta != 0.0:
        c, s = np.cos(theta), np.sin(theta)
        x, y   =  c * x + s * y, -s * x + c * y
        gx, gy = goal
        goal   = (c * gx + s * gy, -s * gx + c * gy)

    gx, gy = goal
    dx = gx - x
    dy = gy - y
    delta     = np.minimum(np.abs(dx) - np.abs(dy), 0.0)

    gradient = np.where(delta==0, gradient, 1)


    dy_warped = (dy - np.sign(dy) * delta)/gradient
    dx = dx/gradient


    diff      = np.array([dx, dy_warped]) * np.asarray(error_bias).reshape(2, *([1] * np.ndim(y)))
    return -(np.linalg.norm(diff, axis=0))


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


def reward_obstacle_triangle(
    x: np.ndarray,
    y: np.ndarray,
    obstacle: Tuple[float, float],
    goal: Tuple[float, float],
    radius: float = ROBOT_WIDTH + OBSTACLE_RADIUS,
    height: float = ROBOT_LENGTH + OBSTACLE_RADIUS,
    penalty: float = None,
) -> np.ndarray:
    ox, oy = obstacle
    gx, gy = goal
    eps = 1e-9

    # Unit vector from obstacle away from goal
    dg_x, dg_y = gx - ox, gy - oy
    D_og = np.sqrt(dg_x**2 + dg_y**2) + eps
    ux, uy = -dg_x / D_og, -dg_y / D_og  # obstacle → away from goal
    px, py = -uy, ux                       # perpendicular

    # Displacement from obstacle to each grid point
    vx, vy = x - ox, y - oy

    s_along = vx * ux + vy * uy     # + = away from goal
    s_perp  = vx * px + vy * py     # lateral distance

    d_obs = np.sqrt(vx**2 + vy**2)

    # Semi-circle on the goal-facing side of the obstacle
    in_circle = (d_obs < radius) & (s_along <= 0)

    # Triangle: base at obstacle (full width = 2*radius), apex at s_along=height (zero width).
    width = 2.0 * radius
    half_w = (width / 2.0) * np.clip(1.0 - s_along / (height + eps), 0.0, 1.0)
    in_triangle = (s_along >= 0) & (s_along <= height) & (np.abs(s_perp) <= half_w)

    if penalty is not None:
        return np.where(in_circle | in_triangle, penalty, 0.0)

    # Smooth rewards: -1 at base/center, 0 at zone boundary
    circle_rew   = np.clip(d_obs / radius - 1.0, -1.0, 0.0)
    triangle_rew = -(1.0 - np.clip(s_along / height, 0.0, 1.0))  # -1 at base, 0 at apex

    rew = np.where(in_triangle, triangle_rew, 0.0)
    rew = np.where(in_circle, np.minimum(circle_rew, rew), rew)
    return rew


def reward_obstacle_triangle_mod(
    x: np.ndarray,
    y: np.ndarray,
    obstacle: Tuple[float, float],
    goal: Tuple[float, float],
    radius: float = ROBOT_WIDTH + OBSTACLE_RADIUS,
    height: float = ROBOT_LENGTH + OBSTACLE_RADIUS,
    penalty: float = None,
    lateral_scale: float = 1.0,
) -> np.ndarray:
    ox, oy = obstacle
    gx, gy = goal
    eps = 1e-9

    # Unit vector from obstacle away from goal
    dg_x, dg_y = gx - ox, gy - oy
    D_og = np.sqrt(dg_x**2 + dg_y**2) + eps
    ux, uy = -dg_x / D_og, -dg_y / D_og  # obstacle → away from goal
    px, py = -uy, ux                       # perpendicular

    # Displacement from obstacle to each grid point
    vx, vy = x - ox, y - oy

    s_along = vx * ux + vy * uy     # + = away from goal
    s_perp  = vx * px + vy * py     # lateral distance

    d_obs = np.sqrt(vx**2 + vy**2)

    # Semi-circle on the goal-facing side of the obstacle
    in_circle = (d_obs < radius) & (s_along <= 0)

    # Triangle: base at obstacle (full width = 2*radius), apex at s_along=height (zero width).
    width = 2.0 * radius
    half_w = (width / 2.0) * np.clip(1.0 - s_along / (height + eps), 0.0, 1.0)
    in_triangle = (s_along >= 0) & (s_along <= height) & (np.abs(s_perp) <= half_w)

    if penalty is not None:
        return np.where(in_circle | in_triangle, penalty, 0.0)

    # Lateral slope: reward increases toward the edges of the zone (outward from spine),
    # creating a gradient that pushes the robot around the obstacle rather than through it.
    lateral      = lateral_scale * np.abs(s_perp) / (radius + eps)
    circle_rew   = np.clip(d_obs / radius - 1.0, -1.0, 0.0)
    triangle_rew = np.minimum(-(1.0 - np.clip(s_along / height, 0.0, 1.0)) + lateral, 0.0)

    rew = np.where(in_triangle, triangle_rew, 0.0)
    rew = np.where(in_circle, np.minimum(circle_rew, rew), rew)
    return rew


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
    obs_radius: float = ROBOT_WIDTH + OBSTACLE_RADIUS,
    tri_height: float = ROBOT_LENGTH + OBSTACLE_RADIUS,
    obs_penalty: float = -10.0,
    grid_range: float = 6.0,
    resolution: int = 300,
    clip_high: float = 0.0,
    figsize: Tuple[float, float] = (10, 8),
    error_shape: list = None,
    error_bias: list = None,
    linear_step: float = 0.02,
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
    fig.canvas.manager.set_window_title("Daniel's Garage")

    ax      = fig.add_axes([0.05, 0.44, 0.83, 0.51])
    cbar_ax = fig.add_axes([0.90, 0.44, 0.015, 0.51])

    # Three columns: x positions and shared width
    C1, C2, C3, CW = 0.04, 0.36, 0.68, 0.26
    # Five rows (top → bottom), slider height 0.018, pitch 0.040
    R1, R2, R3, R4, R5, SH = 0.350, 0.310, 0.270, 0.230, 0.190, 0.018

    # Col 1 — Reward weights (3 sliders) + gradient
    ax_c1   = fig.add_axes([C1, R1, CW, SH])
    ax_c2   = fig.add_axes([C1, R2, CW, SH])
    ax_c3   = fig.add_axes([C1, R3, CW, SH])
    ax_grad = fig.add_axes([C1, R4, CW, SH])

    # Col 2 — Ellipse shape + bias + exponents (5 sliders)
    ax_sx     = fig.add_axes([C2, R1, CW, SH])
    ax_sy     = fig.add_axes([C2, R2, CW, SH])
    ax_ebx    = fig.add_axes([C2, R3, CW, SH])
    ax_eby    = fig.add_axes([C2, R4, CW, SH])
    ax_exp    = fig.add_axes([C2, R5, CW, SH])

    # Col 3 — Distance threshold + triangle height + lateral scale + clipping (4 sliders)
    ax_radius  = fig.add_axes([C3, R1, CW, SH])
    ax_height  = fig.add_axes([C3, R2, CW, SH])
    ax_lateral = fig.add_axes([C3, R3, CW, SH])
    ax_clip    = fig.add_axes([C3, R4, CW, SH])

    # Toggles — bottom strip
    ax_toggle       = fig.add_axes([C1, 0.105, 0.22, 0.038])
    ax_toggle_shape = fig.add_axes([C2, 0.105, 0.22, 0.038])
    ax_toggle_grad  = fig.add_axes([C3, 0.105, 0.22, 0.038])

    # Path buttons
    ax_path_btn  = fig.add_axes([0.04, 0.060, 0.24, 0.032])
    ax_beam_btn  = fig.add_axes([0.30, 0.060, 0.24, 0.032])
    ax_clear_btn = fig.add_axes([0.56, 0.060, 0.16, 0.032])

    sl_c1     = mwidgets.Slider(ax_c1,     'c1',          0.0, 5.0, valinit=c1,             valstep=0.05)
    sl_c2     = mwidgets.Slider(ax_c2,     'c2',          0.0, 5.0, valinit=0.0,            valstep=0.05)
    sl_c3     = mwidgets.Slider(ax_c3,     'c3',          0.0, 5.0, valinit=c3,             valstep=0.05)
    sl_grad   = mwidgets.Slider(ax_grad,   'gradient',    0.1, 3.0, valinit=1.5,            valstep=0.05)
    sl_radius = mwidgets.Slider(ax_radius, 'radius',  0.1, 3.0, valinit=obs_radius,     valstep=0.05)
    sl_sx     = mwidgets.Slider(ax_sx,     'shape x',     0.2, 4.0, valinit=error_shape[0], valstep=0.1)
    sl_sy     = mwidgets.Slider(ax_sy,     'shape y',     0.2, 4.0, valinit=error_shape[1], valstep=0.1)
    sl_exp    = mwidgets.Slider(ax_exp,    'exponents',   0.2, 4.0, valinit=0.8,            valstep=0.05)
    sl_ebx    = mwidgets.Slider(ax_ebx,    'bias x',      0.2, 4.0, valinit=error_bias[0],  valstep=0.1)
    sl_eby    = mwidgets.Slider(ax_eby,    'bias y',      0.2, 4.0, valinit=error_bias[1],  valstep=0.1)
    sl_height  = mwidgets.Slider(ax_height,  'tri height',  0.1, 8.0, valinit=tri_height, valstep=0.1)
    sl_lateral = mwidgets.Slider(ax_lateral, 'lat scale',   0.0, 3.0, valinit=1.0,        valstep=0.05)
    sl_clip    = mwidgets.Slider(ax_clip,    'clip',       -5.0, 5.0, valinit=clip_high,  valstep=0.1)
    chk_bias     = mwidgets.CheckButtons(ax_toggle,       ['Show bias ellipse'],  [False])
    chk_shape    = mwidgets.CheckButtons(ax_toggle_shape, ['Show shape ellipse'], [False])
    chk_grad     = mwidgets.CheckButtons(ax_toggle_grad,  ['Show gradient'],      [True])
    btn_path  = mwidgets.Button(ax_path_btn,  'Gradient path')
    btn_beam  = mwidgets.Button(ax_beam_btn,  'Beam search path')
    btn_clear = mwidgets.Button(ax_clear_btn, 'Clear paths')

    # Group labels
    fig.text(C1, 0.385, 'Reward weights',           fontsize=8, color='grey', fontweight='bold')
    fig.text(C2, 0.385, 'Ellipse shape / bias',     fontsize=8, color='grey', fontweight='bold')
    fig.text(C3, 0.385, 'Distance / height / clip', fontsize=8, color='grey', fontweight='bold')
    fig.text(C1, 0.03,
             'W/X: drive along arc  |  A/D: adjust steering  |  S: straighten  |  M: reset  |  '
             'Left-click: reposition obstacle',
             fontsize=8, color='grey', style='italic')

    # --------------------------------------------------------- mutable state
    state = {
        'goal':         list(goal),
        'obstacle':     list(obstacle),
        'steering':     0.0,
        'chess_px':     0.0,
        'chess_py':     0.0,
        'chess_phi':    0.0,
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
        _radius = sl_radius.val
        _height = sl_height.val
        _clip   = sl_clip.val

        rg = get_reward_hourglass(X, Y, tuple(state['goal']), error_bias=GOAL_BIAS, gradient=sl_grad.val)
        _exp = sl_exp.val
        ro = reward_obstacle_reverse_ellipse(X, Y, tuple(state['obstacle']),
                                             error_shape=[_sx, _sy],
                                             error_bias=[_ebx, _eby],
                                             radius=_radius,
                                             exponents=[_exp, _exp])
        rc = reward_obstacle_triangle_mod(X, Y, tuple(state['obstacle']), tuple(state['goal']),
                                      radius=_radius, height=_height, lateral_scale=sl_lateral.val)
        R_raw = _c1 * rg + _c2 * ro + _c3 * rc
        _lo = obs_penalty
        _hi = _clip if _clip > _lo else _lo + 1e-3
        return np.clip(R_raw, _lo, _hi), R_raw

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

        state['obs_marker'].set_center((_obs[1], _obs[0]))

    # ------------------------------------------------ gradient path helpers
    def _reward_at(px, py):
        _c1  = sl_c1.val;  _c2  = sl_c2.val;  _c3  = sl_c3.val
        _sx  = sl_sx.val;  _sy  = sl_sy.val;   _exp = sl_exp.val
        _ebx = sl_ebx.val; _eby = sl_eby.val
        _r   = sl_radius.val; _h = sl_height.val
        rg = get_reward_hourglass(px, py, tuple(state['goal']), error_bias=GOAL_BIAS)
        ro = reward_obstacle_reverse_ellipse(
            px, py, tuple(state['obstacle']),
            error_shape=[_sx, _sy], error_bias=[_ebx, _eby],
            radius=_r, exponents=[_exp, _exp])
        rc = reward_obstacle_triangle(
            px, py, tuple(state['obstacle']), tuple(state['goal']),
            radius=_r, height=_h)
        return float(_c1 * rg + _c2 * ro + _c3 * rc)

    def _run_gradient_path():
        STEP      = 0.01
        FD        = 0.02
        MAX       = 500
        THRESH    = ROBOT_WIDTH / 2
        MAX_STEER = 0.57  # RR100 joint limits ±0.57 rad

        px, py  = 0.0, 0.0
        heading = 0.0
        goal    = state['goal']
        xs, ys  = [px], [py]

        for _ in range(MAX):
            if np.sqrt((px - goal[0])**2 + (py - goal[1])**2) < THRESH:
                break

            gx = (_reward_at(px + FD, py) - _reward_at(px - FD, py)) / (2 * FD)
            gy = (_reward_at(px, py + FD) - _reward_at(px, py - FD)) / (2 * FD)
            if np.sqrt(gx**2 + gy**2) < 1e-9:
                break

            phi     = np.arctan2(gy, gx)
            dhead_f = (phi - heading + np.pi) % (2 * np.pi) - np.pi  # [-π, π]

            if abs(dhead_f) <= np.pi / 2:
                ds    = STEP
                delta = np.clip(np.arctan2(dhead_f * WHEELBASE, 2 * STEP),
                                -MAX_STEER, MAX_STEER)
            else:
                ds      = -STEP
                dhead_b = dhead_f - (np.pi if dhead_f >= 0 else -np.pi)
                delta   = np.clip(np.arctan2(-dhead_b * WHEELBASE, 2 * STEP),
                                  -MAX_STEER, MAX_STEER)

            if abs(delta) < 1e-6:
                px += ds * np.cos(heading)
                py += ds * np.sin(heading)
            else:
                R_turn  = WHEELBASE / (2 * np.tan(delta))
                dtheta  = ds / R_turn
                dx_h    = R_turn * np.sin(dtheta)
                dy_h    = R_turn * (1 - np.cos(dtheta))
                ch, sh  = np.cos(heading), np.sin(heading)
                px     += ch * dx_h - sh * dy_h
                py     += sh * dx_h + ch * dy_h
                heading += dtheta

            xs.append(px)
            ys.append(py)

        return np.array(xs), np.array(ys)

    def _run_beam_search_path():
        NUM_BEAMS = 10
        MAX_STEPS = 400
        N_STEERS  = 9
        N_SPEEDS  = 5
        THRESH    = ROBOT_WIDTH / 2

        goal = state['goal']

        def _kin(px, py, h, delta, ds):
            if abs(delta) < 1e-6:
                return px + ds * np.cos(h), py + ds * np.sin(h), h
            R_t = WHEELBASE / (2 * np.tan(delta))
            dt  = ds / R_t
            ch, sh = np.cos(h), np.sin(h)
            dx = R_t * np.sin(dt)
            dy = R_t * (1 - np.cos(dt))
            return px + ch*dx - sh*dy, py + sh*dx + ch*dy, h + dt

        # Each beam: [cum_reward, px, py, heading, steer, vel, done, path_xs, path_ys]
        beams = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, [0.0], [0.0]]]

        for _ in range(MAX_STEPS):
            if all(b[6] for b in beams):
                break

            cands = []
            for bi, beam in enumerate(beams):
                cum_r, px, py, h, steer, vel, done = (
                    beam[0], beam[1], beam[2], beam[3], beam[4], beam[5], beam[6])
                if done:
                    cands.append((cum_r, px, py, h, steer, vel, True, bi))
                    continue

                v_lo = max(-1.0, vel - LIN_ACCEL_LIMIT / RATE)
                v_hi = min( 1.0, vel + LIN_ACCEL_LIMIT / RATE)
                for ds in np.linspace(v_lo, v_hi, N_SPEEDS):
                    v_lim = _steer_limit(abs(ds))
                    s_lo  = max(-v_lim, steer - STEER_ACCEL_LIMIT / RATE)
                    s_hi  = min( v_lim, steer + STEER_ACCEL_LIMIT / RATE)
                    for delta in np.linspace(s_lo, s_hi, N_STEERS):
                        nx, ny, nh = _kin(px, py, h, delta, ds)
                        r = _reward_at(nx, ny)
                        is_done = np.hypot(nx - goal[0], ny - goal[1]) < THRESH
                        cands.append((cum_r + r, nx, ny, nh, delta, ds, is_done, bi))

            cands.sort(key=lambda c: c[0], reverse=True)
            top = cands[:NUM_BEAMS]

            new_beams = []
            for (cum_r, px, py, h, steer, vel, done, bi) in top:
                parent = beams[bi]
                new_beams.append([cum_r, px, py, h, steer, vel, done,
                                   parent[7] + [px], parent[8] + [py]])
            beams = new_beams

            if beams[0][6]:
                break

        best = beams[0]
        return np.array(best[7]), np.array(best[8])

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

        R, R_raw = _compute()
        mid = resolution // 2
        r_robot = R[mid, mid]
        im.set_data(R.T)
        im.set_clim(R.min(), R.max())
        chess_im.set_data(_chess_array())
        _update_overlay()
        state['goal_marker'].set_data([_goal[1]], [_goal[0]])
        _height = sl_height.val
        ax.set_title(
            f"Hourglass  |  r = {_c1:.2f}·rg + {_c2:.2f}·ro + {_c3:.2f}·rt\n"
            f"goal=({_goal[0]:.2f},{_goal[1]:.2f})  |  obs=({_obs[0]:.2f},{_obs[1]:.2f})  |  "
            f"r={_radius:.2f}  shape=[{_sx:.1f},{_sy:.1f}]  bias=[{_ebx:.1f},{_eby:.1f}]  "
            f"tri h={_height:.1f} w={2*_radius:.1f}",
            fontsize=9,
        )
        _dg = lin[1] - lin[0]
        gx = (R_raw[mid, mid+1] - R_raw[mid, mid-1]) / (2 * _dg)
        gy = (R_raw[mid+1, mid] - R_raw[mid-1, mid]) / (2 * _dg)
        _gmag = np.sqrt(gx**2 + gy**2)
        _scale = 1.5 / _gmag if _gmag > 1e-9 else 0.0
        grad_arrow.set_UVC(gy * _scale, gx * _scale)
        grad_arrow.set_visible(chk_grad.get_status()[0])
        grad_text.set_text(f"∇r = ({gx:.2f}, {gy:.2f})")
        reward_text.set_text(f"r(robot) = {r_robot:.3f}")
        fig.canvas.draw_idle()

    # --------------------------------------------------------- initial render
    _lo0 = obs_penalty
    _hi0 = clip_high if clip_high > _lo0 else _lo0 + 1e-3

    rg0 = get_reward_hourglass(X, Y, tuple(state['goal']), error_bias=GOAL_BIAS)
    ro0 = reward_obstacle_reverse_ellipse(X, Y, tuple(state['obstacle']),
                                          error_shape=error_shape,
                                          error_bias=error_bias,
                                          radius=obs_radius)
    rc0 = reward_obstacle_triangle(X, Y, tuple(state['obstacle']), tuple(state['goal']),
                                   radius=obs_radius, height=tri_height)
    R0  = np.clip(c1 * rg0 + c2 * ro0 + c3 * rc0, _lo0, _hi0)

    im = ax.imshow(
        R0.T,
        extent=[-grid_range, grid_range, -grid_range, grid_range],
        origin='lower', cmap='jet', interpolation='bilinear',
        vmin=R0.min(), vmax=R0.max(), zorder=0,
    )
    fig.colorbar(im, cax=cbar_ax).set_label('Reward (clipped)', fontsize=8)

    def _chess_array():
        phi = state['chess_phi']
        c, s = np.cos(phi), np.sin(phi)
        px, py = state['chess_px'], state['chess_py']
        # Robot-frame grid → world frame, then check cell parity
        Rx = lin[:, None]   # robot x (forward), shape (resolution, 1)
        Ry = lin[None, :]   # robot y (lateral),  shape (1, resolution)
        wx = px + c * Rx - s * Ry
        wy = py + s * Rx + c * Ry
        return ((np.floor(wx).astype(int) + np.floor(wy).astype(int)) % 2).astype(float)

    chess_im = ax.imshow(
        _chess_array(),
        extent=[-grid_range, grid_range, -grid_range, grid_range],
        origin='lower', cmap='gray', alpha=0.05,
        interpolation='nearest', zorder=1,
    )

    marker_custom = CustomMarkers()

    _s_thr0  = np.linalg.norm([error_shape[0] * obs_radius, error_shape[1] * obs_radius])
    s_half_x = _s_thr0 / error_shape[0]
    s_half_y = _s_thr0 / error_shape[1]
    _b_thr0  = np.linalg.norm([error_bias[0] * obs_radius, error_bias[1] * obs_radius])
    b_half_x = _b_thr0 / error_bias[0]
    b_half_y = _b_thr0 / error_bias[1]

    obs_circle = Circle(
        xy=(state['obstacle'][1], state['obstacle'][0]),
        radius=OBSTACLE_RADIUS,
        facecolor='white', edgecolor='black', linewidth=1.5,
        label='Obstacle', zorder=6,
    )
    ax.add_patch(obs_circle)

    goal_m, = ax.plot(
        state['goal'][1], state['goal'][0], 'wX',
        markersize=14, markeredgecolor='black', markerfacecolor='white',
        label='Goal', zorder=6,
    )

    robot_plot, = ax.plot(
        0, 0,
        marker=marker_custom.rr100_marker,
        markersize=20, markeredgecolor='black', markerfacecolor='gray',
        label='Robot (origin)', zorder=6,
    )

    ax.annotate(
        '', xy=(0, ROBOT_LENGTH * 0.7), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='red', lw=2.0),
        zorder=7,
    )

    grad_arrow = ax.quiver(
        0, 0, 0, 0,
        angles='xy', scale_units='xy', scale=1,
        color='lime', width=0.012, zorder=8,
        edgecolors='black', linewidth=1.2,
    )

    steering_arc, = ax.plot([], [], color='yellow', linewidth=1.5,
                            linestyle='--', alpha=0.8, zorder=7)

    def _update_steering_arc():
        steering = state['steering']
        if abs(steering) < 1e-4:
            xs = np.array([0.0, 0.8])
            ys = np.array([0.0, 0.0])
        else:
            R_turn = WHEELBASE / (2 * np.tan(steering))
            ts = np.linspace(0, 0.8 / R_turn, 40)
            xs = R_turn * np.sin(ts)
            ys = R_turn * (1 - np.cos(ts))
        steering_arc.set_data(ys, xs)

    traj_glow, = ax.plot([], [], color='cyan',   linewidth=5,   alpha=0.35, zorder=8)
    traj_line, = ax.plot([], [], color='white',  linewidth=1.8, linestyle='--', zorder=9)
    beam_glow, = ax.plot([], [], color='orange', linewidth=5,   alpha=0.35, zorder=8)
    beam_line, = ax.plot([], [], color='yellow', linewidth=1.8, linestyle='--', zorder=9)

    def _update_robot_markersize():
        # Convert ROBOT_WIDTH × ROBOT_LENGTH from data units to points at current zoom
        bbox = ax.get_window_extent()          # axis size in display units (pixels)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        pts_per_unit_x = bbox.width  / abs(xlim[1] - xlim[0]) * (72 / fig.dpi)
        pts_per_unit_y = bbox.height / abs(ylim[1] - ylim[0]) * (72 / fig.dpi)
        # Use the average so the symmetric marker looks right
        markersize = (ROBOT_WIDTH * pts_per_unit_x + ROBOT_LENGTH * pts_per_unit_y) / 2
        robot_plot.set_markersize(markersize)

    fig.canvas.mpl_connect('draw_event', lambda _: _update_robot_markersize())

    state['obs_marker']  = obs_circle
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

    reward_text = fig.text(
        0.15, 0.695, '',
        ha='center', va='center',
        fontsize=15, fontweight='bold', rotation=0,
    )

    grad_text = fig.text(
        0.15, 0.655, '',
        ha='center', va='center',
        fontsize=12, fontweight='bold', rotation=0,
    )

    _update_steering_arc()
    _redraw()

    # -------------------------------------------- slider callbacks
    for sl in (sl_c1, sl_c2, sl_c3, sl_grad, sl_radius, sl_sx, sl_sy, sl_height, sl_lateral, sl_clip, sl_ebx, sl_eby, sl_exp):
        sl.on_changed(lambda _: _redraw())
    chk_bias.on_clicked(lambda _: _redraw())
    chk_shape.on_clicked(lambda _: _redraw())
    chk_grad.on_clicked(lambda _: _redraw())

    # -------------------------------------------- click callback
    def _on_click(event):
        if event.inaxes is not ax or event.button != 1:
            return
        state['obstacle'] = [event.ydata, event.xdata]
        _redraw()

    # -------------------------------------------- keyboard callback
    _MAX_STEER = 0.57  # RR100: ±0.57 rad (≈33°)

    def _steer_limit(v):
        if abs(v) < 1e-9:
            return _MAX_STEER
        return min(_MAX_STEER, float(np.arctan(WHEELBASE * LAT_ACCEL_LIMIT / (2 * v ** 2))))

    def _arc_transform(pos, ds):
        steering = state['steering']
        if abs(steering) < 1e-6:
            return [pos[0] - ds, pos[1]]
        R_turn = WHEELBASE / (2 * np.tan(steering))
        dtheta = ds / R_turn
        c, s = np.cos(dtheta), np.sin(dtheta)
        px = pos[0] - R_turn * s
        py = pos[1] - R_turn * (1 - c)
        return [c * px + s * py, -s * px + c * py]

    def _update_chess_arc(ds):
        steering = state['steering']
        if abs(steering) < 1e-6:
            phi = state['chess_phi']
            state['chess_px'] += ds * np.cos(phi)
            state['chess_py'] += ds * np.sin(phi)
            return
        R_turn = WHEELBASE / (2 * np.tan(steering))
        dtheta = ds / R_turn
        dx_r = R_turn * np.sin(dtheta)
        dy_r = R_turn * (1 - np.cos(dtheta))
        phi = state['chess_phi']
        c, s = np.cos(phi), np.sin(phi)
        state['chess_px']  += c * dx_r - s * dy_r
        state['chess_py']  += s * dx_r + c * dy_r
        state['chess_phi'] += dtheta

    def _on_key(event):
        key = event.key
        if key == 'w':
            state['goal']     = _arc_transform(state['goal'],     linear_step)
            state['obstacle'] = _arc_transform(state['obstacle'], linear_step)
            _update_chess_arc(linear_step)
        elif key == 'x':
            state['goal']     = _arc_transform(state['goal'],     -linear_step)
            state['obstacle'] = _arc_transform(state['obstacle'], -linear_step)
            _update_chess_arc(-linear_step)
        elif key == 'a':
            lim = _steer_limit(linear_step)
            new = np.clip(state['steering'] + angular_step, -lim, lim)
            state['steering'] = 0.0 if abs(new) < angular_step * 0.5 else new
            _update_steering_arc()
            fig.canvas.draw_idle()
            return
        elif key == 'd':
            lim = _steer_limit(linear_step)
            new = np.clip(state['steering'] - angular_step, -lim, lim)
            state['steering'] = 0.0 if abs(new) < angular_step * 0.5 else new
            _update_steering_arc()
            fig.canvas.draw_idle()
            return
        elif key == 's':
            state['steering'] = 0.0
            _update_steering_arc()
            fig.canvas.draw_idle()
            return
        elif key == 'm':
            state['goal']      = list(goal)
            state['obstacle']  = list(obstacle)
            state['steering']  = 0.0
            state['chess_px']  = 0.0
            state['chess_py']  = 0.0
            state['chess_phi'] = 0.0
            traj_glow.set_data([], [])
            traj_line.set_data([], [])
            beam_glow.set_data([], [])
            beam_line.set_data([], [])
            _update_steering_arc()
        else:
            return
        _redraw()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event',    _on_key)

    def _on_path_btn(_):
        xs, ys = _run_gradient_path()
        traj_glow.set_data(ys, xs)
        traj_line.set_data(ys, xs)
        fig.canvas.draw_idle()

    def _on_beam_btn(_):
        xs, ys = _run_beam_search_path()
        beam_glow.set_data(ys, xs)
        beam_line.set_data(ys, xs)
        fig.canvas.draw_idle()

    def _on_clear_btn(_):
        traj_glow.set_data([], [])
        traj_line.set_data([], [])
        beam_glow.set_data([], [])
        beam_line.set_data([], [])
        fig.canvas.draw_idle()

    btn_path.on_clicked(_on_path_btn)
    btn_beam.on_clicked(_on_beam_btn)
    btn_clear.on_clicked(_on_clear_btn)
    fig._widgets = [btn_path, btn_beam, btn_clear]

    return fig


if __name__ == "__main__":
    fig = plot_reward_heatmap_hourglass(
        goal=(2.0, 0.0),
        obstacle=(0.8, 0.4),
        c1=1.0,
        c2=5.0,
        c3=2.0,
        obs_radius=ROBOT_WIDTH + OBSTACLE_RADIUS,
        tri_height=ROBOT_LENGTH + OBSTACLE_RADIUS,
        obs_penalty=-5.0,
        clip_high=0.0,
        figsize=(10, 8),
        error_shape=[2.0, 1.0],
        error_bias=[2.0, 1.0],
    )
    plt.show()
