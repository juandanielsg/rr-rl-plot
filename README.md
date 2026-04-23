# rr-rl-plot

Visualisation tools for RR100 reinforcement-learning experiments.

## Installation

```bash
uv sync
```

---

## reward_heatmap.py

Interactive side-by-side heatmap of the reward landscape in the car's reference frame.
The left panel uses the **bowtie** goal-reward shape; the right uses the **hourglass** shape.
Both share the same obstacle, sliders and keyboard controls.

### Run

```bash
uv run reward_heatmap.py
```

### Controls

| Input | Effect |
|---|---|
| **W / X** | Drive forward / backward (translates goal & obstacle) |
| **A / D** | Steer left / right (rotates goal & obstacle around origin) |
| **Left-click** on either heatmap | Reposition the obstacle |
| Sliders | Adjust `c1`, `c2`, obstacle radius, ellipse bias axes, colormap clip |

### Key parameters (`plot_reward_heatmap`)

| Parameter | Default | Description |
|---|---|---|
| `goal` | `(0.0, 2.0)` | Goal position in car frame `(x, y)` |
| `obstacle` | `(-0.8, 0.0)` | Obstacle centre in car frame `(x, y)` |
| `c1` | `1.0` | Weight on goal reward |
| `c2` | `5.0` | Weight on obstacle penalty |
| `obs_radius` | `0.25` | Obstacle collision radius (m) |
| `obs_penalty` | `-5.0` | Colormap floor / penalty magnitude |
| `clip_high` | `0.0` | Upper clip for colormap (keeps the 1/r spike readable) |
| `error_bias` | `[2.0, 1.0]` | Ellipse scaling `[along-goal axis, perpendicular]` |
| `grid_range` | `6.0` | Half-width of the displayed grid (m) |
| `resolution` | `300` | Grid points per axis (lower = more responsive) |
| `linear_step` | `0.2` | Translation per W/X key press (m) |
| `angular_step` | `0.1` | Rotation per A/D key press (rad) |

---

## log_reader.py

Reads structured RL training logs and produces:

- An **animated GIF** with one frame per episode (reward plot + two coloured trajectories).
- A **collection of PNG plots** sorted into `succeeded/` and `failed/` subfolders.
- A **Gazebo simulation file** listing the goal and obstacle position for each episode.

All outputs land in a `test_<logname>/` folder next to the script.

### Run

```bash
uv run log_reader.py --logfile <filename> [--type <workshop|maneuvernet>]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--logfile` | yes | — | Filename inside the `logs/` folder (e.g. `log_bordel.txt`) |
| `--type` | no | `maneuvernet` | Observation format (`workshop` or `maneuvernet`) |

### Example

```bash
uv run log_reader.py --logfile log_bordel.txt --type maneuvernet
```

### Output structure

```
test_log_bordel/
    episodes.gif                     # animated GIF, 2 s per episode
    log_bordel/
        succeeded/
            episode_1.png
            episode_4.png
            ...
        failed/
            episode_2.png
            episode_3.png
            ...
    log_bordel_gazebo_sim.txt        # one line per episode: goal -/- obstacle
```

### Episode plots

Each PNG / GIF frame contains three panels:

| Panel | Content |
|---|---|
| Left | Reward and obstacle-penalty curves over steps |
| Centre | Robot trajectory coloured by total reward |
| Right | Robot trajectory coloured by obstacle penalty |

The title of each frame shows the episode index, outcome (**SUCCESS** / **FAILURE**), and the final distance between the robot and the goal.
An episode is counted as successful when the final robot position is within **0.20 m** of the goal (`SUCCESS_RADIUS`).

### Log file format

Each log file must follow this structure:

```
Episode 1
<obs>-/-<action>-/-<reward>
<obs>-/-<action>-/-<reward>
...
Episode 2
...
```

Logs must be placed inside the `logs/` folder.
