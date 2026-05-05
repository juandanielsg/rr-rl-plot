import argparse
import io
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from my_custom_markers import CustomMarkers
from pathlib import Path
from typing import OrderedDict
from PIL import Image

SUCCESS_RADIUS = 0.20


class LogReader():

    def __init__(self, filename, type="workshop"):

        self.filename = filename
        self.raw_data = self.extract_raw_data()
        self.classified_raw_data = self.separate_data()
        self.process_data()
        self.type = type

        self.processing_functions = {
            "workshop":    self.process_observation_workshop,
            "maneuvernet": self.process_observation,
        }
        self.processing_function = self.processing_functions[self.type]

        self.data = []

    def extract_raw_data(self):
        with open("logs/" + self.filename) as file:
            return file.readlines()

    def separate_data(self):
        classified_raw_data = []
        episode = None
        for line in self.raw_data:
            if line.find("Episode") != -1:
                if episode is not None:
                    classified_raw_data.append(episode)
                episode = []
            else:
                episode.append(line.replace("\n", ""))
        classified_raw_data.append(episode)
        return classified_raw_data

    def process_data(self):
        for i in range(len(self.classified_raw_data)):
            for j in range(len(self.classified_raw_data[i])):
                current_line = self.classified_raw_data[i][j].split("-/-")
                self.classified_raw_data[i][j] = [
                    self.process_datum(datum) for datum in current_line
                ]

    def process_datum(self, datum):
        try:
            return json.loads(datum)
        except:
            datum = datum.replace("[", "").replace("]", "").replace(" ", "")
            datum = datum.split(",")
            return [float(element) for element in datum]

    def process_observation_workshop(self, observation):
        observation = observation["observation"][0]
        return {
            "obstacle":              observation[14:16],
            "goal":                  observation[0:2],
            "robot_position":        observation[2:4],
            "wheel_velocities":      observation[4:6],
            "steering_angles":       observation[6:8],
            "steering_velocities":   observation[8:10],
            "robot_velocity":        observation[10:12],
            "mobile_base_orientation": observation[12],
            "mobile_base_angular":   observation[13],
        }

    def process_observation(self, observation):
        if isinstance(observation, dict):
            observation = observation["observation"][0]
        if isinstance(observation[0], (list, np.ndarray)):
            observation = observation[0]
        return {
            "robot_position":        observation[0:2],
            "goal":                  observation[2:4],
            "wheel_velocities":      observation[4:6],
            "steering_angles":       observation[6:8],
            "steering_velocities":   observation[8:10],
            "robot_velocity":        observation[10:12],
            "mobile_base_orientation": observation[12],
            "mobile_base_angular":   observation[13],
            "obstacle":              [observation[14], observation[15]],
        }

    def process_step(self, step):
        observation, action, reward = step
        action = action[0] if isinstance(action[0], (list, np.ndarray)) else action
        return {
            "observation": self.processing_function(observation),
            "action":      [action[0], action[1]],
            "reward":      reward,
        }

    def study_episode(self, idx):
        episode = self.classified_raw_data[idx - 1]
        actions, rewards, observations = [], [], []
        for step in episode:
            step = self.process_step(step)
            actions.append(step["action"])
            rewards.append(step["reward"])
            observations.append(step["observation"])

        # The log appends the reset observation (robot back to ~[0,0], new goal) as the
        # last line before the next "Episode" marker.  Strip it when present so that
        # observations[-1] is always the true final step of this episode.
        if len(observations) > 1 and not np.allclose(
            observations[-1]["goal"], observations[0]["goal"]
        ):
            observations = observations[:-1]
            actions = actions[:-1]
            rewards = rewards[:-1]

        last = observations[-1]
        rx, ry = last["robot_position"]
        gx, gy = last["goal"]

        #print("Ep jump")
        #print(rx, ry)
        #print(gx, gy)
        #print(np.sqrt((rx - gx) ** 2 + (ry - gy) ** 2))
        success = np.sqrt((rx - gx) ** 2 + (ry - gy) ** 2) < SUCCESS_RADIUS

        return actions, rewards, observations, success

    def study_all(self):
        for i in range(1, len(self.classified_raw_data) + 1):
            self.data.append(self.study_episode(i))

    def get_max_rewards(self, reward_type):
        rewards = [reward for data in self.data for reward in data[1]]
        if isinstance(rewards[0], dict):
            return (
                min(reward[reward_type] for reward in rewards)[0],
                max(reward[reward_type] for reward in rewards)[0],
            )
        return (
            min(reward for reward in rewards)[0],
            max(reward for reward in rewards)[0],
        )


class Plotter():

    def __init__(self):
        pass

    def plot_rewards(self, rewards, idx):
        if isinstance(rewards[0], (dict, OrderedDict)):
            obstacle_penalties = [reward["obstacle_penalty"][0] for reward in rewards]
            whole_rewards      = [reward["reward"][0]           for reward in rewards]
            steps = np.arange(1, len(obstacle_penalties) + 1)

            goal_rewards = [w - o for w, o in zip(whole_rewards, obstacle_penalties)]

            fig, ax = plt.subplots()
            # ax.fill_between(steps, whole_rewards, alpha=0.5, color='#4C4C9D')
            # ax.fill_between(steps, obstacle_penalties, alpha=0.35, color='#71A2B6')
            ax.plot(steps, obstacle_penalties, color='#71A2B6', label="obstacle penalty",
                    linewidth=2.5, marker='o', markersize=8, markevery=(19, 20))
            ax.plot(steps, obstacle_penalties, lw=0, marker='o', markersize=3,
                    markevery=(19, 20), markerfacecolor='black', markeredgecolor='none',
                    label='_nolegend_')
            ax.plot(steps, goal_rewards, color='#60B2E5', label="goal reward",
                    linewidth=2.5, marker='o', markersize=8, markevery=(19, 20))
            ax.plot(steps, goal_rewards, lw=0, marker='o', markersize=3,
                    markevery=(19, 20), markerfacecolor='black', markeredgecolor='none',
                    label='_nolegend_')
            ax.plot(steps, whole_rewards, color='#4C4C9D', label="reward",
                    linewidth=2.5, marker='o', markersize=8, markevery=(19, 20))
            ax.plot(steps, whole_rewards, lw=0, marker='o', markersize=3,
                    markevery=(19, 20), markerfacecolor='black', markeredgecolor='none',
                    label='_nolegend_')
            for s in range(20, len(steps) + 1, 20):
                ax.axvline(x=s, color='gray', linewidth=0.7, alpha=0.5, linestyle='--')
            ax.set_xticks(np.arange(20, len(steps) + 1, 20))
            ax.legend(loc='lower right')
            plt.ylabel('Reward')
            plt.xlabel('Step')
        else:
            steps = np.arange(1, len(rewards) + 1)
            # plt.fill_between(steps, rewards, alpha=0.5, color='#4C4C9D')
            plt.plot(steps, rewards, color='#4C4C9D', linewidth=2.5,
                     marker='o', markersize=8, markevery=(19, 20))
            plt.plot(steps, rewards, lw=0, marker='o', markersize=3,
                     markevery=(19, 20), markerfacecolor='black', markeredgecolor='none',
                     label='_nolegend_')
            for s in range(20, len(steps) + 1, 20):
                plt.axvline(x=s, color='gray', linewidth=0.7, alpha=0.5, linestyle='--')
            plt.xticks(np.arange(20, len(steps) + 1, 20))
            plt.ylabel('Reward')
            plt.xlabel('Step')

        plt.title("Reward evolution for episode " + str(idx))
        plt.show()

    def plot_trajectory(
        self,
        observations,
        rewards,
        idx,
        type="reward",
        val_min=0.0,
        val_max=10.0,
    ):
        if type == "reward":
            values = (
                [r["reward"][0]          for r in rewards] if isinstance(rewards[0], dict)
                else [r[0]               for r in rewards]
            )
        elif type == "obstacle":
            values = (
                [r["obstacle_penalty"][0] for r in rewards] if isinstance(rewards[0], dict)
                else [r[0]               for r in rewards]
            )
        elif type == "goal":
            values = (
                [r["reward"][0] - r["obstacle_penalty"][0] for r in rewards]
                if isinstance(rewards[0], dict)
                else [r[0] for r in rewards]
            )

        marker_custom = CustomMarkers()
        positions = [step["robot_position"] for step in observations]
        positions_x, positions_y = zip(*positions[:-1])
        positions_y = [-y for y in positions_y]
        goal     = [-observations[0]["goal"][1],     observations[0]["goal"][0]]
        obstacle = [-observations[0]["obstacle"][1], observations[0]["obstacle"][0]]

        fig, ax = plt.subplots()
        ax.axis("equal")
        ax.set(xlim=(-4, 4), ylim=(-4, 4))
        draw_chessboard(ax)

        ax.add_patch(Circle(obstacle, radius=0.2))
        ax.add_patch(Circle(goal, radius=0.1, edgecolor="r", facecolor="none", alpha=0.5))

        lc = plot_colored_line(
            ax, positions_y, positions_x, values,
            cmap="jet", linewidth=3, val_min=val_min, val_max=val_max,
        )
        plt.scatter(positions_y[0], positions_x[0],
                    marker=marker_custom.rr100_marker, s=1600,
                    facecolor="none", edgecolor="g")
        plt.scatter(goal[0], goal[1], marker="x", c="r")

        cbar_label = {"reward": "reward", "obstacle": "obstacle penalty", "goal": "goal reward"}
        fig.colorbar(lc, ax=ax, label=cbar_label.get(type, type))
        plt.title("Trajectory for episode " + str(idx))
        plt.show()

    def create_episode_gif(
        self,
        data,
        output_path,
        n_seconds=3,
        val_min=0.0,
        val_max=10.0,
        obstacle_val_min=0.0,
        obstacle_val_max=10.0,
    ):
        """
        Saves an animated GIF with one frame per episode.

        Each frame is a 3-panel figure:
          left   — reward evolution over steps
          center — robot trajectory coloured by total reward
          right  — robot trajectory coloured by obstacle penalty

        Args:
            data:               list of (actions, rewards, observations, success) per episode,
                                as returned by LogReader.study_all / study_episode.
            output_path:        destination path for the .gif file.
            n_seconds:          how long each episode frame is displayed (seconds).
            val_min/val_max:    colormap range for the reward trajectory.
            obstacle_val_min/obstacle_val_max:
                                colormap range for the obstacle-penalty trajectory.
        """
        frames = []

        for ep_idx, (actions, rewards, observations, success) in enumerate(data, start=1):
            fig = self._render_episode_figure(
                ep_idx, rewards, observations, success,
                val_min, val_max, obstacle_val_min, obstacle_val_max,
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()

        if not frames:
            return

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(n_seconds * 1000),
            loop=0,
        )

    def generate_collection(
        self,
        data,
        log_filename,
        val_min=0.0,
        val_max=10.0,
        obstacle_val_min=0.0,
        obstacle_val_max=10.0,
        output_dir=None,
    ):
        """
        Saves one PNG per episode into a structured folder tree.

        Layout:
            [output_dir/]<log_filename_stem>/
                succeeded/episode_<n>.png
                failed/episode_<n>.png

        Args:
            data:          list of (actions, rewards, observations, success) per episode.
            log_filename:  original log filename (e.g. "log_hs_re_2.txt"); used as the
                           root folder name (extension stripped).
            val_min/val_max:               colormap range for the reward trajectory.
            obstacle_val_min/obstacle_val_max: colormap range for the obstacle trajectory.
            output_dir:    optional parent directory; defaults to the current directory.
        """
        base = Path(output_dir) if output_dir else Path(".")
        root = base / Path(log_filename).stem
        succeeded_dir = root / "succeeded"
        failed_dir    = root / "failed"
        succeeded_dir.mkdir(parents=True, exist_ok=True)
        failed_dir.mkdir(parents=True, exist_ok=True)

        for ep_idx, (actions, rewards, observations, success) in enumerate(data, start=1):
            fig = self._render_episode_figure(
                ep_idx, rewards, observations, success,
                val_min, val_max, obstacle_val_min, obstacle_val_max,
            )
            dest = succeeded_dir if success else failed_dir
            fig.savefig(dest / f"episode_{ep_idx}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

    def _render_episode_figure(
        self,
        ep_idx,
        rewards,
        observations,
        success,
        val_min,
        val_max,
        obstacle_val_min,
        obstacle_val_max,
    ):
        """Build and return the 3-panel figure for one episode (caller must close it)."""
        marker_custom = CustomMarkers()
        is_dict = isinstance(rewards[0], dict)

        values_reward = (
            [r["reward"][0]           for r in rewards] if is_dict else [r[0] for r in rewards]
        )
        values_obstacle = (
            [r["obstacle_penalty"][0] for r in rewards] if is_dict else [r[0] for r in rewards]
        )

        positions = [step["robot_position"] for step in observations]
        pos_x, pos_y = zip(*positions)
        pos_y        = [-y for y in pos_y]
        goal         = [-observations[0]["goal"][1],     observations[0]["goal"][0]]
        obstacle_pos = [-observations[0]["obstacle"][1], observations[0]["obstacle"][0]]

        final_pos  = positions[-1]
        final_plot = (-final_pos[1], final_pos[0])

        last_obs = observations[-1]
        rx, ry   = last_obs["robot_position"]
        gx, gy   = last_obs["goal"]
        final_dist_goal = np.sqrt((rx - gx) ** 2 + (ry - gy) ** 2)

        steps         = np.arange(1, len(rewards) + 1)
        outcome       = "SUCCESS" if success else "FAILURE"
        outcome_color = "#2ca02c" if success else "#d62728"

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(
            f"Episode {ep_idx}  —  {outcome}  |  Final distance to goal: {final_dist_goal:.2f} m",
            fontsize=14, fontweight="bold", color=outcome_color,
        )

        # Left: reward evolution
        ax0 = axes[0]
        if is_dict:
            obs_penalties = [r["obstacle_penalty"][0] for r in rewards]
            whole_rewards = [r["reward"][0]           for r in rewards]
            goal_rewards  = [w - o for w, o in zip(whole_rewards, obs_penalties)]
            # ax0.fill_between(steps, whole_rewards, alpha=0.5, color="#4C4C9D")
            # ax0.fill_between(steps, obs_penalties, alpha=0.35, color="#71A2B6")
            ax0.plot(steps, obs_penalties, color="#71A2B6", label="obstacle penalty",
                     linewidth=2.5, marker='o', markersize=8, markevery=(19, 20))
            ax0.plot(steps, obs_penalties, lw=0, marker='o', markersize=3,
                     markevery=(19, 20), markerfacecolor='black', markeredgecolor='none',
                     label='_nolegend_')
            ax0.plot(steps, goal_rewards, color="#60B2E5", label="goal reward",
                     linewidth=2.5, marker='o', markersize=8, markevery=(19, 20))
            ax0.plot(steps, goal_rewards, lw=0, marker='o', markersize=3,
                     markevery=(19, 20), markerfacecolor='black', markeredgecolor='none',
                     label='_nolegend_')
            ax0.plot(steps, whole_rewards, color="#4C4C9D", label="reward",
                     linewidth=2.5, marker='o', markersize=8, markevery=(19, 20))
            ax0.plot(steps, whole_rewards, lw=0, marker='o', markersize=3,
                     markevery=(19, 20), markerfacecolor='black', markeredgecolor='none',
                     label='_nolegend_')
            for s in range(20, len(steps) + 1, 20):
                ax0.axvline(x=s, color='gray', linewidth=0.7, alpha=0.5, linestyle='--')
            ax0.legend(loc='lower right')
        else:
            # ax0.fill_between(steps, values_reward, alpha=0.5, color="#4C4C9D")
            ax0.plot(steps, values_reward, color="#4C4C9D", linewidth=2.5,
                     marker='o', markersize=8, markevery=(19, 20))
            ax0.plot(steps, values_reward, lw=0, marker='o', markersize=3,
                     markevery=(19, 20), markerfacecolor='black', markeredgecolor='none',
                     label='_nolegend_')
            for s in range(20, len(steps) + 1, 20):
                ax0.axvline(x=s, color='gray', linewidth=0.7, alpha=0.5, linestyle='--')
        if success:
            ax0.set_xticks(np.arange(20, len(steps) + 1, 20))
        else:
            ax0.set_xticks([20, len(steps)])
        ax0.set_ylabel("Reward")
        ax0.set_xlabel("Step")
        ax0.set_title("Reward evolution")

        # Centre: trajectory coloured by total reward
        ax1 = axes[1]
        ax1.set_aspect("equal")
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(-4, 4)
        draw_chessboard(ax1)
        ax1.add_patch(Circle(obstacle_pos, radius=0.2))
        ax1.add_patch(Circle(goal, radius=0.1, edgecolor="r", facecolor="none", alpha=0.5))
        lc1 = plot_colored_line(
            ax1, pos_y, pos_x, values_reward,
            cmap="jet", linewidth=3, val_min=val_min, val_max=val_max,
        )
        ax1.scatter(pos_y[0], pos_x[0],
                    marker=marker_custom.rr100_marker, s=1600,
                    facecolor="none", edgecolor="g", zorder=5)
        ax1.scatter(goal[0], goal[1], marker="x", c="r", zorder=5)
        ax1.scatter(*final_plot, marker="P" if success else "X",
                    s=100, c=outcome_color, zorder=6, label=outcome)
        ax1.legend(loc="upper left", fontsize=8)
        fig.colorbar(lc1, ax=ax1, label="reward")
        ax1.set_title("Global trajectory (reward)")

        # Right: trajectory coloured by obstacle penalty
        ax2 = axes[2]
        ax2.set_aspect("equal")
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(-4, 4)
        draw_chessboard(ax2)
        ax2.add_patch(Circle(obstacle_pos, radius=0.2))
        ax2.add_patch(Circle(goal, radius=0.1, edgecolor="r", facecolor="none", alpha=0.5))
        lc2 = plot_colored_line(
            ax2, pos_y, pos_x, values_obstacle,
            cmap="jet", linewidth=3, val_min=obstacle_val_min, val_max=obstacle_val_max,
        )
        ax2.scatter(pos_y[0], pos_x[0],
                    marker=marker_custom.rr100_marker, s=1600,
                    facecolor="none", edgecolor="g", zorder=5)
        ax2.scatter(goal[0], goal[1], marker="x", c="r", zorder=5)
        ax2.scatter(*final_plot, marker="P" if success else "X",
                    s=100, c=outcome_color, zorder=6, label=outcome)
        ax2.legend(loc="upper left", fontsize=8)
        fig.colorbar(lc2, ax=ax2, label="obstacle penalty")
        ax2.set_title("Global trajectory (obstacle only)")

        plt.tight_layout()
        return fig

    def plot_trajectory_interactive(
        self,
        observations,
        rewards,
        idx,
        val_min_reward=0.0,
        val_max_reward=10.0,
        val_min_obstacle=0.0,
        val_max_obstacle=10.0,
    ):
        is_dict = isinstance(rewards[0], dict)
        values_reward = (
            [r["reward"][0]           for r in rewards] if is_dict else [r[0] for r in rewards]
        )
        values_obstacle = (
            [r["obstacle_penalty"][0] for r in rewards] if is_dict else [r[0] for r in rewards]
        )

        marker_custom = CustomMarkers()
        positions     = [step["robot_position"] for step in observations]
        pos_x, pos_y  = zip(*positions[:-1])
        pos_y         = [-y for y in pos_y]
        goal          = [-observations[0]["goal"][1],     observations[0]["goal"][0]]
        obstacle      = [-observations[0]["obstacle"][1], observations[0]["obstacle"][0]]

        fig = plt.figure(figsize=(7, 7))
        ax  = fig.add_axes([0.08, 0.15, 0.80, 0.78])
        ax.axis("equal")
        ax.set(xlim=(-4, 4), ylim=(-4, 4))
        draw_chessboard(ax)

        ax_radio = fig.add_axes([0.25, 0.02, 0.50, 0.09])
        radio    = mwidgets.RadioButtons(
            ax_radio, labels=("reward", "obstacle"), active=0, activecolor="#1f77b4"
        )
        for label in radio.labels:
            label.set_fontsize(10)

        ax.add_patch(Circle(obstacle, radius=0.2, label="Obstacle"))
        ax.add_patch(Circle(goal, radius=0.1, edgecolor="r", facecolor="none",
                            alpha=0.5, label="Goal"))
        ax.scatter(pos_y[0], pos_x[0],
                   marker=marker_custom.rr100_marker, s=8000,
                   facecolor="none", edgecolor="g", zorder=5, label="Robot start")
        ax.scatter(goal[0], goal[1], marker="x", c="r", zorder=5)
        ax.legend(loc="upper left", fontsize=8)

        lc = plot_colored_line(
            ax, pos_y, pos_x, values_reward,
            cmap="jet", linewidth=3,
            val_min=val_min_reward, val_max=val_max_reward,
        )
        # Use a list so the closure always reads the *current* colorbar
        cbar_holder = [fig.colorbar(lc, ax=ax, label="reward")]
        ax.set_title(f"Trajectory for episode {idx}  —  reward")

        def _on_toggle(label):
            for coll in list(ax.collections):
                if isinstance(coll, LineCollection):
                    coll.remove()

            # Fully remove the colorbar and its axes from the figure
            cbar_holder[0].remove()
            cbar_holder[0].ax.remove()      # <-- this is the missing line
            fig.subplots_adjust()           # reset layout so the main ax expands back

            if label == "reward":
                new_lc = plot_colored_line(
                    ax, pos_y, pos_x, values_reward,
                    cmap="jet", linewidth=3,
                    val_min=val_min_reward, val_max=val_max_reward,
                )
                cbar_holder[0] = fig.colorbar(new_lc, ax=ax, label="reward")
                ax.set_title(f"Trajectory for episode {idx}  —  reward")
            else:
                new_lc = plot_colored_line(
                    ax, pos_y, pos_x, values_obstacle,
                    cmap="jet", linewidth=3,
                    val_min=val_min_obstacle, val_max=val_max_obstacle,
                )
                cbar_holder[0] = fig.colorbar(new_lc, ax=ax, label="obstacle penalty")
                ax.set_title(f"Trajectory for episode {idx}  —  obstacle penalty")

            fig.canvas.draw_idle()


def generate_gazebo_episode_data(data, filename):
    """Write one line per episode containing its goal and obstacle positions.

    Each line has the format:  goal -/- obstacle
    where goal and obstacle are the values from the first observation of each episode
    (they are fixed for the duration of an episode).
    """
    with open(filename, "w") as file:
        for actions, rewards, observations, success in data:
            goal     = observations[0]["goal"]
            obstacle = observations[0]["obstacle"]
            file.write(str(goal) + "-/-" + str(obstacle) + "\n")


def draw_chessboard(ax, x_min=-4, x_max=4, y_min=-4, y_max=4):
    import math
    colors = ["white", "#E8E8E8"]
    for i in range(math.floor(x_min), math.ceil(x_max)):
        for j in range(math.floor(y_min), math.ceil(y_max)):
            color = colors[(i + j) % 2]
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=color, edgecolor="none", zorder=0))


def plot_colored_line(ax, x, y, values, cmap="plasma", linewidth=2,
                      val_min=0.0, val_max=10.0, **kwargs):
    x      = np.asarray(x,      dtype=float)
    y      = np.asarray(y,      dtype=float)
    values = np.asarray(values, dtype=float)

    points   = np.stack([x, y], axis=1)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    seg_values = (values[:-1] + values[1:]) / 2

    norm = Normalize(vmin=val_min, vmax=val_max)
    lc   = LineCollection(segments, cmap=cmap, norm=norm,
                          linewidth=linewidth, **kwargs)
    lc.set_array(seg_values)
    ax.add_collection(lc)
    return lc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", required=True, help="Log filename inside the logs/ folder")
    parser.add_argument("--type", choices=["workshop", "maneuvernet"], default="maneuvernet",
                        help="Observation format used during logging")
    args = parser.parse_args()

    datalog = LogReader(args.logfile, type=args.type)
    datalog.study_all()

    obstacle_val_min, obstacle_val_max = datalog.get_max_rewards(reward_type="obstacle_penalty")
    val_min, val_max = datalog.get_max_rewards(reward_type="reward")

    plotme = Plotter()

    logname = Path(datalog.filename).stem
    output_dir = Path("test_" + logname)
    output_dir.mkdir(exist_ok=True)

    plotme.create_episode_gif(
        datalog.data,
        output_path=output_dir / "episodes.gif",
        n_seconds=2,
        val_min=val_min,
        val_max=val_max,
        obstacle_val_min=obstacle_val_min,
        obstacle_val_max=obstacle_val_max,
    )

    plotme.generate_collection(
        datalog.data,
        log_filename=datalog.filename,
        val_min=val_min,
        val_max=val_max,
        obstacle_val_min=obstacle_val_min,
        obstacle_val_max=obstacle_val_max,
        output_dir=output_dir,
    )

    generate_gazebo_episode_data(
        datalog.data,
        output_dir / (logname + "_gazebo_sim.txt"),
    )

    """for idx in range(13, 17):
        actions, rewards, observations, success = datalog.study_episode(idx)

        plotme.plot_rewards(rewards, idx)

        # Single figure with a reward / obstacle toggle button
        plotme.plot_trajectory_interactive(
            observations, rewards, idx,
            val_min_reward=val_min,       val_max_reward=val_max,
            val_min_obstacle=obstacle_val_min, val_max_obstacle=obstacle_val_max,
        )"""
