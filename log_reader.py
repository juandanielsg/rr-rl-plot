import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from my_custom_markers import CustomMarkers
from typing import OrderedDict


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
        try:
            observation = observation[0]
        except:
            pass
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
        return {
            "observation": self.processing_function(observation),
            "action":      [action[0][0], action[0][1]],
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
        return actions, rewards, observations

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
class LogReader():

    def __init__(self, filename, type="workshop"):

        self.filename = filename
        self.raw_data = self.extract_raw_data()
        self.classified_raw_data = self.separate_data()
        self.process_data()
        self.type = type

        self.processing_functions = {"workshop": self.process_observation_workshop, "maneuvernet": self.process_observation}
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
                episode.append(line.replace("\n",""))

        #Adds last episode to the list
        classified_raw_data.append(episode)

        return classified_raw_data
    
    def process_data(self):

        for i in range(len(self.classified_raw_data)):

            for j in range(len(self.classified_raw_data[i])):
                
                current_line = self.classified_raw_data[i][j]

                #print(current_line)

                current_line = current_line.split("-/-")

                

                # Now we have a state, action and reward in this order
                self.classified_raw_data[i][j] = [self.process_datum(datum) for datum in current_line]

        
        

    def process_datum(self, datum):

        try:

            return json.loads(datum)
        
        except:

            datum = datum.replace("[", "").replace("]","").replace(" ","")
            datum = datum.split(",")
            datum = [float(element) for element in datum]
            return datum
        
    def process_observation_workshop(self, observation):

        """ obs = np.concatenate(
            (
                obstacle,
                goal,
                robot_position,
                rr100_wheel_velocities,
                rr100_steering_angles,
                rr100_steering_velocities,
                robot_velocity,
                mobile_base_orientation,
                mobile_base_angular,
            )"""

        #print(observation)
        observation = observation["observation"][0]
        

        obs = {"obstacle": observation[14:16],
               "goal": observation[0:2],
               "robot_position": observation[2:4],
               "wheel_velocities": observation[4:6],
               "steering_angles": observation[6:8],
               "steering_velocities": observation[8:10],
               "robot_velocity": observation[10:12],
               "mobile_base_orientation": observation[12],
               "mobile_base_angular": observation[13]}

        return obs
        
        
    def process_observation(self, observation):

        """obs = np.concatenate(
            (
                robot_position,
                goal[:2],
                rr100_wheel_velocities,
                rr100_steering_angles,
                rr100_steering_velocities,
                robot_velocity,
                mobile_base_orientation,
                mobile_base_angular,
                # [self.distance_to_obstacle, 0.2],  # distance, obstacle radius
                self.obstacle,
                
            )"""

        if isinstance(observation, dict):

            observation = observation["observation"][0]

        try:
            a = observation[0][0]
            observation = observation[0]
        except:
            pass

        obs = {"robot_position": observation[0:2],
               "goal": observation[2:4],
               "wheel_velocities": observation[4:6],
               "steering_angles": observation[6:8],
               "steering_velocities": observation[8:10],
               "robot_velocity": observation[10:12],
               "mobile_base_orientation": observation[12],
               "mobile_base_angular": observation[13],
               "obstacle": [observation[14], observation[15]]}
        
        return obs

        
            
    def process_step(self, step):

        observation, action, reward = step
        #print(step)

        step = {"observation": self.processing_function(observation),
                "action": [action[0][0], action[0][1]],
                "reward": reward}

        return step

    

    def study_episode(self, idx):

        episode = self.classified_raw_data[idx-1]

        actions = []
        rewards = []
        observations = []

        for step in episode:

            step = self.process_step(step)
            actions.append(step["action"])
            rewards.append(step["reward"])
            observations.append(step["observation"])


        return actions, rewards, observations

    def study_all(self):

        for i in range(1, len(self.classified_raw_data)+1):

            self.data.append(self.study_episode(i))
    
    def get_max_rewards(self, reward_type):

        rewards = []

        rewards = [data[1] for data in self.data]
        rewards = [reward for episode in rewards for reward in episode]



        if isinstance(rewards[0], dict):

            print([reward[reward_type] for reward in rewards])

            return min([reward[reward_type] for reward in rewards])[0], max([reward[reward_type] for reward in rewards])[0]
        
        
        else:

            return min([reward for reward in rewards])[0], max([reward for reward in rewards])[0]


class Plotter():

    def __init__(self):
        pass

    def plot_rewards(self, rewards, idx):
        if isinstance(rewards[0], (dict, OrderedDict)):
            obstacle_penalties = [reward["obstacle_penalty"][0] for reward in rewards]
            whole_rewards      = [reward["reward"][0]           for reward in rewards]
            steps = np.arange(1, len(obstacle_penalties) + 1)

            fig, ax = plt.subplots()
            ax.plot(steps, obstacle_penalties, color='#ff7f0e', label="obstacle penalty")
            ax.fill_between(steps, obstacle_penalties, alpha=0.2, color='#ff7f0e')
            ax.plot(steps, whole_rewards, color='#1f77b4', label="reward")
            ax.fill_between(steps, whole_rewards, alpha=0.5, color='#1f77b4')
            ax.legend()
            plt.ylabel('Reward')
            plt.xlabel('Step')
        else:
            plt.plot(np.arange(1, len(rewards) + 1), rewards)
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

        ax.add_patch(Circle(obstacle, radius=0.2))
        ax.add_patch(Circle(goal, radius=0.1, edgecolor="r", facecolor="none", alpha=0.5))

        lc = plot_colored_line(
            ax, positions_y, positions_x, values,
            cmap="jet", linewidth=3, val_min=val_min, val_max=val_max,
        )
        plt.scatter(positions_y[0], positions_x[0],
                    marker=marker_custom.rr100_marker, s=8000,
                    facecolor="none", edgecolor="g")
        plt.scatter(goal[0], goal[1], marker="x", c="r")

        cbar_label = {"reward": "reward", "obstacle": "obstacle penalty", "goal": "goal reward"}
        fig.colorbar(lc, ax=ax, label=cbar_label.get(type, type))
        plt.title("Trajectory for episode " + str(idx))
        plt.show()

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
            # Remove only LineCollections, leave scatter PathCollections alone
            for coll in list(ax.collections):
                if isinstance(coll, LineCollection):
                    coll.remove()
    
            # Remove current colorbar via the mutable holder
            cbar_holder[0].remove()
    
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
    
        radio.on_clicked(_on_toggle)
        fig._radio = radio
        plt.show()


def generate_gazebo_episode_data(observations, filename):
    with open(filename, "w") as file:
        for observation in observations[:-1]:
            goal     = observation["goal"]
            obstacle = observation["obstacle"]
            file.write(str(goal) + "-/-" + str(obstacle) + "\n")


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

    datalog = LogReader("log_hs_re_2.txt", type="maneuvernet")
    datalog.study_all()

    obstacle_val_min, obstacle_val_max = datalog.get_max_rewards(reward_type="obstacle_penalty")
    val_min, val_max = datalog.get_max_rewards(reward_type="reward")

    plotme = Plotter()

    for idx in range(13, 17):
        actions, rewards, observations = datalog.study_episode(idx)

        plotme.plot_rewards(rewards, idx)

        # Single figure with a reward / obstacle toggle button
        plotme.plot_trajectory_interactive(
            observations, rewards, idx,
            val_min_reward=val_min,       val_max_reward=val_max,
            val_min_obstacle=obstacle_val_min, val_max_obstacle=obstacle_val_max,
        )
