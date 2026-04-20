import numpy as np
import json
import matplotlib.pyplot as plt
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

        #print(type(rewards[0]))

        if isinstance(rewards[0], dict) or isinstance(rewards[0], OrderedDict):

            obstacle_penalties = [reward["obstacle_penalty"][0] for reward in rewards]
            whole_rewards = [reward["reward"][0] for reward in rewards]
            steps = np.arange(1, len(obstacle_penalties)+1)

            #print(obstacle_penalties)

            fig,ax = plt.subplots()
            ax.plot(steps, obstacle_penalties, color = '#ff7f0e', label = "obstacle penalty")
            ax.fill_between(steps,obstacle_penalties, alpha = 0.2, color = '#ff7f0e')
            ax.plot(steps, whole_rewards, color = '#1f77b4', label = "reward")
            ax.fill_between(steps,whole_rewards, alpha = 0.5, color = '#1f77b4')
            temp = ax.legend()
            plt.ylabel('Reward')
            plt.xlabel('Step')

        else:   

            plt.plot(np.arange(1, len(rewards)+1), rewards)
            plt.ylabel('Reward')
            plt.xlabel('Step')


        plt.title("Reward evolution for episode " + str(idx))
        plt.show()



    
    def plot_trajectory(self, observations, rewards, idx, type="reward", val_min = 0.0, val_max = 10.0):


        if type == "reward":
            rewards = [reward["reward"][0] for reward in rewards] if isinstance(rewards[0], dict) else [reward[0] for reward in rewards]
        
        elif type == "obstacle":
            rewards = [reward["obstacle_penalty"][0] for reward in rewards] if isinstance(rewards[0], dict) else [reward[0] for reward in rewards]
        
        elif type == "goal":
            rewards = [reward["reward"][0] - reward["obstacle_penalty"][0] for reward in rewards] if isinstance(rewards[0], dict) else [reward[0] for reward in rewards]

        #print(rewards)
        marker_custom = CustomMarkers()

        #print(observations[0])

        positions = [step["robot_position"] for step in observations]
        #print(positions)
        positions_x, positions_y = zip(*positions[:-1])
        positions_y = [-y for y in positions_y]
        goal = [-observations[0]["goal"][1], observations[0]["goal"][0]]
        #print(goal)
        obstacle = [-observations[0]["obstacle"][1], observations[0]["obstacle"][0]]


        fig, ax = plt.subplots()

        ax.axis("equal")
        ax.set(xlim=(-4, 4), ylim=(-4, 4))

        obs = Circle(obstacle, radius=0.2)
        goalcircle = Circle(goal, radius=0.1, edgecolor="r", facecolor="none", alpha=0.5)

        #Plot the trajectory
        lc = plot_colored_line(ax, positions_y, positions_x, rewards, cmap="jet", linewidth=3, val_min = val_min, val_max=val_max)
        #plt.plot(positions_y, positions_x, c="g", marker=",")

        #Plot the trajectory starting point
        plt.scatter(positions_y[0], positions_x[0], marker=marker_custom.rr100_marker, s=8000, facecolor="none", edgecolor="g")

        #Plot the goal
        plt.scatter(goal[0], goal[1], marker="x", c="r")

        ax.add_patch(obs)
        ax.add_patch(goalcircle)
        if type == "obstacle":
            fig.colorbar(lc, ax=ax, label="obstacle penalty")
        
        elif type== "reward":
            fig.colorbar(lc, ax=ax, label="reward")
        
        elif type == "goal":
            fig.colorbar(lc, ax=ax, label="goal reward")

        plt.title("Trajectory for episode " + str(idx))
        plt.show()
        pass



def generate_gazebo_episode_data(observations, filename):
    """Creates a txt file containing 2 tuples of goal and obstacle
    positions in order to replicate PyBullet test runs in Gazebo or
    any other simulator."""

    with open(filename, "w") as file:

        for observation in observations[:-1]:

            goal = observation["goal"]
            obstacle = observation["obstacle"]

            file.write(str(goal) + "-/-" + str(obstacle) + "\n")

    
def plot_colored_line(ax, x, y, values, cmap="plasma", linewidth=2, val_min=0.0, val_max=10.0, **kwargs):
    """
    Plot a line on `ax` where each segment is colored according to `values`.
 
    Parameters
    ----------
    ax      : matplotlib Axes
    x       : array-like, x coordinates
    y       : array-like, y coordinates
    values  : array-like, scalar value at each point (same length as x, y)
              drives the colormap — e.g. speed, temperature, time
    cmap    : str or Colormap, default "plasma"
    linewidth: float, default 2
    **kwargs: forwarded to LineCollection (e.g. alpha, zorder)
 
    Returns
    -------
    lc  : LineCollection  (add a colorbar with fig.colorbar(lc, ax=ax))
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
 
    # Build segments: each segment connects point i to point i+1
    # Shape: (N-1, 2, 2)
    points = np.stack([x, y], axis=1)                  # (N, 2)
    segments = np.stack([points[:-1], points[1:]], axis=1)  # (N-1, 2, 2)
 
    # Color each segment by the average value of its two endpoints
    seg_values = (values[:-1] + values[1:]) / 2
 

    # Represent the minimal and maximal values for the line
    norm = Normalize(vmin=val_min, vmax=val_max)



    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        linewidth=linewidth, **kwargs)
    lc.set_array(seg_values)
    ax.add_collection(lc)
 
    # Expand axes limits to fit the line
    #ax.set_xlim(x.min(), x.max())
    #ax.set_ylim(y.min(), y.max())
    #ax.autoscale_view()
 
    return lc



if __name__ == "__main__":

    
    #datalog = LogReader("log_hs_re_2.txt", type="maneuvernet")
    datalog = LogReader("log_hs_re_2.txt", type="maneuvernet")
    datalog.study_all()
    obstacle_val_min, obstacle_val_max = datalog.get_max_rewards(reward_type="obstacle_penalty")
    val_min, val_max = datalog.get_max_rewards(reward_type="reward")
    plotme = Plotter()

    for idx in range(13, 17): #len(datalog.classified_raw_data)+1):
        actions, rewards, observations = datalog.study_episode(idx)
        #print(rewards)
        plotme.plot_rewards(rewards, idx)
        plotme.plot_trajectory(observations, rewards, idx, type = "reward", val_min = val_min, val_max=val_max)
        plotme.plot_trajectory(observations, rewards, idx, type = "obstacle", val_min = obstacle_val_min, val_max = obstacle_val_max)
        #plotme.plot_trajectory(observations, rewards, idx, type="goal")
        #generate_gazebo_episode_data(observations, filename="sim_ep_" + str(idx) + ".txt")
    