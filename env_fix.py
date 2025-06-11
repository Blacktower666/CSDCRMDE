import config
import numpy as np
import random
from queue import PriorityQueue
from collections import Counter
from itertools import chain
from math import log


class Node:
    def __init__(self, mem, disk, cpu_freq, bandwidth, x, y, energy_ratio):
        self.mem = mem
        self.disk = disk
        self.init_mem = self.mem

        self.cpu_freq = cpu_freq
        self.total_cpu_freq = cpu_freq
        self.bandwidth = bandwidth
        self.energy_ratio = energy_ratio

        self.task_queue = PriorityQueue()
        self.lfu_cache_image = {}
        self.status = 0

        self.download_finish_time = 0
        # image three states: 0: not local 1: downloading 2: local
        self.image_list = [0] * config.IMAGE_NUM
        self.image_download_time = [0] * config.IMAGE_NUM

        self.x = x
        self.y = y


class Image:
    def __init__(self, image_size):
        self.image_size = image_size


class Task:
    def __init__(self, mem, cpu_freq, start_time, image_id, ddl, x, y, task_size, transmission_energy):
        self.mem = mem
        self.cpu_freq = cpu_freq        # Record the cpu frequency required by the task
        self.start_time = start_time    # Record the task start time
        self.image_id = image_id        # Record the id of the image required by the task
        self.ddl = ddl                  # Record ddl
        self.cpu_freq_using = 0         # Record the cpu frequency assigned to the task
        # Distance of task from base station
        self.x = x
        self.y = y
        self.task_size = task_size      # Size of task
        self.transmission_energy = transmission_energy# Energy consumption when task is transmitted to the base station

class Env:
    def __init__(self):
        self.__init_env()

    def seed(self, seed):
        self.seed = seed

    def __init_env(self):
        self.node_num = config.EDGE_NODE_NUM
        self.image_num = config.IMAGE_NUM
        self.n_observations = 7 * self.node_num + 1 + 5 #The length of #state

        self.node = []
        self.image = []
        self.task = []

        self.time = -1
        self.reward = 0
        self.rewards = []

        self.total_time = 0
        self.total_energy = 0
        self.download_time = 0
        self.trans_time = 0
        self.comp_time = 0

        self.task_finish_list = []
        self.num_on_time = 0
        self.total_task = 0
        self.done = False # Whether all tasks are completed

    def reset(self):
        # random.seed(self.seed)
        # np.random.seed(self.seed)
        self.__init_env()

        # 1. create nodes
        for _ in range(self.node_num):
            ratio = random.randint(config.min_E, config.max_E)
            self.node.append(Node(
                random.randint(config.node_mem_min, config.node_mem_max),# Set the memory size
                random.randint(config.node_disk_min, config.node_disk_max),# Set the storage size
                random.randint(config.node_cpu_freq_min, config.node_cpu_freq_max),# Set the cpu frequency
                random.uniform(config.node_band_min, config.node_band_max),# Set bandwidth
                # Consider node location
                random.random() * config.max_x,
                random.random() * config.max_y,
                ratio))# Set node power consumption

        # 2. create image
        for _ in range(self.image_num):
            self.image.append(Image(
                random.uniform(config.image_size_min, config.image_size_max)))# Set the size of image

        # 3. create task
        # Each t time comes several tasks
        for i in range(2000):
            for _ in range(random.randint(config.min_tasks, config.max_tasks)):
                randn = int(np.random.normal(self.image_num // 2, 8, 1))
                image_id = randn if randn >= 0 and randn <= self.image_num - 1 else random.randint(0,self.image_num - 1)
                cpu_freq = config.task_cpu_freq_min + random.random() * (config.task_cpu_freq_max - config.task_cpu_freq_min)
                ddl = i + random.randint(3, 5)
                # ddl=i
                self.task.append(Task(
                    random.random() * config.task_mem_max,
                    cpu_freq,
                    start_time=i, image_id=image_id, ddl=ddl,
                    x=random.random() * config.max_x,
                    y=random.random() * config.max_y,
                    task_size=random.random() * config.task_size_max / 1000,
                    transmission_energy=random.uniform(0.7, 1.0) * config.max_t))
                self.total_task += 1

    def imgae_download(self, task, idx):
        if self.node[idx].image_list[task.image_id] == 0:
            # Download image
            self.node[idx].image_list[task.image_id] = 1
            self.node[idx].disk -= self.image[task.image_id].image_size
            # Download time
            # TODO
            download_time = self.image[task.image_id].image_size / (2 * self.node[idx].bandwidth)  # Time required to download: Time period
            self.node[idx].image_download_time[task.image_id] = max(self.node[idx].download_finish_time,
                                                                    self.time) + download_time  # Download completion time: point in time
            # image on node queued for download
            self.node[idx].download_finish_time = self.node[idx].image_download_time[task.image_id]
            download_finish_time = self.node[idx].image_download_time[task.image_id]
        # If the node is downloading this image, get the remaining download time
        elif self.node[idx].image_list[task.image_id] == 1:
            download_finish_time = self.node[idx].image_download_time[task.image_id]
        else:
            download_finish_time = max(0, self.time)

        return download_finish_time

    def _add_task(self, tasks, actions, t_on_n):
        rewards = []
        # observations = []
        for i in range(len(tasks)):
            task = tasks[i] # Gets the task under consideration
            idx = actions[i] # Gets the node to which the current task is assigned
            if self.node[idx].image_list[task.image_id] == 0:
                # Download image
                self.node[idx].image_list[task.image_id] = 1
                self.node[idx].disk -= self.image[task.image_id].image_size
                # Download time
                # TODO
                download_time = self.image[task.image_id].image_size / (2 * self.node[idx].bandwidth)  # Time required to download: Time period
                self.node[idx].image_download_time[task.image_id] = max(self.node[idx].download_finish_time,
                                                                        self.time) + download_time  # Download completion time: point in time
                # image on node queued for download
                self.node[idx].download_finish_time = self.node[idx].image_download_time[task.image_id]
                download_finish_time = self.node[idx].image_download_time[task.image_id]
            # If the node is downloading this image, get the remaining download time
            elif self.node[idx].image_list[task.image_id] == 1:
                download_finish_time = self.node[idx].image_download_time[task.image_id]
            else:
                download_finish_time = max(0, self.time)
            # Update the resources occupied by the task and the completion time of the task
            self.node[idx].mem -= task.mem
            # print('idx',idx)
            task.cpu_freq_using = self.node[idx].cpu_freq / t_on_n[idx] # Average the remaining cpu frequency of the current node according to the number of tasks assigned to the node
            self.node[idx].cpu_freq -= task.cpu_freq_using
            comp_time = task.cpu_freq / task.cpu_freq_using # Calculate the time required for the operation
            comp_engery = self.node[idx].energy_ratio * comp_time * task.cpu_freq_using / self.node[idx].total_cpu_freq# Computing power consumption
            trans_time = task.task_size / self.uplink_trans_rate(task, self.node[idx], t_on_n[idx])# Calculate the task transfer time
            trans_engery = trans_time * task.transmission_energy# Calculate the energy consumption of the transmission
            task_finish_time = download_finish_time + comp_time + trans_time# Calculate the total time required
            self.node[idx].task_queue.put((task_finish_time, random.random(), task))# Add the task to the task queue of this node
            delay = task.ddl - task_finish_time# Calculate delay based on ddl
            encourage = 1
            if delay >= 0: # Consider whether the task can be completed before ddl to calculate the completion rate
                self.num_on_time += 1
                # encourage = 0
            energy = -trans_engery - comp_engery# Calculate total energy consumption
            reward = config.e1 * (energy) + config.e2 * (delay*encourage)# Calculate reward
            rewards.append(reward)# Add the reward to the reward list
            self.total_time += task_finish_time - task.start_time # Calculate the total time required to complete the task
            self.total_energy += comp_engery + trans_engery # Calculate the total energy required to complete the task
            self.download_time += download_finish_time - task.start_time
            self.trans_time += trans_time
            self.comp_time += comp_time
            task.x = self.node[idx].x
            task.y = self.node[idx].y

        return rewards

    # Update the environment every time t
    # Upgrade node first, then task
    def env_up(self):
        self.time += 1
        for idx, n in enumerate(self.node):
            # Determine whether the task on each node has been completed
            # The execution time of the task is not every t time-1, but at a certain time, the task is directly executed and completed
            while n.task_queue.empty() == False:
                curr_task = n.task_queue.get()
                if self.time >= curr_task[0]:# Release resources if the task is complete
                    n.cpu_freq += curr_task[2].cpu_freq_using
                    n.mem += curr_task[2].mem
                    self.task_finish_list.append(str(self.time - curr_task[2].start_time))
                else:
                    n.task_queue.put(curr_task)# Put the task back in the task queue
                    break

            # Check whether the image on each node is downloaded
            for i in range(len(n.image_download_time)):
                if n.image_list[i] == 1 and self.time >= n.image_download_time[i]:
                    n.image_list[i] = 2
        if not self.task:
            self.done = 1
            return self.done, None, None
        return False, False, None

    def cal_dist(self, task, node):# Calculate the distance between the task and the node
        x = np.array([task.x, task.y])
        y = np.array([node.x, node.y])
        return np.sqrt(sum(np.power((x - y), 2)))

    def uplink_trans_rate(self, task, node, t_on_n):
        #The function is used to calculate the uplink transfer rate
        # Parameter:
        # -task: indicates the task to be transferred
        # -node: The node (device) responsible for the transmission task
        # -t_on_n: The number of tasks that the node processes, or the time that the node is in the active transfer state (assuming the same processing time for each task)
        # Transmit power, in decibels
        trans_power = 23
        # Noise power, in decibels
        noise_power = -174
        # Calculate the distance between the task and the node (in kilometers) and convert it to meters
        dist = self.cal_dist(task, node) / 1000 + 1e-5 # Convert distance to meters, avoid dividing by zero
        # The channel gain is calculated using the free space transmission loss model (dist^(-2))
        channel_gain = dist ** (-2) * 10 ** (-2)
        # Calculate Signal to Noise ratio (SNR)
        gamma = (trans_power * channel_gain) / noise_power ** 2
        # Use the Shannon capacity formula to calculate the spectral efficiency
        eta = log(1 + gamma, 2)
        # Use spectral efficiency and node parameters to calculate the uplink transmission rate
        # (The number of tasks processed by each node multiplied by the uplink rate of each task)
        return (node.bandwidth / t_on_n) * eta

    def get_obs(self, task):
        obs = {}
        # node state
        node_mem_list = [n.mem / 10 for n in self.node]
        node_disk_list = [n.disk / 10 for n in self.node]
        cpu_freq_list = [n.cpu_freq / 3 for n in self.node]
        bandwidth_list = [n.bandwidth for n in self.node]
        engery_ratio_list = [n.energy_ratio for n in self.node]
        obs["node"] = np.hstack((node_mem_list, node_disk_list, cpu_freq_list, bandwidth_list, engery_ratio_list))
        node_image_list = [n.image_list[task.image_id] for n in self.node]
        download_time_list = []
        for n in self.node:
            if n.image_list[task.image_id] == 2:
                download_time_list.append(0)
            elif n.image_list[task.image_id] == 1:
                download_time_list.append(n.image_download_time[task.image_id] - self.time)
            else:
                download_time_list.append(max(self.time, n.download_finish_time) + self.image[
                    task.image_id].image_size / n.bandwidth - self.time)
        obs["image"] = np.hstack((node_image_list, download_time_list, self.image[task.image_id].image_size))

        # task state
        obs["task"] = [task.mem, task.cpu_freq / 3, task.image_id, task.transmission_energy, task.ddl-self.time]
        return list(chain(*obs.values()))

    def step(self, tasks, actions, t_on_n, states):
        observations = []# Observe the status of the assigned task
        # dones = []
        for i in range(1, len(states)):
            observations.append(states[i])

        rewards = self._add_task(tasks, actions, t_on_n)# Assign tasks in each time slot to the corresponding node and calculate its reward
        if len(self.task)!=0:
            observations.append(self.get_obs(self.task[0]))
        else:
            last_observation = [0] * self.n_observations
            observations.append(last_observation)
        return observations, rewards, None


if __name__ == '__main__':
    env = Env()
    env.seed(config.RANDOM_SEED)
    env.reset()