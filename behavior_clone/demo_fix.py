import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from itertools import count
from env_fix import Env
import config
import csv

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env()
env.seed(config.RANDOM_SEED)
torch.manual_seed(0)
num_state = env.n_observations
num_action = env.node_num

Transition = namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'done', 'hidden', 'next_hidden']
)

def expert_policy(env, task, t_on_n):
    best_score = -float('inf')
    best_node = -1
    alpha = config.e1
    beta = config.e2

    for idx, node in enumerate(env.node):
        if task.mem > node.mem:
            continue
        if node.cpu_freq <= 0:
            continue
        if node.image_list[task.image_id] == 0 and env.image[task.image_id].image_size > node.disk:
            continue

        if node.image_list[task.image_id] == 2:
            download_finish_time = env.time
        elif node.image_list[task.image_id] == 1:
            download_finish_time = node.image_download_time[task.image_id]
        else:
            start_download = max(env.time, node.download_finish_time)
            download_time = env.image[task.image_id].image_size / (2 * node.bandwidth)
            download_finish_time = start_download + download_time

        current_tasks = t_on_n[idx]
        cpu_alloc = node.cpu_freq / (current_tasks + 1)
        comp_time = task.cpu_freq / cpu_alloc

        trans_rate = env.uplink_trans_rate(task, node, current_tasks + 1)
        if trans_rate <= 0:
            continue
        trans_time = task.task_size / trans_rate

        finish_time = download_finish_time + comp_time + trans_time
        delay = task.ddl - finish_time

        comp_energy = node.energy_ratio * comp_time * cpu_alloc / node.total_cpu_freq
        trans_energy = trans_time * task.transmission_energy
        total_energy = comp_energy + trans_energy

        score = beta * delay - alpha * total_energy

        if score > best_score:
            best_score = score
            best_node = idx

    return best_node

def main():
    total_times = []
    total_energys = []
    record = []
    total_download_time = []
    save_interval = 10
    save_filename = 'log/expert_policy' + str(config.e1) + str(config.e2) + 'node' + str(config.EDGE_NODE_NUM) + 'cpu' + str(config.node_cpu_freq_max) + 'task' + str(config.max_tasks) + '_' + str(config.min_tasks) + str(config.epoch) + '.csv'

    i_epoch = 0
    while i_epoch < config.epoch:
        ep_reward = []
        env.reset()

        cnt = 0
        failed = False
        for t in count():
            done, _, idx = env.env_up()
            if done:
                if not failed:
                    i_epoch += 1
                    total_times.append(env.total_time)
                    total_energys.append(env.total_energy)
                    total_download_time.append(env.download_time)
                    num_on_time = env.num_on_time
                    total_task = env.total_task
                    complete_ratio = num_on_time / total_task

                    print('Episode: {}, reward: {}, total_time: {}, total_energy: {}, complete_ratio: {}, download time: {}'.format(
                        i_epoch, round(np.mean(ep_reward), 3), env.total_time, env.total_energy, complete_ratio,
                        env.download_time))

                    record.append([i_epoch, round(np.mean(ep_reward), 3), env.total_time,
                                   env.total_energy, complete_ratio, env.download_time])
                break

            temp = 0
            actions = []
            states = []
            t_on_n = [0] * env.node_num
            tasks = []
            while env.task and env.task[0].start_time == env.time:
                temp += 1
                curr_task = env.task.pop(0)
                state = env.get_obs(curr_task)

                action = expert_policy(env, curr_task, t_on_n)
                if action == -1:
                    failed = True
                    break

                states.append(state)
                tasks.append(curr_task)
                actions.append(action)

            if failed:
                break

            for n_id in actions:
                t_on_n[n_id] += 1
            if tasks:
                next_states, rewards, download_finish_time = env.step(tasks, actions, t_on_n, states)
                for i in range(0, len(rewards)):
                    cnt += 1
                    reward = rewards[i]
                    ep_reward.append(reward)

        if i_epoch % save_interval == 0 and i_epoch > 0:
            with open(save_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time"])
                writer.writerows(record)

    with open(save_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time"])
        writer.writerows(record)

if __name__ == '__main__':
    main()
