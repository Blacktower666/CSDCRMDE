import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import random
import pickle
from env_fix import Env
import config
import datetime

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env()
env.seed(config.RANDOM_SEED)
env.reset()
num_state = env.n_observations
num_action = env.node_num

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

def hybrid_policy(env, task, t_on_n):
    expert_action = expert_policy(env, task, t_on_n)
    return expert_action, 0.0

def main():
    NUM_EPISODES = 10
    offline_data = []

    print("Start offline data collection (each time slot is a sequence)...")

    all_episode_rewards = []

    for ep in range(NUM_EPISODES):
        env.reset()
        episode_sequences = []
        episode_rewards = []

        while True:
            done, _, _ = env.env_up()
            if done:
                break

            t_on_n = [0] * num_action
            tasks = []
            time_slot_sequence = []

            while env.task and env.task[0].start_time == env.time:
                curr_task = env.task.pop(0)
                state = env.get_obs(curr_task)
                action, logp = hybrid_policy(env, curr_task, t_on_n)

                if action >= 0:
                    t_on_n[action] += 1

                time_slot_sequence.append((state, action))
                tasks.append(curr_task)

            if tasks:
                next_states, rewards, _ = env.step(tasks, [a for _, a in time_slot_sequence], t_on_n, [s for s, _ in time_slot_sequence])
                episode_rewards.extend(rewards)

            if time_slot_sequence:
                episode_sequences.append(time_slot_sequence)

        num_tasks = sum(len(ts) for ts in episode_sequences)

        total_ep_reward = sum(episode_rewards) if episode_rewards else 0
        all_episode_rewards.append(total_ep_reward)

        print(f"Episode {ep+1} finished, Time Slot count: {len(episode_sequences)}, "
              f"Total tasks: {num_tasks}, Total reward: {total_ep_reward:.2f}")

        offline_data.append(episode_sequences)

    avg_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0
    print(f"Average total reward: {avg_reward:.2f}")

    filename = (
        f"offline_data_seq_n{config.EDGE_NODE_NUM}_cpu{config.node_cpu_freq_max}_"
        f"task{config.max_tasks}-{config.min_tasks}_ep{NUM_EPISODES}.pkl"
    )
    with open(filename, "wb") as f:
        pickle.dump(offline_data, f)

    print(f"Data collection complete, saved to: {filename}")

if __name__ == '__main__':
    main()
