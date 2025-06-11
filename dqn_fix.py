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
import math

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env()
env.seed(config.RANDOM_SEED)
torch.manual_seed(0)
num_state = env.n_observations
num_action = env.node_num

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'done'])

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.action_head = nn.Linear(32, num_action)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.action_head(x)
        return x

class DQN():
    buffer_capacity = 20000
    minimal_size = 2000
    batch_size = 128
    epsilon = 0.01

    def __init__(self):
        self.buffer = []
        self.q_net = Qnet().to(device)
        self.target_q_net = Qnet().to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.gamma = 0.98
        self.tau = 0.005

    def select_action(self, env, task, state):
        if np.random.random() < self.epsilon:
            random_numbers = [random.random() for _ in range(num_action)]
            count = 0
            for i in range(env.node_num):
                flag = env.image[task.image_id].image_size > env.node[i].disk and env.node[i].image_list[task.image_id] == 0
                if flag or task.mem > env.node[i].mem or env.node[i].cpu_freq <= 0:
                    count += 1
                    random_numbers[i] = -math.inf
                    if count == num_action:
                        print("DQN can't find fit action")
                        return -1, 0
            action = np.array(random_numbers).argmax().item()
        else:
            state = torch.tensor([state], dtype=torch.float).to(device)
            with torch.no_grad():
                action_q = self.q_net(state)
            count = 0
            for i in range(env.node_num):
                flag = env.image[task.image_id].image_size > env.node[i].disk and env.node[i].image_list[task.image_id] == 0
                if flag or task.mem > env.node[i].mem or env.node[i].cpu_freq <= 0:
                    count += 1
                    action_q[0, i] = -math.inf
                    if count == num_action:
                        print("DQN can't find fit action")
                        return -1, 0
            action = action_q.argmax().item()
        return action, 0

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def store_transition(self, transition):
        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(transition)
        else:
            self.buffer.pop(0)
            self.buffer.append(transition)
        return len(self.buffer) % self.minimal_size == 0

    def update(self):
        tiny_batch = random.sample(self.buffer, self.batch_size)
        states = torch.tensor([t.state for t in tiny_batch], dtype=torch.float).to(device)
        actions = torch.tensor([t.action for t in tiny_batch], dtype=torch.long).view(-1, 1).to(device)
        rewards = torch.tensor([t.reward for t in tiny_batch], dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor([t.next_state for t in tiny_batch], dtype=torch.float).to(device)
        dones = torch.tensor([t.done for t in tiny_batch], dtype=torch.float).view(-1, 1).to(device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.q_net, self.target_q_net)

def main(env):
    agent = DQN()
    record = []
    fail_time = 0
    i_epoch = 0
    save_interval = 10
    save_filename = 'log/fcDQN' + str(config.e1) + str(config.e2) + 'node' + str(config.EDGE_NODE_NUM) + 'cpu' + str(
        config.node_cpu_freq_max) + 'task' + str(config.max_tasks) + '_' + str(config.min_tasks) + '.csv'

    while i_epoch < config.epoch:
        ep_reward = []
        env.reset()
        cnt = 0
        fail = False

        for t in count():
            done, _, _ = env.env_up()
            if done:
                i_epoch += 1
                complete_ratio = env.num_on_time / env.total_task
                print(f'Episode: {i_epoch}, reward: {round(np.mean(ep_reward), 3)}, total_time: {env.total_time}, '
                      f'total_energy: {env.total_energy}, complete_ratio: {complete_ratio}, download: {env.download_time}')
                record.append([i_epoch, round(np.mean(ep_reward), 3), env.total_time,
                               env.total_energy, complete_ratio, env.download_time])
                break

            temp = 0
            actions, states, action_probs, tasks = [], [], [], []
            t_on_n = [0] * env.node_num

            while env.task and env.task[0].start_time == env.time:
                temp += 1
                task = env.task.pop(0)
                state = env.get_obs(task)
                action, action_prob = agent.select_action(env, task, state)
                if action == -1:
                    fail = True
                    break
                tasks.append(task)
                states.append(state)
                actions.append(action)
                action_probs.append(action_prob)

            if fail:
                fail_time += 1
                # i_epoch -= 1
                break

            for n_id in actions:
                t_on_n[n_id] += 1

            next_states, rewards, download_finish_time = env.step(tasks, actions, t_on_n, states)
            dones = [len(env.task) == 0] * temp
            for i in range(temp):
                cnt += 1
                ep_reward.append(rewards[i])
                trans = Transition(states[i], actions[i], rewards[i], action_probs[i], next_states[i], dones[i])
                agent.store_transition(trans)
                if cnt > 3000 and cnt % 1000 == 0:
                    agent.update()

        if i_epoch % save_interval == 0 and i_epoch > 0:
            with open(save_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time", "Fail Times"])
                for row in record:
                    row.append(fail_time)
                    writer.writerow(row)

    with open(save_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time", "Fail Times"])
        for row in record:
            row.append(fail_time)
            writer.writerow(row)

if __name__ == '__main__':
    main(env)
