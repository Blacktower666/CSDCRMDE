import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import csv
from collections import namedtuple
from itertools import count

from env_fix import Env
import config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env()
env.seed(config.RANDOM_SEED)
torch.manual_seed(0)
num_state = env.n_observations
num_action = env.node_num

Transition = namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'done']
)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.action_head = nn.Linear(32, num_action)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.action_head(x)
        action_prob = F.softmax(x, dim=-1)
        return action_prob

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 16)
        self.state_value = nn.Linear(16, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.state_value(x)
        return value

class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity = 4000
    batch_size = 4000

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().to(device)
        self.critic_net = Critic().to(device)
        self.buffer = []
        self.counter = 0
        self.gamma = 0.98
        self.lmbda = 0.95

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-5)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-4)

    def select_action(self, env, task, state):
        state = torch.tensor([state], dtype=torch.float).to(device)
        action_mask = torch.ones(num_action, dtype=torch.float).to(device)
        count = 0
        for i in range(num_action):
            flag = env.image[task.image_id].image_size > env.node[i].disk and env.node[i].image_list[task.image_id] == 0
            if flag or task.mem > env.node[i].mem or env.node[i].cpu_freq <= 0:
                action_mask[i] = 0
                count += 1
        if count == num_action:
            print("fc_ppo can't find fit action")
            return -1, 0

        with torch.no_grad():
            action_prob = self.actor_net(state).squeeze(0)
            action_prob = action_prob * action_mask
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample().item()
        return action, 0

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float).to(device)

    def update(self):
        states = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        next_states = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float).to(device)
        dones = torch.tensor([t.done for t in self.buffer], dtype=torch.float).to(device)
        rewards = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).to(device)
        actions = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)

        action_prob = self.actor_net(states)
        old_action_log_probs = torch.log(action_prob.gather(1, actions)).detach()

        actor_losses = []
        critic_losses = []
        for _ in range(self.ppo_epoch):
            td_target = rewards + self.gamma * self.critic_net(next_states).squeeze(1) * (1 - dones)
            td_target = td_target.unsqueeze(1)
            v_values = self.critic_net(states)
            td_delta = td_target - v_values
            advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta)

            action_prob = self.actor_net(states)
            action_log_probs = torch.log(action_prob.gather(1, actions))

            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(v_values, td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_net_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_net_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.buffer.clear()
        return np.mean(actor_losses), np.mean(critic_losses)

def main():
    agent = PPO()

    total_times = []
    total_energys = []
    record = []
    total_download_time = []
    save_interval = 10
    save_filename = (
        f"log/fcPPO_e1{config.e1}_e2{config.e2}_node{config.EDGE_NODE_NUM}"
        f"_cpu{config.node_cpu_freq_max}_task{config.max_tasks}_{config.min_tasks}_{config.epoch}.csv"
    )

    i_epoch = 0
    while i_epoch < config.epoch:
        ep_reward = []
        env.reset()
        episode_actor_losses = []
        episode_critic_losses = []

        cnt = 0
        fail = False
        for t in count():
            done, _, _ = env.env_up()
            if done:
                i_epoch += 1
                total_times.append(env.total_time)
                total_energys.append(env.total_energy)
                total_download_time.append(env.download_time)
                num_on_time = env.num_on_time
                total_task = env.total_task
                complete_ratio = num_on_time / total_task

                avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0
                avg_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0

                print('Episode: {}, reward: {}, total_time: {}, total_energy: {}, complet_ratio: {}, download time: {}, actor_loss: {:.6f}, critic_loss: {:.6f}'.format(
                    i_epoch, round(np.mean(ep_reward), 3), env.total_time, env.total_energy, complete_ratio,
                    env.download_time, avg_actor_loss, avg_critic_loss))
                record.append([i_epoch, round(np.mean(ep_reward), 3), env.total_time,
                               env.total_energy, complete_ratio, env.download_time, avg_actor_loss, avg_critic_loss])
                break

            temp = 0
            actions = []
            states = []
            action_probs = []
            t_on_n = [0] * env.node_num
            tasks = []
            dones = []
            while env.task and env.task[0].start_time == env.time:
                temp += 1
                curr_task = env.task.pop(0)
                state = env.get_obs(curr_task)
                action, action_prob = agent.select_action(env, curr_task, state)
                states.append(state)
                tasks.append(curr_task)
                dones.append(len(env.task) == 0)
                actions.append(action)
                action_probs.append(action_prob)
                if action == -1:
                    dones[-1] = 1
                    fail = True
                    break

            if fail:
                for n_id in actions:
                    if n_id >= 0:
                        t_on_n[n_id] += 1
                next_states, rewards, _ = env.step(tasks[:-1], actions[:-1], t_on_n, states[:-1])
                for i in range(0, temp - 1):
                    cnt += 1
                    reward = rewards[i]
                    ep_reward.append(reward)
                    trans = Transition(states[i], actions[i], reward, action_probs[i], next_states[i], dones[i])
                    agent.store_transition(trans)
                if len(states) >= 2:
                    trans = Transition(states[-2], actions[-2], -99999, action_probs[-2], [0] * num_state, dones[-2])
                    cnt += 1
                    agent.store_transition(trans)
                if cnt > agent.buffer_capacity:
                    a_loss, c_loss = agent.update()
                    episode_actor_losses.append(a_loss)
                    episode_critic_losses.append(c_loss)
                ep_reward.append(-99999)
                break

            for n_id in actions:
                if n_id >= 0:
                    t_on_n[n_id] += 1
            next_states, rewards, _ = env.step(tasks, actions, t_on_n, states)
            for i in range(temp):
                cnt += 1
                reward = rewards[i]
                ep_reward.append(reward)
                trans = Transition(states[i], actions[i], reward, action_probs[i], next_states[i], dones[i])
                agent.store_transition(trans)
                if agent.counter >= agent.buffer_capacity and agent.counter % 2000 == 0:
                    a_loss, c_loss = agent.update()
                    episode_actor_losses.append(a_loss)
                    episode_critic_losses.append(c_loss)

        if i_epoch % save_interval == 0 and i_epoch > 0:
            with open(save_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time", "Actor Loss", "Critic Loss"])
                writer.writerows(record)

    with open(save_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time", "Actor Loss", "Critic Loss"])
        writer.writerows(record)

if __name__ == '__main__':
    main()
