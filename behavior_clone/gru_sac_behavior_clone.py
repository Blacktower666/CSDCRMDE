import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from itertools import count
import pickle
import csv
from torch.distributions import Categorical
import datetime

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
    ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'done', 'hidden', 'next_hidden']
)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.task_emb = nn.Linear(5, 64)
        self.node_emb = nn.Linear(num_state - 5, 64)
        self.gru = nn.GRU(64, 64, 1, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.action_head = nn.Linear(32, num_action)

    def forward(self, x, hidden):
        x_task = x[:, :, -5:]
        x_node = x[:, :, :-5]
        x_task = self.task_emb(x_task)
        x_node = self.node_emb(x_node)
        x_node, h = self.gru(x_node, hidden)
        x_cat = torch.cat((x_node, x_task), dim=2)
        x_cat = F.leaky_relu(self.fc1(x_cat))
        x_cat = F.leaky_relu(self.fc2(x_cat))
        x_out = self.action_head(x_cat)
        action_prob = F.softmax(x_out, dim=2)
        return action_prob, h

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 16)
        self.state_value = nn.Linear(16, num_action)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.state_value(x)
        return value

def get_initial_states(batch_size):
    h_0 = torch.zeros((1, batch_size, 64), dtype=torch.float).to(device)
    return h_0


def evaluate_policy(actor, num_episodes=3):
    actor.eval()
    total_rewards = []
    for ep in range(num_episodes):
        env.reset()
        ep_reward = []
        hidden = get_initial_states(1)
        while True:
            done, _, _ = env.env_up()
            if done:
                break
            tasks = []
            states = []
            actions = []
            while env.task and env.task[0].start_time == env.time:
                curr_task = env.task.pop(0)
                state = env.get_obs(curr_task)
                tasks.append(curr_task)
                states.append(state)
                state_tensor = torch.tensor([[state]], dtype=torch.float).to(device)
                action_prob, hidden = actor(state_tensor, hidden)
                action_prob = action_prob.squeeze(0).squeeze(0)
                action_mask = torch.ones(num_action, dtype=torch.float).to(device)
                count = 0
                for i in range(num_action):
                    flag = env.image[curr_task.image_id].image_size > env.node[i].disk and \
                           env.node[i].image_list[curr_task.image_id] == 0
                    if flag or curr_task.mem > env.node[i].mem or env.node[i].cpu_freq <= 0:
                        count += 1
                        action_mask[i] = 0
                if count == num_action:
                    action = -1
                else:
                    masked_prob = action_prob * action_mask
                    action = torch.argmax(masked_prob).item()
                actions.append(action)
            if tasks:
                t_on_n = [0] * num_action
                for a in actions:
                    if a >= 0:
                        t_on_n[a] += 1
                next_states, rewards, _ = env.step(tasks, actions, t_on_n, states)
                ep_reward.extend(rewards)
        total_rewards.append(np.mean(ep_reward) if len(ep_reward) > 0 else 0)
    actor.train()
    return np.mean(total_rewards)

def imitation_learning():
    with open("offline_data_seq_n15_cpu650_task20-5_ep10.pkl", "rb") as f:
        expert_data = pickle.load(f)

    actor = Actor().to(device)
    for param in actor.gru.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    num_epochs = config.epoch_imitation

    for epoch in range(num_epochs):
        epoch_losses = []
        random.shuffle(expert_data)

        for episode in expert_data:
            for time_slot_seq in episode:
                if len(time_slot_seq) == 0:
                    continue

                states = torch.tensor([s for s, _ in time_slot_seq], dtype=torch.float).unsqueeze(0).to(device)  # (1, T, obs_dim)
                actions = torch.tensor([a for _, a in time_slot_seq], dtype=torch.long).unsqueeze(0).to(device)  # (1, T)

                hidden = get_initial_states(batch_size=1)

                action_probs, _ = actor(states, hidden)  # (1, T, num_action)
                log_probs = torch.log(action_probs + 1e-8)
                loss = F.nll_loss(
                    log_probs.transpose(1, 2),  # → (1, num_action, T)
                    actions,                   # → (1, T)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        print(f"Imitation Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    model_filename = f"imitation_gru_sac_ep{config.epoch_imitation}_{timestamp}.pth"
    torch.save(actor.state_dict(), model_filename)
    print(f"Imitation learning finished. Model saved as '{model_filename}'.")

class SAC():
    clip_param = 0.2
    max_grad_norm = 0.5
    buffer_capacity = 22000
    minimal_size = 1500
    batch_size = 3000

    def __init__(self):
        super(SAC, self).__init__()
        self.buffer = []
        self.counter = 0
        self.actor = Actor().to(device)
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        model_filename = f"imitation_gru_sac_ep{config.epoch_imitation}_{timestamp}.pth"
        self.actor.load_state_dict(torch.load(model_filename))
        for param in self.actor.gru.parameters():
            param.requires_grad = True
        self.critic_1 = Critic().to(device)
        self.critic_2 = Critic().to(device)
        self.target_critic_1 = Critic().to(device)
        self.target_critic_2 = Critic().to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.target_entropy = -1
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.gamma = 0.98
        self.tau = 0.005
        self.target_update = 100
        self.count = 0

    def get_initial_states(self):
        h_0 = torch.zeros((self.actor.gru.num_layers, 1, self.actor.gru.hidden_size), dtype=torch.float).to(device)
        return h_0

    def calc_target(self, rewards, next_states, dones, next_hiddens):
        next_probs, _ = self.actor(next_states, next_hiddens)
        next_probs = next_probs.squeeze(1)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states).squeeze(1)
        q2_value = self.target_critic_2(next_states).squeeze(1)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def select_action(self, env, task, state, hidden):
        state = torch.tensor([[state]], dtype=torch.float).to(device)
        action_mask = torch.tensor([1] * env.node_num, dtype=torch.float).to(device)
        count = 0
        for i in range(env.node_num):
            flag = env.image[task.image_id].image_size > env.node[i].disk and env.node[i].image_list[task.image_id] == 0
            if flag or task.mem > env.node[i].mem or env.node[i].cpu_freq <= 0:
                count += 1
                action_mask[i] = 0
                if count == num_action:
                    action_prob, next_h = self.actor(state, hidden)
                    print("sac_gru can't find fit action")
                    print('image:', flag)
                    print('memory:', task.mem > env.node[i].mem)
                    print('cpu:', env.node[i].cpu_freq <= 0)
                    return -1, 0, next_h
        with torch.no_grad():
            action_prob, next_h = self.actor(state, hidden)
            action_prob = action_prob.squeeze(1).squeeze(0)
            action_prob = torch.mul(action_prob, action_mask)
        action_dist = Categorical(action_prob)
        action = action_dist.sample().item()
        return action, 0, next_h

    def store_transition(self, transition):
        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(transition)
        else:
            self.buffer.pop(0)
            self.buffer.append(transition)
        return len(self.buffer) % self.minimal_size == 0

    def update(self):
        tiny_batch = random.sample(self.buffer, self.batch_size)
        states = torch.tensor([[t.state] for t in tiny_batch], dtype=torch.float).to(device)
        actions = torch.tensor([[t.action] for t in tiny_batch], dtype=torch.int64).to(device)
        rewards = torch.tensor([t.reward for t in tiny_batch], dtype=torch.float).view(-1, 1).to(device)
        hiddens = torch.cat([t.hidden for t in tiny_batch], dim=1).to(device)
        next_hiddens = torch.cat([t.next_hidden for t in tiny_batch], dim=1).to(device)
        next_states = torch.tensor([[t.next_state] for t in tiny_batch], dtype=torch.float).to(device)
        dones = torch.tensor([t.done for t in tiny_batch], dtype=torch.float).view(-1, 1).to(device)
        td_target = self.calc_target(rewards, next_states, dones, next_hiddens)

        q_values1 = self.critic_1(states).squeeze(1)
        critic_1_q_values = q_values1.gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))

        q_values2 = self.critic_2(states).squeeze(1)
        critic_2_q_values = q_values2.gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        probs, next_hiddens = self.actor(states, hiddens)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)


        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        return actor_loss.item(), (critic_1_loss.item() + critic_2_loss.item()) / 2.0

def online_training():
    agent = SAC()
    record = []
    total_times = []
    total_energys = []
    total_download_time = []
    save_interval = 10
    fail_time = 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    save_filename = (
        f"log/gruBC_SAC_ep{config.epoch_imitation}_alpha{config.e2}_n{config.EDGE_NODE_NUM}"
        f"_cpu{config.node_cpu_freq_max}_task{config.max_tasks}-{config.min_tasks}"
        f"_{timestamp}.csv"
    )

    agent.buffer.clear()
    i_epoch = 0
    while i_epoch < config.epoch:
        ep_reward = []
        episode_actor_losses = []
        episode_critic_losses = []

        env.reset()
        cnt = 0
        fail = False
        while True:
            done, _, idx = env.env_up()
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
            t_on_n = [0] * num_action
            tasks = []
            hiddens = []
            next_hiddens = []
            dones = []
            hidden = agent.get_initial_states()
            while env.task and env.task[0].start_time == env.time:
                temp += 1
                curr_task = env.task.pop(0)
                state = env.get_obs(curr_task)
                action, action_prob, next_hidden = agent.select_action(env, curr_task, state, hidden)
                states.append(state)
                tasks.append(curr_task)
                hiddens.append(hidden)
                dones.append(len(env.task) == 0)
                next_hiddens.append(next_hidden)
                hidden = next_hidden
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
                next_states, rewards, download_finish_time = env.step(tasks[:-1], actions[:-1], t_on_n, states[:-1])
                for i in range(0, temp - 1):
                    cnt += 1
                    reward = rewards[i]
                    ep_reward.append(reward)
                    trans = Transition(states[i], actions[i], reward, action_probs[i], next_states[i], dones[i], hiddens[i], next_hiddens[i])
                    agent.store_transition(trans)
                if len(states) >= 2:
                    trans = Transition(states[-2], actions[-2], -999999, action_probs[-2], [0] * num_state, dones[-2], hiddens[-2], next_hiddens[-2])
                    cnt += 1
                    agent.store_transition(trans)
                if cnt > 3000:
                    # pass
                    a_loss, c_loss = agent.update()
                    episode_actor_losses.append(a_loss)
                    episode_critic_losses.append(c_loss)
                ep_reward.append(-999999)
                i_epoch += 1
                total_times.append(env.total_time)
                total_energys.append(env.total_energy)
                total_download_time.append(env.download_time)
                num_on_time = env.num_on_time
                total_task = env.total_task
                complete_ratio = num_on_time / total_task
                print('Episode: {}, reward: {}, total_time: {}, total_energy: {}, complet_ratio: {}, download time: {}, actor_loss: {:.6f}, critic_loss: {:.6f}'.format(
                    i_epoch, round(np.mean(ep_reward), 3), env.total_time, env.total_energy, complete_ratio,
                    env.download_time, np.mean(episode_actor_losses) if episode_actor_losses else 0, np.mean(episode_critic_losses) if episode_critic_losses else 0))
                record.append([i_epoch, round(np.mean(ep_reward), 3), env.total_time,
                               env.total_energy, complete_ratio, env.download_time, np.mean(episode_actor_losses) if episode_actor_losses else 0, np.mean(episode_critic_losses) if episode_critic_losses else 0])
                break

            for n_id in actions:
                if n_id >= 0:
                    t_on_n[n_id] += 1
            next_states, rewards, download_finish_time = env.step(tasks, actions, t_on_n, states)
            for i in range(0, temp):
                cnt += 1
                reward = rewards[i]
                ep_reward.append(reward)
                trans = Transition(states[i], actions[i], reward, action_probs[i], next_states[i], dones[i], hiddens[i], next_hiddens[i])
                agent.store_transition(trans)
                if cnt > 18000 and cnt % 2000 == 0:
                    # pass
                    a_loss, c_loss = agent.update()
                    episode_actor_losses.append(a_loss)
                    episode_critic_losses.append(c_loss)

        if i_epoch % save_interval == 0 and i_epoch > 0:
            with open(save_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time", "Actor Loss", "Critic Loss", "Fail Times"])
                writer.writerows(record)

    with open(save_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time", "Actor Loss", "Critic Loss", "Fail Times"])
        record[0].append(fail_time)
        writer.writerows(record)

def main():
    print("=== Phase 1: Imitation Learning (Behavior Cloning) ===")
    imitation_learning()
    print("=== Phase 2: Online Reinforcement Learning (SAC) ===")
    online_training()

if __name__ == '__main__':
    main()