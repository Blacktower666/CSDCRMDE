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


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


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
        x = torch.cat((x_node, x_task), dim=2)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.action_head(x)
        action_prob = F.softmax(x, dim=2)
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
        # Policy Network
        self.actor = Actor().to(device)
        # First Q-network
        self.critic_1 = Critic().to(device)
        # Second Q-network
        self.critic_2 = Critic().to(device)
        self.target_critic_1 = Critic().to(device)  # First target Q-network
        self.target_critic_2 = Critic().to(device)  # Second target Q-network
        # Initialize target Q-networks with the same parameters as Q-networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)
        # Use the log value of alpha for more stable training results
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # Allow gradient computation for alpha
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)
        self.target_entropy = -1  # Target entropy level
        self.gamma = 0.98
        self.tau = 0.005  # Soft update rate
        self.target_update = 100  # Target network update frequency
        self.count = 0  # Counter to keep track of update iterations

    def get_initial_states(self):
        h_0 = torch.zeros((self.actor.gru.num_layers, 1, self.actor.gru.hidden_size), dtype=torch.float)
        h_0 = h_0.to(device)
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
            action_prob = action_prob.squeeze(1)
            action_prob = action_prob.squeeze(0)
            action_prob = torch.mul(action_prob, action_mask)
        action_dist = torch.distributions.Categorical(action_prob)
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

        q_values1 = self.critic_1(states).squeeze(1)  # [batch_size, num_action]
        critic_1_q_values = q_values1.gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))

        q_values2 = self.critic_2(states).squeeze(1)  # [batch_size, num_action]
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
        baseline = (probs.detach() * min_qvalue.detach()).sum(dim=1, keepdim=True)  # [batch, 1]

        advantage = min_qvalue - baseline  # [batch, num_action]
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        actor_loss = torch.sum(probs * (-self.log_alpha.exp() * log_probs - advantage), dim=1).mean()





        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        print(f"[SAC] entropy: {entropy.mean().item():.4f}, minQ: {min_qvalue.mean().item():.4f}, actor_loss: {actor_loss.item():.4f}")
        return actor_loss.item(), (critic_1_loss.item() + critic_2_loss.item()) / 2.0


def main():
    agent = SAC()

    total_times = []
    total_energys = []
    record = []
    total_download_time = []
    save_interval = 10
    fail_time = 0

    save_filename = 'log/gruSAC' + str(config.e1) + str(config.e2) + 'node' + str(config.EDGE_NODE_NUM) + 'cpu' + str(config.node_cpu_freq_max) + 'task' + str(config.max_tasks) + '_' + str(config.min_tasks) + str(config.epoch) + '.csv'

    i_epoch = 0
    while i_epoch < config.epoch:
        ep_reward = []
        env.reset()

        episode_actor_losses = []
        episode_critic_losses = []

        cnt = 0
        fail = False
        for t in count():
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
            t_on_n = [0] * env.node_num
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
            if fail == True:
                for n_id in actions:
                    t_on_n[n_id] += 1
                next_states, rewards, download_finish_time = env.step(tasks[:-1], actions[:-1], t_on_n[:-1], states[:-1])
                for i in range(0, temp - 1):
                    cnt += 1
                    reward = rewards[i]
                    ep_reward.append(reward)
                    trans = Transition(states[i], actions[i], reward, action_probs[i], next_states[i], dones[i], hiddens[i], next_hiddens[i])
                    agent.store_transition(trans)
                    if cnt > 3000:
                        a_loss, c_loss = agent.update()
                        episode_actor_losses.append(a_loss)
                        episode_critic_losses.append(c_loss)
                if len(states) >= 2:
                    trans = Transition(states[-2], actions[-2], -999999, action_probs[-2], [0] * env.n_observations, dones[-2], hiddens[-2], next_hiddens[-2])
                    cnt += 1
                    agent.store_transition(trans)
                    if cnt > 3000:
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

                avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0
                avg_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0

                print('Episode: {}, reward: {}, total_time: {}, total_energy: {}, complet_ratio: {}, download time: {}, actor_loss: {:.6f}, critic_loss: {:.6f}'.format(
                    i_epoch, round(np.mean(ep_reward), 3), env.total_time, env.total_energy, complete_ratio,
                    env.download_time, avg_actor_loss, avg_critic_loss))
                record.append([i_epoch, round(np.mean(ep_reward), 3), env.total_time,
                               env.total_energy, complete_ratio, env.download_time, avg_actor_loss, avg_critic_loss])
                break

            for n_id in actions:
                t_on_n[n_id] += 1
            next_states, rewards, download_finish_time = env.step(tasks, actions, t_on_n, states)
            for i in range(0, temp):
                cnt += 1
                reward = rewards[i]
                ep_reward.append(reward)
                trans = Transition(states[i], actions[i], reward, action_probs[i], next_states[i], dones[i], hiddens[i], next_hiddens[i])
                agent.store_transition(trans)
                if cnt > 18000:
                    if cnt % 2000 == 0:
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
