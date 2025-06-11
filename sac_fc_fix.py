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
    ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'done']
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


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
        action_prob = F.softmax(x, dim=1, dtype=torch.float)
        return action_prob


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
    buffer_capacity = 30000
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
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
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

    def calc_target(self, rewards, next_states, dones):
        next_probs= self.actor(next_states)

        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def select_action(self, env, task, state):

        state = torch.tensor([state], dtype=torch.float).to(device)
        action_mask = torch.tensor([1] * env.node_num, dtype=torch.float).to(device)
        count = 0
        for i in range(env.node_num):
            flag = env.image[task.image_id].image_size > env.node[i].disk and env.node[i].image_list[task.image_id] == 0
            if flag or task.mem > env.node[i].mem or env.node[i].cpu_freq <= 0:
                count += 1
                action_mask[i] = 0
                if count == num_action:
                    print("sac_fc can't find fit action")
                    print('image:', flag)
                    print('memory:', task.mem > env.node[i].mem)
                    print('cpu:', env.node[i].cpu_freq <= 0)
                    return -1, 0
                    # exit()

        with torch.no_grad():
            action_prob= self.actor(state)
            action_prob = torch.mul(action_prob, action_mask)

        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample().item()

        return action, 0

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
        actions = torch.tensor([t.action for t in tiny_batch]).view(-1, 1).to(device)
        rewards = torch.tensor([t.reward for t in tiny_batch], dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor([t.next_state for t in tiny_batch], dtype=torch.float).to(device)
        dones = torch.tensor([t.done for t in tiny_batch], dtype=torch.float).view(-1, 1).to(device)
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update the policy network
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # Calculate entropy directly based on probabilities
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # Calculate expectation directly based on probabilities
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the alpha value
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


def main():
    agent = SAC()

    total_times = []
    total_energys = []
    record = []
    total_download_time = []
    save_interval = 10
    fail_time = 0
    save_filename = 'log/fcSAC' + str(config.e1) + str(config.e2) + 'node' + str(config.EDGE_NODE_NUM) + 'cpu' + str(
        config.node_cpu_freq_max) + 'task'+str(config.max_tasks)+'_'+str(config.min_tasks)+'.csv'
    i_epoch=0
    while i_epoch < config.epoch:
        ep_reward = []
        env.reset()

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
                print(
                    'Episode: {}, reward: {}, total_time: {}, total_energy: {}, complet_ratio: {}, donwload time: {}'.format(
                        i_epoch, round(np.mean(ep_reward), 3), env.total_time, env.total_energy, complete_ratio,
                        env.download_time))

                record.append([i_epoch, round(np.mean(ep_reward), 3), env.total_time,
                               env.total_energy, complete_ratio, env.download_time])
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
                states.append(state)
                tasks.append(curr_task)
                dones.append(len(env.task)==0)
                action, action_prob = agent.select_action(env, curr_task, state)
                if action ==-1:
                    fail = True
                    break
                actions.append(action)
                action_probs.append(action_prob)
            if fail == True:
                fail_time+=1
                i_epoch -= 1
                break
            for n_id in actions:
                t_on_n[n_id] += 1
            next_states, rewards, download_finish_time = env.step(tasks, actions, t_on_n, states)

            for i in range(0, temp):
                cnt += 1
                reward = rewards[i]
                ep_reward.append(reward)
                trans = Transition(states[i], actions[i], reward, action_probs[i], next_states[i],  dones[i])
                agent.store_transition(trans)
                if cnt > 3000:
                    if cnt % 1000 == 0:
                        agent.update()

        # Check if the save interval is reached, and if so save the data
        if i_epoch % save_interval == 0 and i_epoch > 0:
            with open(save_filename, mode='w', newline='') as file:
                writer = csv.writer(file)

                # Write to the header line of the CSV file
                writer.writerow(
                    ["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time",
                     "Fail Times"])

                # Write data in memory to CSV file
                writer.writerows(record)

    with open(save_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write to the header line of the CSV file
        writer.writerow(
            ["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time",
             "Fail Times"])
        record[0].append(fail_time)

        # Write data in memory to CSV file
        writer.writerows(record)


if __name__ == '__main__':
    main()
