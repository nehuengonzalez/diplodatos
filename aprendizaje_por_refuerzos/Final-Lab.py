import gym
import itertools
import math
import numpy as np
import os.path
import pandas as pd
import random
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from agents.utils.memory.ReplayMemory import ReplayMemory
from agents.utils.memory.Transition import Transition
from collections import deque

warnings.simplefilter('ignore')


class DQNSolver:
    def __init__(self, env, model, target, double=False, n_episodes=3000, max_env_steps=None, gamma=0.9,
                 epsilon=0.5, epsilon_min=0.05, epsilon_log_decay=0.001, alpha=1e-3,
                 memory_size=10000, batch_size=256, c=10, hidden_layers=1, hidden_size=6,
                 render=False, debug=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.target = target.to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.model.train()

        self.double = double

        self.memory = ReplayMemory(capacity=memory_size)
        self.env = env

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.c = c
        if max_env_steps is not None:
            self.env._max_episode_steps = max_env_steps

        self.render = render
        self.debug = debug
        if debug:
            self.loss_list = []

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

    def choose_action(self, state, epsilon):
        """Chooses the next action according to the model trained and the policy"""

        # exploits the current knowledge if the random number > epsilon, otherwise explores
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q = self.model(state)
                argmax = torch.argmax(q)
                return argmax.item()

    def get_epsilon(self, episode):
        """Returns an epsilon that decays over time until a minimum epsilon value is reached; in this case the minimum
        value is returned"""
        return max(self.epsilon_min, self.epsilon * math.exp(-self.epsilon_decay * episode))

    def replay(self):
        """Previously stored (s, a, r, s') tuples are replayed (that is, are added into the model). The size of the
        tuples added is determined by the batch_size parameter"""

        transitions, _ = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        current_tensors = [s for s in batch.next_state if s is not None]
        if current_tensors:
            non_final_next_states = torch.stack(current_tensors)
        else:
            return

        non_final_mask = torch.stack(batch.done)
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            if not self.double:
                next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()
            else:
                non_final_best_actions_index = torch.argmax(self.model(non_final_next_states), 1).unsqueeze(1)
                next_state_values[non_final_mask] = self.target(non_final_next_states). \
                    gather(1, non_final_best_actions_index).squeeze(1).detach()
            expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute loss
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        if self.debug:
            self.loss_list.append(loss)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def run(self):
        """Main loop that controls the execution of the agent"""

        scores = []
        mean_scores = []
        j = 0  # used for model2 update every c steps
        for e in range(self.n_episodes):
            state = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float)
            done = False
            cum_reward = 0
            while not done:
                action = self.choose_action(
                    state,
                    self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(next_state, device=self.device, dtype=torch.float)

                cum_reward += reward
                self.memory.push(
                    state,  # Converted to tensor in choose_action method
                    torch.tensor([action], device=self.device),
                    None if done else next_state,
                    torch.tensor(reward, device=self.device).clamp_(-1, 1),
                    torch.tensor(not done, device=self.device, dtype=torch.bool))

                if self.memory.__len__() >= self.batch_size:
                    self.replay()

                state = next_state
                j += 1

                # update second model
                if j % self.c == 0:
                    self.target.load_state_dict(self.model.state_dict())
                    self.target.eval()

            scores.append(cum_reward)
            mean_score = np.mean(scores)
            mean_scores.append(mean_score)
            if e % 100 == 0 and self.debug:
                print('[Episode {}] - Mean reward {}.'.format(e, mean_score))

        # noinspection PyUnboundLocalVariable
        print('[Episode {}] - Mean reward {}.'.format(e, mean_score))
        return scores, mean_scores

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)


class DQN(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers=2, hidden_size=24, init=True):
        super(DQN, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)

        self.hidden = []
        for i in range(hidden_layers):
            layer = nn.Linear(hidden_size, hidden_size)
            self.add_module('h' + str(i), layer)
            self.hidden.append(layer)

        self.output = nn.Linear(hidden_size, output_size)

        if init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.input(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.output(x)
        return x


class DuelingDQN(DQN):

    def __init__(self, input_size, output_size, hidden_layers=1, hidden_size=6, v_adv_layers=0, v_adv_size=6):
        super(DuelingDQN, self).__init__(input_size, hidden_size, hidden_layers, hidden_size, False)
        self.output_size = output_size

        self.adv_input = nn.Linear(hidden_size, v_adv_size)
        self.v_input = nn.Linear(hidden_size, v_adv_size)

        self.adv_layers = []
        for i in range(v_adv_layers):
            layer = nn.Linear(v_adv_size, v_adv_size)
            self.add_module('adv' + str(i), layer)
            self.adv_layers.append(layer)

        self.v_layers = []
        for i in range(v_adv_layers):
            layer = nn.Linear(v_adv_size, v_adv_size)
            self.add_module('v' + str(i), layer)
            self.v_layers.append(layer)

        self.adv_output = nn.Linear(v_adv_size, output_size)
        self.v_output = nn.Linear(v_adv_size, 1)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.relu(super(DuelingDQN, self).forward(x))

        adv = F.relu(self.adv_input(x))
        v = F.relu(self.v_input(x))

        for layer in self.adv_layers:
            adv = F.relu(layer(adv))
        for layer in self.v_layers:
            v = F.relu(layer(v))

        adv = self.adv_output(adv)

        if len(adv.size()) > 1:
            # Model used with state single instance
            v = self.v_output(v).expand(adv.size(0), self.output_size)
            mean = adv.mean(1).unsqueeze(1).expand(adv.size(0), self.output_size)
        else:
            # Model used with sta
            v = self.v_output(v).expand(self.output_size)
            mean = adv.mean().unsqueeze(0).expand(self.output_size)

        x = v + adv - mean

        return x


if __name__ == "__main__":

    if not os.path.exists("results.csv"):
        with open("results.csv", "w+") as f:
            f.write("Episode,Model,hidden_layers,layer_size,adv_layers,score,mean\n")

    EPISODES = 20
    DEBUG = True
    BATCH_SIZE = 2

    env = gym.make('CartPole-v0')
    observation_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n

    for hidden_layers in [1, 2, 3]:
        for layer_size in [6, 12, 24]:
            for adv_layers in [0, 1, 2]:
                env.reset()
                model = DuelingDQN(observation_space_size, action_space_size, hidden_layers=hidden_layers,
                                   hidden_size=layer_size, v_adv_layers=adv_layers, v_adv_size=layer_size)
                target = DuelingDQN(observation_space_size, action_space_size, hidden_layers=hidden_layers,
                                    hidden_size=layer_size, v_adv_layers=adv_layers, v_adv_size=layer_size)

                agent = DQNSolver(env, model, target, double=False, n_episodes=EPISODES, debug=DEBUG,
                                  batch_size=BATCH_SIZE)
                scoresDuelingDQN_i, meanDuelingDQN_i = agent.run()

                with open("results.csv", "a+") as f:
                    for i in range(len(scoresDuelingDQN_i)):
                        f.write("%d,%s,%d,%d,%d,%f,%f\n" % (i,
                                                            "Dueling-DQN",
                                                            hidden_layers,
                                                            layer_size,
                                                            adv_layers,
                                                            scoresDuelingDQN_i[i],
                                                            meanDuelingDQN_i[i]))
