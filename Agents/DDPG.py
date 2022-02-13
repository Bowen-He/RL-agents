import gym
import sys
import time
import numpy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import pickle

from Common.buffer_class import buffer_class as ReplayBuffer
from Common import utils_for_q_learning

class DDPGNet(nn.Module):
    def __init__(self, params, env, state_size, action_size, device):
        super(DDPGNet, self).__init__()
        self.env = env
        self.device = device
        self.params = params
        self.max_a = self.env.action_space.high[0]

        # Defaults to False (standard RBFDQN behavior)

        self.buffer_object = ReplayBuffer(
            max_length=self.params['max_buffer_size'],
            env=self.env,
        )

        self.state_size, self.action_size = state_size, action_size

        self.policy_module = self._make_policy_module()
        self.value_module = self._make_value_module()

        self.criterion = nn.MSELoss()

        self.policy_optimizer, self.value_optimizer = self._make_optimizers()
        self.optimizer = self._make_optimizers()
        self.to(self.device)

    def _make_optimizers(self):
        try:
            if self.params['optimizer'] == 'RMSprop':
                Constructor = optim.RMSprop
            elif self.params['optimizer'] == 'Adam':
                Constructor = optim.Adam
            else:
                print('unknown optimizer ....')
        except:
            print("no optimizer specified ... ")
        policy_optimizer = Constructor(
            self.policy_module.parameters(),
            lr=self.params['learning_rate_location_side'])
        value_optimizer = Constructor(
            self.value_module.parameters(),
            lr=self.params['learning_rate'])

        return policy_optimizer, value_optimizer

    def _make_policy_module(self):
        policy_network = nn.Sequential(
            nn.Linear(self.state_size, self.params['layer_size_action_side']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size_action_side'],
                      self.params['layer_size_action_side']), nn.ReLU(),
            nn.Linear(self.params['layer_size_action_side'], self.action_size),
            nn.Tanh())
        torch.nn.init.zeros_(policy_network[-2].bias)
        return policy_network

    def _make_value_module(self):
        value_network = nn.Sequential(
            nn.Linear(self.state_size + self.action_size,
                      self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], 1),
        )
        return value_network

    def _get_best_action(self, s):
        assert len(s.shape) == 2
        policy_module_output = self.policy_module(s)
        return policy_module_output * self.max_a

    def _get_sa_value(self, s, a):
        assert len(s.shape) == len(a.shape) == 2
        sa_concat = torch.cat([s, a], dim=1)
        assert s.shape[0] == sa_concat.shape[0]
        return self.value_module(sa_concat)

    def get_best_qvalue_and_action(self, s):
        '''
		given a batch of states s, return Q(s,a), max_{a} ([batch x 1], [batch x a_dim])
        We want to add target smoothing as an option. If it's enabled, then we actually want to
        get the actions, then smooth it, then forward that one. How's that work. Do I need
        to do this pluck thing? Probably.
        Now, we ALSO want to add something if there are two Q-networks. Annoying but necessary.
        Do we need to change the centroid_values and latent_centroids to work with CDQ as well?
        Looks like it.
        How can we make the action part work correctly?
        Get centroids. Make copy. Add noise to copy. Pass both through both latent modules.
        Get values for both noisy actions. Choose lower of the two. Return that value. Return the
        max actions from the first set always, in case we're doing selection.
        If we're doing action-selection, then we should actually just not do both. That should be an option.
        This is getting ugly and unmaintainable. What should I do?
        One option is sacrifice efficiency, by passing the centroids through latent again,
        whether they have noise added or not. That may be fine.
  		'''
        best_action = self._get_best_action(s)
        value = self._get_sa_value(s, best_action)
        if s.shape[0] == 1:
            # value = value.item()
            best_action = best_action[0]
        return value, best_action

    def forward(self, s, a, use_cdq=False):
        '''
		given a batch of s,a , compute Q(s,a) [batch x 1]
		'''
        return self._get_sa_value(s, a)

    def e_greedy_policy(self, s, episode, train_or_test):
        '''
		Given state s, at episode, take random action with p=eps if training 
		Note - epsilon is determined by episode
		'''
        epsilon = 1.0 / numpy.power(episode,
                                    1.0 / self.params['policy_parameter'])
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a.tolist()
        else:
            self.eval()
            s_matrix = numpy.array(s).reshape(1, self.state_size)
            with torch.no_grad():
                s = torch.from_numpy(s_matrix).float().to(self.device)
                _, a = self.get_best_qvalue_and_action(s)
                a = a.cpu().numpy()
            self.train()
            return a

    def e_greedy_gaussian_policy(self, s, episode, train_or_test):
        '''
		Given state s, at episode, take random action with p=eps if training 
		Note - epsilon is determined by episode
		'''
        epsilon = 1.0 / numpy.power(episode,
                                    1.0 / self.params['policy_parameter'])
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a.tolist()
        else:
            self.eval()
            s_matrix = numpy.array(s).reshape(1, self.state_size)
            with torch.no_grad():
                s = torch.from_numpy(s_matrix).float().to(self.device)
                _, a = self.get_best_qvalue_and_action(s)
                a = a.cpu().numpy()
            self.train()
            noise = numpy.random.normal(loc=0.0,
                                        scale=self.params['noise'],
                                        size=len(a))
            a = a + noise
            return a

    def gaussian_policy(self, s, episode, train_or_test):
        '''
		Given state s, at episode, take random action with p=eps if training 
		Note - epsilon is determined by episode
		'''
        self.eval()
        s_matrix = numpy.array(s).reshape(1, self.state_size)
        with torch.no_grad():
            s = torch.from_numpy(s_matrix).float().to(self.device)
            _, a = self.get_best_qvalue_and_action(s)
            a = a.cpu()
        self.train()
        noise = numpy.random.normal(loc=0.0,
                                    scale=self.params['noise'],
                                    size=len(a))
        a = a + noise
        return a

    def enact_policy(self, s, episode, train_or_test, policy_type="e_greedy"):
        assert policy_type in [
            "e_greedy", "e_greedy_gaussian", "gaussian", "softmax"
        ], f"Bad policy type: {policy_type}"
        polciy_types = {
            'e_greedy': self.e_greedy_policy,
            'e_greedy_gaussian': self.e_greedy_gaussian_policy,
            'gaussian': self.gaussian_policy,
            # 'softmax': self.softmax_policy
        }

        return polciy_types[policy_type](s, episode, train_or_test)

    def update(self, target_Q, sync_networks=True):
        if len(self.buffer_object) < self.params['batch_size']:
            return 0, {
            "average_Q":0,
            "average_Q_star": 0
        }
        s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix = self.buffer_object.sample(
            self.params['batch_size'])
        r_matrix = numpy.clip(r_matrix,
                              a_min=-self.params['reward_clip'],
                              a_max=self.params['reward_clip'])

        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)

        with torch.no_grad():
            Q_star, _ = target_Q.get_best_qvalue_and_action(sp_matrix)
            Q_star = Q_star.reshape((self.params['batch_size'], -1))
            y = r_matrix + (self.params['gamma'] * (1 - done_matrix) * Q_star)

        y_hat = self.forward(s_matrix, a_matrix)
        loss = self.criterion(y_hat, y)
        self.zero_grad()
        loss.backward()

        self.value_optimizer.step()
        self.zero_grad()

        _, best_actions = self.get_best_qvalue_and_action(s_matrix)
        neg_y_hat = -1 * self.forward(s_matrix, best_actions)
        # print('neg is not neg now')
        neg_y_hat_mean = neg_y_hat.mean()
        neg_y_hat_mean.backward()
        self.policy_optimizer.step()
        self.zero_grad()

        if sync_networks:
            utils_for_q_learning.sync_networks(
                target=target_Q,
                online=self,
                alpha=self.params['target_network_learning_rate'],
                copy=False)
        loss = loss.item()
        average_q = y_hat.mean().item()
        average_next_q_max = Q_star.mean().item()
        return loss, {
            "average_Q": average_q,
            "average_Q_star": average_next_q_max
        }