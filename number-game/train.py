# About discriministic policy gradient
# Please refer to the following papers: 
# 1. http://proceedings.mlr.press/v32/silver14.pdf
# 2. https://arxiv.org/pdf/1509.02971.pdf

from environment import Game
from model import *
from itertools import count
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import logging

bound = [0., 1.]

# Hyper parameters
explore_sigma = 0.005
n_r = 20 

def explore_noise():
    return np.random.normal(0, explore_sigma)

def toggle_module(mod, requires_grad=True):
    for p in mod.parameters():
        p.requires_grad = requires_grad

def update_module(mod_name):
    global optim_r
    global optim_sa
    global optim_sc
    global game_pool
    global R
    global SA
    global SC
    
    for i in range(n_games):
        game_pool[i].reset()
    
    s_input = []
    for i in range(n_games):
        s_input.append(game_pool[i].sender_input())

    if mod_name == 'actor':
        s_input = Variable(torch.Tensor(s_input))
        action_s = SA(s_input)
        
        r_input = []
        for i in range(n_games):
            r_input.append(game_pool[i].receiver_input(action_s.data[i][0] + explore_noise()))

        r_input = Variable(torch.Tensor(r_input), volatile=True)
        toggle_module(R, requires_grad=False)
        action_r = R(r_input)
        discrete_action_r = torch.max(action_r, 1)[1]
        reward = []
        for i in range(n_games):
            reward.append(game_pool[i].reward(discrete_action_r.data[i]))
        
        r_input.volatile=False
        r_input.requires_grad=True
        predict_reward = SC(r_input)
        target_reward = torch.FloatTensor(reward).view(-1, 1)
        optim_sc.zero_grad()
        loss = F.mse_loss(predict_reward, Variable(target_reward))
        print loss.data[0]
        loss.backward()
        optim_sc.step()
        
        optim_sa.zero_grad()
        action_s.backward(r_input.grad[:, -1].contiguous().view(-1, 1))
        optim_sa.step()
    elif mod_name == 'critic':
        s_input = Variable(torch.Tensor(s_input), volatile=True)
        action_s = SA(s_input)

        r_input = []
        for i in range(n_games):
            r_input.append(game_pool[i].receiver_input(action_s.data[i][0] + explore_noise()))

        r_input = Variable(torch.Tensor(r_input))
        
        toggle_module(R)
        action_r = R(r_input)
        discrete_action_r = torch.max(action_r, 1)[1]
        reward = []
        for i in range(n_games):
            reward.append(game_pool[i].reward(discrete_action_r.data[i]))

        optim_r.zero_grad()
        loss = F.nll_loss(action_r, Variable(torch.LongTensor(reward)))
        loss.backward()
        optim_r.step()
    else:
        raise ValueError('Invalid parameter')
    
    return sum(reward) / 32.

n_numbers = 2
n_hidden = 50
n_games = 32
game_pool = []
for i in range(n_games):
    game_pool.append(Game(n_numbers))

R = Receiver(n_numbers, n_hidden)
SA = SenderActor(n_numbers, n_hidden)
SC = SenderCritic(n_numbers, n_hidden)

optim_r = optim.SGD(R.parameters(), lr=1e-5)
optim_sa = optim.SGD(SA.parameters(), lr=1e-4)
optim_sc = optim.SGD(SC.parameters(), lr=1e-4)

running_succ_rate = 0.5
for epoch in count(1):
    if epoch % n_r == 0:
        succ_rate = update_module('actor')
    else:
        succ_rate = update_module('critic')
    
    running_succ_rate = running_succ_rate * 0.95 + succ_rate * 0.05
    print('successful_rate = {}'.format(running_succ_rate))