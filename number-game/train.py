# About discriministic policy gradient
# Please refer to the following papers: 
# 1. http://proceedings.mlr.press/v32/silver14.pdf
# 2. https://arxiv.org/pdf/1509.02971.pdf

from environment import Game
from model import *
from itertools import count
from torch.autograd import Variable
import torch.optim as optim
import training_monitor.logger as tmlog
import numpy as np
import argparse

logger = tmlog.Logger('log')

parser = argparse.ArgumentParser('Number Game')
parser.add_argument('--number', '-n', type=int, default=10)
parser.add_argument('--k', '-k', type=int, default=2)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--batch', type=int, default=32)
args = parser.parse_args()

bound = [0., 1.]

n_numbers = args.number
n_k = args.k
cuda = args.cuda
# Hyper parameters
explore_sigma = 0.4 / (n_numbers + 1)
n_r = 20

def tocuda(var, cuda=False):
    if cuda:
        return var.cuda()
    else:
        return var

def explore_noise():
    return np.random.normal(0, explore_sigma)

def toggle_module(mod, requires_grad=True):
    for p in mod.parameters():
        p.requires_grad = requires_grad

def update_module(mod_name):
    global optim_r, optim_sa, optim_sc, game_pool, R, SA, SC, cuda
    
    s_input = []
    for i in range(n_games):
        s_input.append(game_pool[i].sender_input())

    if mod_name == 'actor':
        s_input = tocuda(Variable(torch.Tensor(s_input)), cuda)
        action_s = SA(s_input)
        
        r_input = []
        for i in range(n_games):
            r_input.append(game_pool[i].receiver_input(action_s.data[i][0] + explore_noise()))

        r_input = tocuda(Variable(torch.Tensor(r_input), volatile=True), cuda)
        toggle_module(R, requires_grad=False)
        action_r = R(r_input)
        discrete_action_r = torch.max(action_r, 1)[1]
        reward = []
        for i in range(n_games):
            reward.append(game_pool[i].reward(discrete_action_r.data[i]))
        
        r_input.volatile=False
        r_input.requires_grad=True
        predict_reward = SC(r_input)
        target_reward = tocuda(torch.FloatTensor(reward).view(-1, 1))
        #optim_sc.zero_grad()
        loss = F.mse_loss(predict_reward, tocuda(Variable(target_reward), cuda))
        loss.backward()
        #optim_sc.step()
        
        optim_sa.zero_grad()
        action_s.backward(r_input.grad[:, -1].contiguous().view(-1, 1))
        optim_sa.step()
    elif mod_name == 'critic':
        s_input = tocuda(Variable(torch.Tensor(s_input), volatile=True), cuda)
        action_s = SA(s_input)

        r_input = []
        for i in range(n_games):
            r_input.append(game_pool[i].receiver_input(action_s.data[i][0] + explore_noise()))

        r_input = tocuda(Variable(torch.Tensor(r_input), requires_grad=False), cuda)
        
        toggle_module(R)
        action_r = R(r_input)
        discrete_action_r = torch.max(action_r, 1)[1]
        reward = []
        labels = []
        for i in range(n_games):
            reward.append(game_pool[i].reward(discrete_action_r.data[i]))
            labels.append(game_pool[i].target)

        predict_reward = SC(r_input)
        target_reward = tocuda(torch.FloatTensor(reward).view(-1, 1), cuda)
        optim_sc.zero_grad()
        loss = F.mse_loss(predict_reward, Variable(target_reward))
        loss.backward()
        optim_sc.step()
        
        optim_r.zero_grad()
        loss = F.nll_loss(action_r, tocuda(Variable(torch.LongTensor(labels)), cuda))
        loss.backward()
        optim_r.step()
    else:
        raise ValueError('Invalid parameter')
    
    for i in range(n_games):
        game_pool[i].reset()
    
    return sum(reward) / 32.

n_hidden = 200
n_games = args.batch
game_pool = []
for i in range(n_games):
    game_pool.append(Game(n_numbers, n_k))

R = Receiver(n_numbers, n_k, n_hidden)
SA = SenderActor(n_numbers, n_k, n_hidden)
SC = SenderCritic(n_numbers, n_k, n_hidden)
if cuda:
    R.cuda()
    SA.cuda()
    SC.cuda()

optim_r = optim.Adam(R.parameters(), lr=1e-3)
optim_sa = optim.Adam(SA.parameters(), lr=1e-3)
optim_sc = optim.Adam(SC.parameters(), lr=1e-3)

running_succ_rate = 1. / n_k
for epoch in count(1):
    if epoch % n_r == 0:
        succ_rate = update_module('actor')
    else:
        succ_rate = update_module('critic')
    
    running_succ_rate = running_succ_rate * 0.95 + succ_rate * 0.05
    print('Epoch {}: successful_rate = {}'.format(epoch, running_succ_rate))
    
    if epoch % args.interval == 0:
        logger.add_scalar('succ_rate', running_succ_rate, epoch)
    
    if running_succ_rate > 0.95:
        break

stat = [[] for _ in range(n_numbers)]
for _ in range(50):
    for i in range(n_games):
        game_pool[i].reset()
    
    s_input = []
    for i in range(n_games):
        s_input.append(game_pool[i].sender_input())
    
    s_input = Variable(torch.Tensor(s_input), volatile=True)
    if cuda:
        s_input = s_input.cuda()
    action_s = SA(s_input).view(-1).data.tolist()
    for i in range(n_games):
        select_val = game_pool[i].states[game_pool[i].target]
        stat[select_val].append(action_s[i])

for j in range(n_numbers):
    print np.mean(stat[j]), np.std(stat[j])

import matplotlib.pyplot as plt
import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0, 1, n_numbers))
for x, c in enumerate(colors):
    for y in stat[x]:
        plt.scatter(x, y, color=c, s=0.3)
plt.savefig('viz.png')