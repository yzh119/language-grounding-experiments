
from environment import Game
from model import *
from itertools import count
import torch.optim as optim
import logging

bound = [0., 1.]

# Hyper parameters
explore_sigma = 0.05
n_r = 10 

n_numbers = 10
n_hidden = 50
n_games = 32
game_pool = []
for i in range(n_games):
    game_pool.append(Game(n_numbers))

R = Receiver(n_numbers, n_hidden)
SA = SenderActor(n_numbers, n_hidden)
SC = SenderCritic(n_numbers, n_hidden)

optim_r = optim.SGD(R.parameters(), lr=1e-4)
optim_sa = optim.SGD(SA.parameters(), lr=1e-4)
optim_sc = optim.SGD(SC.parameters(), lr=1e-4)

running_succ_rate = 0
for epoch in count():
    succ_rate = 0
    for i in range(n_games):
        game_pool[i].reset()
    
    
    
    
    running_succ_rate = running_succ_rate * 0.95 + succ_rate * 0.05
    print('successful_rate = {}'.format(running_succ_rate))