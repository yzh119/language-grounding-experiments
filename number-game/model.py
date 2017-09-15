
import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
Format:
input vector: (kn + 1)
output vector(prob): (k)
"""
class Receiver(nn.Module):
    def __init__(self, n, k, hid):
        super(Receiver, self).__init__()
        self.n = n
        self.k = k
        self.hid = hid
        self.net = nn.Sequential(
            nn.Linear(k * n + 1, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, k),
            nn.LogSoftmax()
        )
    
    def forward(self, input):
        input_size = input.size(1)
        assert input_size == self.n * self.k + 1
        return self.net(input)

"""
Format:
input vector: (kn)
output vector(real number range from 0 to 1): (1)
"""
class SenderActor(nn.Module):
    def __init__(self, n, k, hid):
        super(SenderActor, self).__init__()
        self.n = n
        self.k = k
        self.hid = hid
        self.net = nn.Sequential(
            nn.Linear(k * n, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        input_size = input.size(1)
        assert input_size == self.n * self.k
        return self.net(input)

"""
Format:
input vector: (kn + 1)
output vector(Q value): (1)
"""
class SenderCritic(nn.Module):
    def __init__(self, n, k, hid):
        super(SenderCritic, self).__init__()
        self.n = n
        self.k = k
        self.hid = hid
        self.net = nn.Sequential(
            nn.Linear(k * n + 1, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )
    
    def forward(self, input):
        input_size = input.size(1)
        assert input_size == self.n * self.k + 1
        return self.net(input)