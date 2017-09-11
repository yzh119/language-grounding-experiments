
import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
Format:
input vector: (2n + 1)
output vector(prob): (2)
"""
class Receiver(nn.Module):
    def __init__(self, n, hid):
        super(Receiver, self).__init__()
        self.n = n
        self.hid = hid
        self.net = nn.Sequential(
            nn.Linear(2 * n + 1, hid),
            nn.SELU(),
            nn.Linear(hid, hid),
            nn.SELU(),
            nn.Linear(hid, 2),
            nn.LogSoftmax()
        )
    
    def forward(self, input):
        input_size = input.size(1)
        assert input_size == self.n * 2 + 1
        return self.net(input)

"""
Format:
input vector: (2n)
output vector(real number range from 0 to 1): (1)
"""
class SenderActor(nn.Module):
    def __init__(self, n, hid):
        super(SenderActor, self).__init__()
        self.n = n
        self.hid = hid
        self.net = nn.Sequential(
            nn.Linear(2 * n, hid),
            nn.SELU(),
            nn.Linear(hid, hid),
            nn.SELU(),
            nn.Linear(hid, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        input_size = input.size(1)
        assert input_size == self.n * 2
        return self.net(input)

"""
Format:
input vector: (2n + 1)
output vector(Q value): (1)
"""
class SenderCritic(nn.Module):
    def __init__(self, n, hid):
        super(SenderCritic, self).__init__()
        self.n = n
        self.hid = hid
        self.net = nn.Sequential(
            nn.Linear(2 * n + 1, hid),
            nn.SELU(),
            nn.Linear(hid, hid),
            nn.SELU(),
            nn.Linear(hid, 1),
        )
    
    def forward(self, input):
        input_size = input.size(1)
        assert input_size == self.n * 2 + 1
        return self.net(input)