import numpy as np
import random
def k_hot(n, lst):
    k = len(lst)
    assert k > 0
    ret = np.zeros(k * n)
    ret[[i * n + x for i, x in enumerate(lst)]] = 1
    return ret

class Game(object):
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.target = None
        self.states = []
        self.reset()
    
    def reset(self):
        self.states = np.random.choice(range(self.n), self.k, replace=False).tolist()
        self.target = 0
    
    def sender_input(self):
        return k_hot(self.n, self.states)
        
    def receiver_input(self, val):
        shuffled_index, shuffled_states = zip(*sorted(zip(range(self.k), self.states), key=lambda _: random.random()))
        self.target = list(shuffled_index).index(0)
        return np.concatenate((k_hot(self.n, shuffled_states), [val]))
    
    def reward(self, out):
        assert self.target is not None
        if out == self.target:
            return 1
        else:
            return 0    

if __name__ == '__main__':
    g = Game(10, 3)
    print(g.sender_input())
    print(g.receiver_input(0.5))
    g.reset()
    print(g.sender_input())
    print(g.receiver_input(0.5))