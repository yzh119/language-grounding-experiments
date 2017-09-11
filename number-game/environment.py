import numpy as np

def two_hot(n, a, b):
    ret = np.zeros(2 * n)
    ret[[a, n + b]] = 1
    return ret

class Game(object):
    def __init__(self, n):
        self.n = n
        self.target = None
        self.flip = None
        self.x = None
        self.y = None
        self.reset()
    
    def reset(self):
        self.flip = np.random.randint(2)
    
    def sender_input(self):
        x, y = np.random.choice(range(self.n), 2, replace=False)
        self.x, self.y = x, y
        self.target = 0
        return two_hot(self.n, x, y)
        
    def receiver_input(self, val):
        if self.flip == 1: 
            self.target = 1 - self.target
            self.x, self.y = self.y, self.x
        return np.concatenate((two_hot(self.n, self.x, self.y), [val]))
    
    def reward(self, out):
        assert self.target is not None
        if out == self.target:
            return 1
        else:
            return 0    

if __name__ == '__main__':
    g = Game(10)
    print(g.sender_input())
    print(g.receiver_input(0.5))
    g.reset()
    print(g.sender_input())
    print(g.receiver_input(0.5))