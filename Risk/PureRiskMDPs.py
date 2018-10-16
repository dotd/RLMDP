from gym import Env
import numpy as np


class PureRiskMDP(Env):
    def __init__(self,
                 random_generator,
                 shape,
                 noise_prob,
                 ):
        self.random_generator = random_generator
        self.shape = shape
        self.noise_prob = noise_prob
        self.location = [0, 0] # (x,y)


    # 0 up, 1 right, 2 down, 3 left
    # more negative reward
    # ^ up
    # |
    # |
    # +---------------> right - more positive reward
    def step(self, action):
        random_noise = self.random_generator.uniform(0,1) < self.noise_prob
        if random_noise:
            self.reset()

        if action == 0 and self.location[1]+1 < self.shape[1]:
            self.location[1] += 1
        if action == 1 and self.location[0]+1 < self.shape[0]:
            self.location[0] += 1
        if action == 2 and self.location[1] > 0:
            self.location[1] -= 1
        if action == 3 and self.location[0] > 0:
            self.location[0] -= 1

        reward = self.location[0] if self.random_generator.uniform(0, 1) < 0.5 else -self.location[1]
        return self.location, reward, random_noise, random_noise

    def reset(self):
        self.location = [self.random_generator.randint(0,self.shape[i]) for i in range(2)]
        return self.location

    def render(self, mode='human'):
        pass

    def seed(self, seed=42):
        return np.random.RandomState(seed)

    def close(self):
        pass

