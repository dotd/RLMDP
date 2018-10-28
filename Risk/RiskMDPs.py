from collections import namedtuple
import numpy as np

from gym import Env
from gym import error, spaces


class PureRiskMDP(Env):
    def __init__(self,
                 random_generator,
                 shape,
                 noise_prob,
                 ):
        self.action_space = spaces.Discrete(5)

        self.random_generator = random_generator
        self.shape = shape
        self.noise_prob = noise_prob
        self.reset()

    #  0 up, 1 right, 2 down, 3 left, 4 None
    #  more variance
    #           ^ up
    #           |
    #           |
    #  <--------+----------> right - more positive reward
    #  left - more negative reward
    def step(self, action):
        random_noise = self.random_generator.uniform(0,1) < self.noise_prob
        if random_noise:
            self.reset()

        reward = self.next_location(action)

        return self.location, reward, random_noise, random_noise

    def reset(self):
        self.location = [0, 0]
        self.location[0] = self.random_generator.randint(-self.shape[0]+1,self.shape[0])
        self.location[1] = self.random_generator.randint(0,self.shape[1])
        return self.location

    def next_location(self, action):
        #  up
        if action == 0 and self.location[1]+1 < self.shape[1]:
            self.location[1] += 1
        #  right
        if action == 1 and self.location[0]+1 < self.shape[0]:
            self.location[0] += 1
        #  down
        if action == 2 and self.location[1] > 0:
            self.location[1] -= 1
        #  left
        if action == 3 and self.location[0]-1 > -self.shape[0]:
            self.location[0] -= 1

        reward = self.location[0] + self.random_generator.choice([-1,1]) * self.location[1]
        return reward

    def render(self, mode='human'):
        pass

    def seed(self, seed=42):
        return np.random.RandomState(seed)

    def close(self):
        pass

    def action_index_to_string(self, action):
        if action == 0:
            return "up"
        if action == 1:
            return "right"
        if action == 2:
            return "down"
        if action == 3:
            return "left"
        if action == 4:
            return "None"


Tuple = namedtuple('Tuple', ('state', 'action', 'next_state', 'reward', 'cnt', 'done', 'random_noise'))


class NoisyStepsMDP(PureRiskMDP):

    def __init__(self,
                 random_generator,
                 shape,
                 noise_prob,
                 maximal_steps
                 ):
        PureRiskMDP.__init__(self, random_generator, shape, noise_prob)
        self.maximal_steps = maximal_steps
        self.render_information = None
        self.reset()
        self.cnt = 0

    def reset(self):
        self.location = [0, self.shape[1]-1]  # (x,y)
        return self.location

    def step(self, action):
        # rememebr last steps
        last_state = self.location[:]
        last_cnt = self.cnt

        random_noise = self.random_generator.uniform(0,1) < self.noise_prob
        if random_noise:
            noise_as_action = self.random_generator.choice([0,1,2,3])
            self.next_location(noise_as_action)
        reward = self.next_location(action)

        done = False
        if self.cnt >= self.maximal_steps-1:
            self.cnt = 0
            self.reset()
            done = True
        else:
            self.cnt += 1

        self.render_information = Tuple(last_state, action, self.location, reward, last_cnt, done, random_noise)
        return self.location, reward, done, random_noise

    def render(self, mode='human'):
        print("s={},\ta={}, {},\tns={},\tr={},\tcnt={},\tdone={},\trnd_noise={}".format(self.render_information.state,
                                                                              self.render_information.action,
                                                                              self.action_index_to_string(self.render_information.action),
                                                                              self.render_information.next_state,
                                                                              self.render_information.reward,
                                                                              self.render_information.cnt,
                                                                              self.render_information.done,
                                                                              self.render_information.random_noise))
