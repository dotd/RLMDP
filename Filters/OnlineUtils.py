from collections import deque
import numpy as np


def get_exponential_filter(discount_factor, length):
    filter0 = np.zeros(shape=(length,))
    filter0[0] = 1
    for i in range(1, length):
        filter0[i] = filter0[i-1] * discount_factor
    return filter0


class OnlineFilter:
    def __init__(self,
                 filter_shape,  # The filter shape to consider
                 sampler_len=None,  # The sampler len (if needed.) If not needed is should be 0 or None
                 ):
        self.filter = filter_shape
        self.sampler_len = sampler_len

        # Get filter len (for both numpy and list)
        self.filter_len = self.filter.shape[0] if type(self.filter) is np.ndarray else len(self.filter)

        # implementation as deque
        self.deque_time = deque(maxlen=self.filter_len)
        self.deque_reward = deque(maxlen=self.filter_len)
        self.deque_state = deque(maxlen=self.filter_len)
        self.deque_action = deque(maxlen=self.filter_len)

        self.deque_result_time = deque(maxlen=self.filter_len)
        self.deque_result = deque(maxlen=self.filter_len)
        self.deque_result_state = deque(maxlen=self.filter_len)
        self.deque_result_action = deque(maxlen=self.filter_len)

        # fill-up with zeros
        for i in range(self.filter_len):
            self.deque_time.append(None)
            self.deque_state.append(None)
            self.deque_action.append(None)
            self.deque_reward.append(None)

            self.deque_result.append(None)
            self.deque_result_time.append(None)
            self.deque_result_state.append(None)
            self.deque_result_action.append(None)
        self.samples_counter = 0

        # Sampler
        self.sampler = {} if isinstance(sampler_len, int) and sampler_len>0 else None

    def add(self, reward, state, action=None):
        """
        Add sample and update the different queues
        After compute the result.
        :param reward:
        :param state:
        :return: Triplet of (
        """
        self.deque_reward.append(reward)
        self.deque_state.append(state)
        self.deque_action.append(action)
        self.deque_time.append(self.samples_counter)
        self.samples_counter += 1

        res = np.sum([x * y for x, y in zip(self.deque_reward, self.filter)]) if self.is_valid() else None
        self.deque_result.append(res)
        self.deque_result_time.append(self.deque_time[0])
        current_result = self.get()
        self._update_sampler(current_result)

        # Current result: filtered result, time, state
        return current_result

    def _update_sampler(self, current_result):
        """
        Update the sampler.
        :param current_result:
        :return:
        """
        if self.sampler is not None and current_result is not None:  # As long as we have a valid result
            sample = current_result[0]  # The value of this trajectory
            state = current_result[2]  # The state if this trajectory

            if state not in self.sampler:  # If we do not have entry for this state, we add it.
                self.sampler[state] = deque(maxlen=self.sampler_len)
            self.sampler[state].append(sample)

    def get(self):
        if not self.is_valid():
            return None
        return self.deque_result[-1], self.deque_time[0], self.deque_state[0], self.deque_action[0]

    def is_valid(self):
        """
        Checks whether the result is valid, meaning, there are enought samples for a real result.
        :return:
        """
        return self.samples_counter >= self.filter_len


if __name__ == "__main__":
    pass

