from env import BlackjackEnv

import numpy as np
import random

class BlackjackAgent:
    """
    Blackjack agent that uses a simple policy to play the game.
    """
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((32, 11, 2, 2))  # player sum, dealer card, usable ace, action
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # exploration rate

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # explore
        else:
            return np.argmax(self.q_table[state])  # exploit