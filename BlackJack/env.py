# BlackJack/env.py

import gym
from gym import spaces
import numpy as np

class BlackjackEnv(gym.Env):
    """
    Blackjack environment for OpenAI Gym.
    """
    def __init__(self):
        self.action_space = spaces.Discrete(2)  # 0=stand, 1=hit
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),   # player sum
            spaces.Discrete(11),   # dealer card (1-10)
            spaces.Discrete(2),    # usable ace
        ))
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def draw_card(self):
        return self.np_random.choice([1,2,3,4,5,6,7,8,9,10,10,10,10])

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def usable_ace(self, hand):
        return 1 in hand and sum(hand) + 10 <= 21

    def sum_hand(self, hand):
        if self.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    def is_bust(self, hand):
        return self.sum_hand(hand) > 21

    def score(self, hand):
        return 0 if self.is_bust(hand) else self.sum_hand(hand)

    def is_natural(self, hand):
        return sorted(hand) == [1, 10]

    def reset(self):
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        self.done = False
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)
        if self.done:
            raise ValueError("Call reset before step() again.")

        if action:  # hit
            self.player.append(self.draw_card())
            if self.is_bust(self.player):
                self.done = True
                reward = -1
            else:
                reward = 0
        else:  # stand
            self.done = True
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = np.sign(self.score(self.player) - self.score(self.dealer))
        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        return (self.sum_hand(self.player),
                self.dealer[0],
                self.usable_ace(self.player))

    def render(self, mode='human'):
        print(f"Player: {self.player}, total: {self.sum_hand(self.player)}")
        print(f"Dealer: {self.dealer}, total: {self.sum_hand(self.dealer)}" if self.done else f"Dealer: [{self.dealer[0]}, ?]")