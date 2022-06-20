# TODO: Create trading env as entry point for gym/Rllib

import tempfile
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import scale
# TODO: How can TradingEnvironment access market_interface?
#from agent.agent import BaseAgent
#from rlagent.rlagent import RLAgent # Problem: circular import
#from env.rlreplay import Backtest

class TradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # I could include a config file to define env parameters

        # good practive to define spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.zeros(40), high=np.array([1_000_000_000] * 40))
        # call reset, get first observation (only mandatory thing in __init__)

        self.reset()

    def step(self, action, agent_market_interface, market_id, quantity):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.
        """
        # 1) Take Action
        # TODO: Das funktioniert zwar so, aber am Ende darf step() nur action als einzigen input nehmen...
        self.take_action(action, agent_market_interface, market_id, quantity)

        # 2) Get New Observation

        # 3) Determine Rewards

        # 4) Done


        #return obs, reward, done, info

    def reset(self):
        # reset all environment variables
        # usually returns the first observation for the first iteration step of an episode
        pass

    def _get_obs(self):
        "Generate New observation (should be based on run)"
        pass


    def take_action(self, action, agent_market_interface, market_id, quantity): # later: _take_action when its only called inside class
        """
        Takes action as input and commits the corresponding trading actions
        via the market_interface, e.g. submit_order. Will be called by the
        step() method.
        """
        #TODO: Dirty Solution, wie würde man es professioneller machen?
        if not agent_market_interface.get_filtered_orders(
                market_id, status="ACTIVE"):

            # action == 0: submit market sell order
            if action == 0:
                agent_market_interface.submit_order(market_id, "sell", quantity)

            # action == 2: submit market buy order
            elif action == 2:
                agent_market_interface.submit_order(market_id, "buy", quantity)

            # action == 1: wait
            else:
                pass

    def render(self, mode='human'):
        pass

    # optional?
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]