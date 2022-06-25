import gym
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
import random
import pandas
from gym.envs.registration import register
from gym_linkage.tradingenv_v8 import TradingEnvironment
from rlagent.rlagent_v2 import RLAgent



import ray
# Start a new instance of Ray (when running this tutorial locally) or
# connect to an already running one (when running this tutorial through Anyscale).
ray.init()

from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
'''
agent = RLAgent(
    name="RLAgent",
    quantity=100)

print(agent)

env_config = {"agent":agent,
              "config_dict":None
              }

'''
config = {
    "env": TradingEnvironment,
    #"env_config": {
    #    "config": {
    #        "agent": agent,
    #        "config_dict": None,
    #    },
    #}
}



rllib_trainer = PPOTrainer(config=config)
rllib_trainer
