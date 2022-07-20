"""
Notes:
----------
- It is not necessary to register a gym environment to use it
    in an external training loop.
- I just want to play around with it and test it
"""


from gym.envs.registration import register
import gym
from rlagent.rlagent_v2 import RLAgent

# register custom gym environment
register(
    id='trading-v0',
    # correct version...
    entry_point='tradingenv_v7:TradingEnvironment',
)

agent=RLAgent('test_agent', 100)

# gym make:
trading_environment = gym.make('trading-v0',
                               agent=agent)

