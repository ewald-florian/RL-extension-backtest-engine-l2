
# from agent.agent import BaseAgent
from agent.agent_rl_env import BaseAgent
# Import backtest from rlreplay (rl-adapted version of replay)
#from env.rlreplay import Backtest # also needs to be imported in agent.py
#from env.replay import Backtest
import datetime

# RL imports
# TODO: import correct RL agent version into context
#from context.context import MarketContext, ObservationFilter
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import tensorflow as tf

# from gym_linkage.tradingenv_v2 import TradingEnvironment

class RLAgent(BaseAgent):

    def __init__(self, name: str,
                 quantity: int,
                 ):
        """
        Trading agent implementation.
        """
        super(RLAgent, self).__init__(name)

        # static attributes
        self.quantity = quantity
        self.market_interface.transaction_cost_factor = 0

    def on_quote(self, market_id: str, book_state: pd.Series):

        if book_state.shape != (41,):
            print(100*'#')
            print(100 * '#')
        # take book_state series, ignore timestamp, convert to array with only numbers
        #self.obs = book_state[1:].array
        #print(self.obs)
        pass

    def on_trade(self, market_id: str, trades_state: pd.Series):
        pass

    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):
        pass








