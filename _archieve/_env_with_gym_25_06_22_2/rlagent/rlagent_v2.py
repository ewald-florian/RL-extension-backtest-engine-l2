from agent.agent_rl_env import BaseAgent

# TODO: import correct RL agent version into context
#from context.context import MarketContext, ObservationFilter
import numpy as np
import pandas as pd
import os


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
        pass

    def on_trade(self, market_id: str, trades_state: pd.Series):
        pass

    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):
        pass








