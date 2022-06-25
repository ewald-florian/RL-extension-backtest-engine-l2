
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
        self.start_time = datetime.time(8, 15)
        self.end_time = datetime.time(16, 15)
        self.market_interface.transaction_cost_factor = 0

        #self.context = MarketContext()
        # TODO: What is the best way to connect to tradingenv?
        # self.env = TradingEnvironment()

        # action
        self.pnl_old = 0
        self.last_obs = np.zeros(40, dtype='object')
        self.last_action = 1
        self.last_reward = 0

        # TODO: Solve Hardcode
        self.state_dim = 40
        self.num_actions = 3

        # dynamic attributes
        self.book_state_list = []
        self.obs = np.array([], dtype='float32')

        self.step = 0
        self.actions = []  # better array?
        self.pnls = []
        self.positions = []  # should I store long/short?
        self.trades = []

    def on_quote(self, market_id: str, book_state: pd.Series):

        # TODO: Call market_context
        #self.context.store_market_context(book_state=book_state)
        #self.obs = self.context.market_context

        # OBSERVATION SPACE
        self.obs = np.array(book_state[1:], dtype='float32')  # without timestamp
        # print(len(self.obs))
        # normalization mit tickprize? / min-max
        # oder zscore mit 5 tagen

        """
        # ACTION
        # include trading for testing
        action = np.random.randint(0, 2)
        if action == 0:
            self.market_interface.submit_order(market_id, "sell", self.quantity)

        # action == 2: submit market buy order
        elif action == 2:
            self.market_interface.submit_order(market_id, "buy", self.quantity)

        # action == 1: wait
        else:
            pass

        """
        # DONE (Problem: Done is not necessary in our env...)
        done = 0

        # REWARD
        pnl_new = (self.market_interface.pnl_unrealized_total + self.market_interface.pnl_realized_total)
        pnl_diff = pnl_new - self.pnl_old
        reward = pnl_diff
        self.pnl_old = pnl_new

        # INFO
        info = {}

        # save observation for next iteration
        self.last_obs = self.obs
        self.last_action = action
        self.last_reward = reward

        # STEP
        """

        self.env.step(action=action,
                        agent_market_interface=self.market_interface,
                        market_id=market_id,
                        quantity=self.quantity)
        """
        """

        # Call the env.take_action() method directly
        self.env.take_action(agent_market_interface=self.market_interface,
                             action=action,
                             market_id=market_id,
                             quantity=self.quantity)
        """


    def on_trade(self, market_id: str, trades_state: pd.Series):
        pass
    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):
        pass








