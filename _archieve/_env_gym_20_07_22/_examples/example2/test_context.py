# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent.agent import BaseAgent
from env.replay import Backtest
from context.context import MarketContext, ObservationFilter
import datetime
import numpy as np
import pandas as pd
from env.market import MarketState, Order, Trade


class CancelAgent(BaseAgent):

    def __init__(self, name: str, quantity: int,
                 ):
        """
        Trading agent implementation example. Improved version.

        :param name:
            str, agent name
        :param barrier_open:
            float, if barrier is hit, agent opens a position
        :param barrier_close:
            float, if barrier is hit, agent closes a position with profit
        :param stop_loss:
            float, if stop loss is hit, agent closes a position with loss
        :param quantity:
            int, defines the amount of shares that are traded
        """
        super(CancelAgent, self).__init__(name)

        # static attributes from arguments
        self.quantity = quantity


        # further static attributes
        self.start_time = datetime.time(8, 15)
        self.end_time = datetime.time(16, 15)
        self.market_interface.transaction_cost_factor = 0

        self.context = MarketContext()
        self.observationfilter = ObservationFilter()
        self.X = []


    def on_quote(self, market_id: str, book_state: pd.Series):
        # test market context
        self.context.store_market_context(book_state=book_state)
        market_context = self.context.market_context
        #print(type(market_context))
        #print(market_context.shape)
        print(book_state)








    def on_trade(self, market_id: str, trades_state: pd.Series):
        pass

    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):
        pass

if __name__ == "__main__":
    identifier_list = ["BMW.BOOK", "BMW.TRADES"]

    agent = CancelAgent(name="test_context",quantity=100)

    backtest = Backtest(agent=agent)

    backtest.run_episode_generator(identifier_list=identifier_list,
                                   date_start="2021-02-24",
                                   date_end="2021-02-26",
                                   episode_interval=30,
                                   episode_shuffle=True,
                                   episode_buffer=5,
                                   episode_length=10,
                                   num_episodes=2,
                                   )







