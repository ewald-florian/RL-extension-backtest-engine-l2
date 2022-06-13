# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent.agent import BaseAgent
from env.replay import Backtest
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
        self.num_orders = 10

        # further static attributes
        self.start_time = datetime.time(8, 15)
        self.end_time = datetime.time(16, 15)
        self.market_interface.transaction_cost_factor = 0

        self.trading_phase = False

    def on_quote(self, market_id: str, book_state: pd.Series):

        # GENERATE ORDER FLOW FOR TESTING
        best_bid = book_state['L1-AskPrice']
        print(best_bid)
        limit = float(best_bid - 0.1)
        print(limit)
        self.market_interface.submit_order(
                market_id, "buy", self.quantity, limit=limit)

        # ORDER LIST
        # Filter Order.history based on market_id, side and status.
        order_list = self.market_interface.get_filtered_orders(market_id, side='buy', status=None)
        #print('_'*75)
        #print(order_list[:3])
        #print('_' * 75)

        # TRADE LIST
        # Filter Trade.history based on market_id and side.
        # Wie genau sieht das aus?
        #trade_list = self.market_interface.get_filtered_trades(market_id, side='buy')
        #if len(trade_list) >= 1:
        #print('_' * 75)
        #print('TRADE LIST: ', len(trade_list))
        #print('_' * 75)

        # Get Entire History:
        order_history = Order.history
        trade_history = Trade.history

        #MarketState
        midpoint = MarketState[market_id].mid_point
        print(midpoint)


    def on_trade(self, market_id: str, trades_state: pd.Series):
        pass

    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):
        pass

if __name__ == "__main__":
    identifier_list = [
        "BMW.BOOK", "BMW.TRADES",
    ]

    agent = CancelAgent(
        name="cancelagent1",
        quantity=100,
    )

    backtest = Backtest(
        agent=agent,
    )

    backtest.run_episode_generator(identifier_list=identifier_list,
                                   date_start="2021-02-24",
                                   date_end="2021-02-26",
                                   episode_interval=30,
                                   episode_shuffle=True,
                                   episode_buffer=5,
                                   episode_length=30,
                                   num_episodes=1,
                                   )







