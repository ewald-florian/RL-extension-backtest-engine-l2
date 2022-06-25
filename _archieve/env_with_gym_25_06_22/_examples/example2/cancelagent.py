# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent.agent import BaseAgent
from env.replay import Backtest
import datetime
import numpy as np
import pandas as pd


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

        best_bid = book_state['L1-AskPrice']
        print(best_bid)
        limit = float(best_bid - 0.1)
        print(limit)
        self.market_interface.submit_order(
                market_id, "buy", self.quantity, limit=limit)
        #self.market_interface.submit_order(market_id, side="buy", quantity=self.quantity, limit=limit)

        order_list = self.market_interface.get_filtered_orders(market_id, side='buy', status='ACTIVE')
        print('order list:', order_list)

    def on_trade(self, market_id: str, trades_state: pd.Series):
        pass

    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):

        trading_time = timestamp.time() > self.start_time and \
                       timestamp.time() < self.end_time

        # Enter trading phase if
        # (1) current time in defined trading_time
        # (2) trading_phase is False up to now
        if trading_time and not self.trading_phase:
            print('Algo is now able to trade...')
            self.trading_phase = True

        # Close trading phase if
        # (1) current time not in defined trading_time
        # (2) trading_phase is True up to now
        elif not trading_time and self.trading_phase:

            for market_id in self.market_interface.market_state_list.keys():

                # cancel active orders for this market
                [self.market_interface.cancel_order(order) for order in
                 self.market_interface.get_filtered_orders(market_id,
                                                           status="ACTIVE")]

                # close positions for this market
                if self.market_interface.exposure[market_id] > 0:
                    self.market_interface.submit_order(
                        market_id, "sell", self.quantity)
                if self.market_interface.exposure[market_id] < 0:
                    self.market_interface.submit_order(
                        market_id, "buy", self.quantity)

            self.trading_phase = False


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
                                   num_episodes=2,
                                   )







