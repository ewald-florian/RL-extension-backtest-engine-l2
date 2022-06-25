# New bottom-up approach to build the env class
from env.rlreplayTTT import Backtest
from env.rlreplayTTT import Episode
from rlagent.rlagent import RLAgent
from env.market import MarketState, Order, Trade
from context.context import MarketContext

import gym
from gym import spaces
import numpy as np
import logging
import copy
import datetime
import random
import pandas as pd

class TradingEnvironment(gym.Env):

    def __init__(self):

        # instantiate agent and backtest
        self.agent = RLAgent(name="RLAgent",quantity=100)
        # TODO: deepcopy? Sollte sich nicht verändern...
        self._agent = self.agent
        self.backtest = Backtest(agent=self.agent)
        self.context = MarketContext()

        # backtest parameter (remove hardcode later)
        self.identifier_list = ["Adidas.BOOK", "Adidas.TRADES"]
        self.date_start = "2021-01-04"
        self.date_end = "2021-01-08"
        self.episode_interval = 10
        self.episode_shuffle = True
        self.episode_buffer = 5
        self.episode_length = 10
        self.num_episodes = 5

        self.episode_start_list = []

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.zeros(40), high=np.array([1_000_000_000] * 40))

    def reset(self):
        pass

    def generate_episode_start_list(self):
        # generate episode_start_list which is needed for run.
        date_start = pd.Timestamp(self.date_start)
        date_end = pd.Timestamp(self.date_end)

        episode_buffer = pd.Timedelta(self.episode_buffer, "min")
        episode_length = pd.Timedelta(self.episode_length, "min")

        episode_start_list = pd.date_range(start=date_start, end=date_end + pd.Timedelta("1d"),
                                           freq=f"{self.episode_interval}min",
                                           normalize=True,  # start at 00:00:00.000
                                           )
        test_list = [
            lambda timestamp: timestamp.weekday() not in [5, 6],  # sat, sun
            lambda timestamp: datetime.time(8, 0, 0) <= timestamp.time(),  # valid start
            lambda timestamp: (timestamp + episode_length).time() <= datetime.time(16, 30, 0),  # valid end
            # ...
        ]

        self.episode_start_list = [start for start in episode_start_list
                              if all(test(start) for test in test_list)
                              ]

        if self.episode_shuffle:
            random.shuffle(self.episode_start_list)

    def initiate_episode(self):
        # 1) Reset all variables
        # 2) Return first observation (?)

        episode_index = 1 # hardcode for dev

        episode_start_buffer = self.episode_start_list[episode_index]
        episode_start = self.episode_start_list[episode_index] + pd.Timedelta(self.episode_buffer, "min")
        episode_end = self.episode_start_list[episode_index] + pd.Timedelta(self.episode_length, "min")

        # call Episode from replay
        self.episode = Episode(
                identifier_list=self.identifier_list,
                episode_start_buffer=episode_start_buffer,
                episode_start=episode_start,
                episode_end=episode_end,
            )

        # backtest needs an agent instance for on_quote etc.
        self.backtest.agent = copy.copy(self._agent)

        identifier_list = set(identifier.split(".")[0] for identifier
                              in self.identifier_list
                              )
        # create market_state instances
        for market_id in identifier_list:
            _ = MarketState(market_id)

        #print('market_id:', market_id)

    def run_step(self):
        display_interval = 10 # hardcode
        #self.__class__.timestamp_global = self.episode.timestamp
        #self.timestamp_global = self.episode.timestamp

        # TODO: Loop verstehen
        for update_store in self.episode:

            market_list = set(identifier.split(".")[0] for identifier in update_store)
            source_list = list(update_store)

            for market_id in market_list:
                self.backtest.market_step(market_id=market_id,
                                  book_update=update_store.get(f"{market_id}.BOOK"),
                                  trade_update=update_store.get(f"{market_id}.TRADES", pd.Series([None] * 3)),
                                  # optional, default to empty pd.Series
                                  )
             # during the buffer phase, do not inform agent about update
            #if episode.episode_buffering:
            #   continue

            # inform agent
            for source_id in source_list:
                self.backtest.agent_step(source_id=source_id,
                                 either_update=update_store.get(source_id),
                                 timestamp=self.episode.timestamp,
                                 timestamp_next=self.episode.timestamp_next,
                                 )
            # finally, report the current state of the agent
            #if not (step % display_interval):
            #print(self.backtest.agent)

    def step(self, action):

        self.take_action(action)
        self.run_step()
        # new obs
        self.context.store_market_context(self.backtest.book_state)
        obs = self.context.market_context
        print('NEW OBSERVATION')
        print(obs)

    def take_action(self, action): # later: _take_action when its only called inside class
        market_id = 'Adidas'
        quantity = 100
        # TODO: take_action an der richtigen Stelle callen
        if not self.agent.market_interface.get_filtered_orders(
                market_id, status="ACTIVE"):

            # action == 0: submit market sell order
            if action == 0:
                self.agent.market_interface.submit_order(market_id, "sell", quantity)

            # action == 2: submit market buy order
            elif action == 2:
                self.agent.market_interface.submit_order(market_id, "buy", quantity)

            # action == 1: wait
            else:
                pass

    def predict_random_action(self):
        action = np.random.randint(0,3)
        return action



    # for testing (not used)
    def try_run_backtest(self, identifier_list):
        self.backtest.run_episode_generator(self.identifier_list,
                                       self.date_start,
                                       self.date_end,
                                       self.episode_interval,
                                       self.episode_shuffle,
                                       self.episode_buffer,
                                       self.episode_length,
                                       self.num_episodes,
                                       )


env = TradingEnvironment()
env.generate_episode_start_list()
env.initiate_episode()
#env.run_step()
env.step(action=0)


#env.reset()
#env.run_step()
#env.try_run_backtest(identifier_list=identifier_list)