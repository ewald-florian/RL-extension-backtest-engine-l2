# use relative imports for other modules
from env.market import MarketState, Order, Trade
# import Episode class
from env.replay import Episode
# import RLAgent to instantiate backtest
from rlagent.rlagent import RLAgent

# general imports
import copy
import datetime
import logging
logging.basicConfig(level=logging.CRITICAL) # logging.basicConfig(level=logging.NOTSET)
import os
import pandas as pd
import random
random.seed(42)
import time

SOURCE_DIRECTORY = "/Users/florianewald/PycharmProjects/l2-backtest-engine-v2X/efn2_backtesting"
DATETIME = "TIMESTAMP_UTC"


# copy backtest class from replay (development of rl env)
class Backtest:
    timestamp_global = None

    def __init__(self,
                 agent,  # backtest is wrapper for trading agent
                 ):
        """
        Backtest wrapper that is used to evaluate a trading agent on one or
        multiple episodes of historical market data.

        Note that the original _agent is used only to derive a fresh copy with
        each additional episode.

        :param agent:
            Agent, trading agent instance that is to be evaluated
        """

        # from arguments
        self._agent = agent

        # TODO: ...
        self.result_list = []

        # traingenv development:
        self.episode_start_list = []
        self.episode_counter = 0
        self.episode_index = 0
        self.display_interval=10

        # market/agent step ---

    def _market_step(self, market_id, book_update, trade_update):
        """
        Update post-trade market state and match standing orders against
        pre-trade market state.

        :param market_id:
            str, market identifier
        :param book_update:
            pd.Series, ...
        :param trade_update:
            pd.Series, ...
        """

        # update market state
        MarketState.instances[market_id].update(
            book_update=book_update,
            trade_update=trade_update,
        )

        # match standing agent orders against pre-trade state
        MarketState.instances[market_id].match()

    def _agent_step(self, source_id, either_update, timestamp, timestamp_next):
        """
        Inform trading agent about either book or trades state through the
        corresponding method. Also, inform trading agent about this and next
        timestamp.

        :param source_id:
            str, source identifier
        :param either_update:
            pd.Series, ...
        :param timestamp:
            pd.Timestamp, ...
        :param timestamp_next:
            pd.Timestamp, ...
        """

        # case 1: alert agent every time that book is updated
        if source_id.endswith("BOOK"):
            self.agent.on_quote(market_id=source_id.split(".")[0],
                                book_state=either_update,
                                )
        # case 2: alert agent every time that trade happens
        elif source_id.endswith("TRADES"):
            self.agent.on_trade(market_id=source_id.split(".")[0],
                                trades_state=either_update,
                                )
        # unknown source_id
        else:
            raise Exception("(ERROR) unable to parse source_id '{source_id}'".format(
                source_id=source_id,
            ))

        # _always_ alert agent with time interval between this and next timestamp
        self.agent.on_time(
            timestamp=timestamp,
            timestamp_next=timestamp_next,
        )

    # TODO: Wo genau sollte sie gecallt werden, in der run_episodes?
    #def run(self,
    def reset_before_run(self,
            identifier_list: list,
            episode_start_buffer: str,
            episode_start: str,
            episode_end: str,
            display_interval: int = 100,
            ):
        """
        Resets agent, market and builds next Episode.
        """
        # try to build episode based on the specified parameters
        try:
            self.episode = Episode(
                identifier_list=identifier_list,
                episode_start_buffer=episode_start_buffer,
                episode_start=episode_start,
                episode_end=episode_end,
            )
        # return if episode could not be generated
        except:
            logging.info("(ERROR) could not run episode with the specified parameters")
            return  # do nothing

        # reset market instances
        MarketState.reset_instances()
        Order.reset_history()
        Trade.reset_history()

        # create fresh copy of the original agent instance (del self.agent reduntant?)
        self.agent = copy.copy(self._agent)

        # setup market environment (Here, we have only a single market_id)
        MarketState(self.market_id)

    # perspektivisch die step funktion (ohne loop)
    def run_episode_steps(self): # perspektivisch die step funktion (ohne loop)
        # TODO: Was genau ist update_store?
        for step, update_store in enumerate(self.episode, start=1):

            # update global timestamp
            self.__class__.timestamp_global = self.episode.timestamp

            # update book_state, match standing orders
            self._market_step(market_id=self.market_id,
                                book_update=update_store.get(f"{self.market_id}.BOOK"),
                                trade_update=update_store.get(f"{self.market_id}.TRADES", pd.Series([None] * 3)),
                                # optional, default to empty pd.Series
                                )

            # TODO: How to handle the buffer Phase in the RL env?
            # during the buffer phase, do not inform agent about update
            if self.episode.episode_buffering:
                continue

            # step 3: inform agent -> based on original data
            source_list = list(update_store)
            # there is only one source ID
            source_id = source_list[0]
            self._agent_step(source_id=source_id,
                                 either_update=update_store.get(source_id),
                                 timestamp=self.episode.timestamp,
                                 timestamp_next=self.episode.timestamp_next,
                                 )

            # report the current state of the agent
            if not (step % self.display_interval):
                print(self.agent)

        # TODO: Report and store Results (e.g. PnLs)
        result = None
        self.result_list.append(result)

    #def run_episode_generator(self,
    def generate_episode_start_list(self,
                              identifier_list: list = ["Adidas.BOOK", "Adidas.TRADES"],
                              date_start: str = "2021-01-04",
                              date_end: str = "2021-01-04",
                              episode_interval: int = 30,  # timestamp quantization
                              episode_shuffle: bool = True,
                              episode_buffer: int = 5,
                              episode_length: int = 30,
                              num_episodes: int = 10,
                              ):
        """
        Generate start dates for Episodes.
        """
        # TODO: Move to Init(?)
        self.identifier_list = identifier_list
        self.date_start = date_start
        self.date_end = date_end
        self.episode_interval = episode_interval
        self.episode_shuffle = episode_shuffle
        self.episode_buffer = episode_buffer
        self.episode_length = episode_length
        self.num_episodes = num_episodes

        #TODO: We hacve only a single market ID, adjust all methods
        self.market_id = self.identifier_list[0].split('.')[0]

        # pd.Timestamp
        self.date_start = pd.Timestamp(self.date_start)
        self.date_end = pd.Timestamp(self.date_end)

        # pd.Timedelta
        self.episode_buffer = pd.Timedelta(self.episode_buffer, "min")
        self.episode_length = pd.Timedelta(self.episode_length, "min")

        # build episode_start_list
        episode_start_list = pd.date_range(start=self.date_start, end=self.date_end + pd.Timedelta("1d"),
                                           freq=f"{self.episode_interval}min",
                                           normalize=True,  # start at 00:00:00.000
                                           )
        # boundaries
        test_list = [
            lambda timestamp: timestamp.weekday() not in [5, 6],  # sat, sun
            lambda timestamp: datetime.time(8, 0, 0) <= timestamp.time(),  # valid start
            lambda timestamp: (timestamp + self.episode_length).time() <= datetime.time(16, 30, 0),  # valid end
            # ...
        ]
        episode_start_list = [start for start in episode_start_list
                              if all(test(start) for test in test_list)
                              ]
        # shuffle
        if episode_shuffle:
            random.shuffle(episode_start_list)

        # store episode start list as class instance for other methods (run_episodes)
        self.episode_start_list = episode_start_list


    def run_episodes(self):
        """
        Iterates over all Episodes in episode_start_list and calls the run method.
        """
        # set episode_counter and index to 0
        episode_counter = 0
        episode_index = 0
        # take next episode until ...
        while episode_counter < min(len(self.episode_start_list), self.num_episodes):
            # call run method to iterate over steps in Episode

            # reset_before_run (the reset part of the run method)
            self.reset_before_run(identifier_list=self.identifier_list,
                     episode_start_buffer=self.episode_start_list[episode_index],
                     episode_start=self.episode_start_list[episode_index] + pd.Timedelta(self.episode_buffer, "min"),
                     episode_end=self.episode_start_list[episode_index] + pd.Timedelta(self.episode_length, "min"),
                     )
            # the run part of the run method
            self.run_episode_steps()

            # update
            episode_index = episode_index + 1
            # TODO: Fix status flag (if episdode coulod not be build)
            episode_counter = episode_counter + 1


# Testing
identifier_list = ["Adidas.BOOK", "Adidas.TRADES"]

agent = RLAgent(
    name="RLAgent",
    quantity=100
    )

# Instantiate RLBacktest here
backtest = Backtest(agent=agent)
backtest.generate_episode_start_list() # with default config
backtest.run_episodes()





