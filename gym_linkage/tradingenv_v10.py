import numpy as np
import pandas as pd
import datetime
import copy
import random
import logging
logging.basicConfig(level=logging.CRITICAL)

from env.rlreplay import Episode
from env.market import MarketState, Order, Trade
import gym
from gym import spaces


class TradingEnvironment(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, env_config:dict):

        self.replay = env_config.get("config").get("replay")
        self.agent = env_config.get("config").get("agent")

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.zeros(40), np.array([10_000]*40))

    def step(self,action):
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        # 1) take action
        self.agent.take_action(action=action)
        # 2) call replay.step()
        # return update_store, timestamp, timestamp_next for agent.step()
        update_store, timestamp, timestamp_next = self.replay.step()
        # 3) call agent.step()
        self.agent.step(update_store, timestamp, timestamp_next)
        # 4) returns:
        # obs
        obs = self.replay.market_obs.copy()
        # reward
        reward = self.agent.agent.market_interface.pnl_realized_total
        # done
        done = self.replay.done
        # info
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.replay.reset()
        self.agent.reset()
        self.replay.step()
        print("..reset for new episode")
        # return first observation
        first_obs = self.replay.market_obs.copy()
        return first_obs

    def render(self):
        pass

    def seed(self):
        pass

class Replay:

    def __init__(self, config_dict=None):

        self.result_list = []
        self.display_interval = 10

        # default configuration for simulation
        config = {"identifier_list":
                            ["Adidas.BOOK", "Adidas.TRADES"],
                               "date_start": "2021-01-04",
                               "date_end": "2021-01-08",
                               "episode_interval": 5,
                               "episode_shuffle": True,
                               "episode_buffer": 5,
                               "episode_length": 10,
                               "num_episodes": 10
                               }
        # update config whith custom configuration
        if config_dict:
            config.update(config_dict)

        #todo: .get()
        # get parameters form config dict
        self.identifier_list = config["identifier_list"]
        self.date_start = config["date_start"]
        self.date_end = config["date_end"]
        self.episode_interval = config["episode_interval"]
        self.episode_shuffle = config["episode_shuffle"]
        self.episode_buffer = config["episode_buffer"]
        self.episode_length = config["episode_length"]
        self.num_episodes = config["num_episodes"]

        # TODO: We have only a single market ID, adjust, all methods
        self.market_id = self.identifier_list[0].split('.')[0]
        self.episode_start_list = []
        self.episode_counter = 0
        self.episode_index = 0
        self.step_counter = 0
        self.done = False
        # generate episode_start_list (required to run the simulation)
        self._generate_episode_start_list()

    def _generate_episode_start_list(self):
        """
        Generate list of start timestamps for episodes.
        """
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

        if self.episode_shuffle:
            random.shuffle(episode_start_list)

        # store episode start list as class instance for other methods (run_all_episodes)
        self.episode_start_list = episode_start_list

        # set episode counter ane episode index to 0:
        # set episode_counter and index to 0
        # Note: this is a logical place to do this because generate_episode_start_list
        # will always be called before any type of iteration loop over episodes starts
        # no matter if it is internal or external (and it cannot be done in reset_before_run)
        self.episode_counter = 0
        #TODO: episode_index if Bedingung (nur wenn Episode erfolgreich gebaut werden konnte...)
        # in reset before run?
        self.episode_index = 0

    def reset(self):

        self._build_new_episode()
        self._reset_market()

    # former reset_before_run()
    def _build_new_episode(self):
        """
        Resets market and builds next Episode.
        """

        # Note: episode index is set to 0 in generate_episode_list and then 1 is added after every episode
        episode_start_buffer = self.episode_start_list[self.episode_counter]
        episode_start = self.episode_start_list[self.episode_counter] + pd.Timedelta(self.episode_buffer, "min")
        episode_end = self.episode_start_list[self.episode_counter] + pd.Timedelta(self.episode_length, "min")

        # try to build episode based on the specified parameters
        try:
            self.episode = Episode(
                identifier_list=self.identifier_list,
                episode_start_buffer=episode_start_buffer,
                episode_start=episode_start,
                episode_end=episode_end,
            )


        #####
        #self.episode.__iter__()
        #####
        # return if episode could not be generated
        except:
            logging.info("(ERROR) could not run episode with the specified parameters")
            print("No Episode was build")
            return  # do nothing

        """
        # Make the episode iterable (call it "self.iterable_episode")
        self.iterable_episode = iter(self.episode)
        # TODO: This could be a bottleneck? (Should be outsourced to episode...)
        self.current_episode_length = len(list(self.episode))
        print('(ENV) CURRENT EPISODE LENGTH:', self.current_episode_length)
        # set step_counter to 0 (for new episode)
        self.step_counter = 0
        # set done flag to false (will be set true in Simulator)
        # if step > max_steps
        self.done = False
        # Adds 1 to episode_counter after each Episode
        self.episode_counter += 1
        # TODO: episode_index if Bedingung (nur wenn Episode erfolgreich gebaut werden konnte...)
        # in reset before run?
        self.episode_index += 1
        # print (note that this actually prints the "next" episode...
        print('(ENV) EPISODE COUNTER:', self.episode_counter)
        #todo: return first observation to env
        """

    def _reset_market(self):
        # reset market instances
        MarketState.reset_instances()
        Order.reset_history()
        Trade.reset_history()

        # take identifier list from ReplayData
        identifier_list = self.identifier_list
        # str split
        identifier_list = set(identifier.split(".")[0] for identifier
            in identifier_list
        )
        # create new MarketState instances
        for market_id in identifier_list:
            _ = MarketState(market_id)

######### Formerly TradingSimulator class ########

    def _market_step(self, market_id, book_update, trade_update):
        """
        Update post-trade market state and match standing orders against
        pre-trade market state.
        """
        # update market state
        MarketState.instances[market_id].update(
            book_update=book_update,
            trade_update=trade_update,
        )
        # match standing agent orders against pre-trade state
        MarketState.instances[market_id].match()

    # TODO: __len__ für episode
    # TODO: episode_steps innerhalb von der episode counten
    # TODO: Episode.length, episode.step (done-flag einbauen)
    def step(self):

        # note: replay data is responsible for step counting
        self.step_counter += 1
        # note: replay is responsible for done flag
        if self.step_counter >= self.current_episode_length:
            self.done = True
            print('(ENV) DONE')

        # self.iterable_episode is the iter object of self.episode
        # self.iterable_episode is assigned in reset_before_run()
        # -> self.iterable_episode = iter(self.episode)
        update_store = next(self.iterable_episode)
        # update global timestamp
        self.__class__.timestamp_global = self.episode.timestamp

        # Update MarketState
        market_list = set(identifier.split(".")[0] for identifier in update_store)
        source_list = list(update_store)

        # OBSERVATION
        #TODO: Include MarketContext class
        #TODO: it would be more intuitively to update the market first and then get
        # the infos directly from MarketState instead of getting them from the update dict..?
        # MARKET OBSERVATION (needed for predict_action())

        # e.g. 'Adidas.BOOK', second would be sometimes 'Adidas.TRADES'
        source_id = source_list[0]
        # save book as array without timestamp and labels
        market_obs = update_store.get(source_id).array[1:]
        # convert to float (for model)
        self.market_obs = market_obs.astype('float32')

        # update market
        for market_id in market_list:

            self._market_step(market_id=market_id,
                              book_update=update_store.get(f"{market_id}.BOOK"),
                              trade_update=update_store.get(f"{market_id}.TRADES", pd.Series([None] * 3)),
                              # optional, default to empty pd.Series
                              )

        # NEW (MARKET) OBSERVATION
        obs = self.market_obs.copy()
        #return obs

        # update store is necessary input for agent.step()
        return update_store, self.episode.timestamp, self.episode.timestamp_next

# TODO: Agent komplett selbstständig machen...
class AgentHelper:

    def __init__(self, agent):

        self.agent = agent
        # store initial agent for reset method
        self._agent = copy.copy(self.agent)

        # for logging
        self.steps = 0
        self.display_interval = 100

    # formerly _agent_step()
    def _inform_agent(self, source_id, either_update, timestamp, timestamp_next):
        """
        Inform trading agent about either book or trades state through the
        corresponding method. Also, inform trading agent about this and next
        timestamp.
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

    def step(self, update_store, timestamp, timestamp_next):

        self.steps = self.steps + 1

        source_list = list(update_store)
        # inform agent
        for source_id in source_list:
            self._inform_agent(source_id=source_id,
                             either_update=update_store.get(source_id),
                             timestamp=timestamp,
                             timestamp_next=timestamp_next,
                             )
        # log agent
        if not (self.steps % self.display_interval):
            print(self.agent)

    def reset(self):

        self.agent = copy.copy(self._agent)
        self.steps = 0

    def take_action(self,action):
        # TODO: remove market_id hardcode...
        if action == 0:  # 0:sell
            self.agent.market_interface.submit_order('Adidas', "sell", 100)
        elif action == 2:  # 2:buy
            self.agent.market_interface.submit_order('Adidas', "buy", 100)
        else:  # 1:wait
            pass




