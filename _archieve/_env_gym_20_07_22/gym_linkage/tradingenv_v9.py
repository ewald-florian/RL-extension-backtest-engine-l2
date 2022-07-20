"""
Notes:
______
- Env for Rllib
- without importing the BaseAgent etc.

"""
"""
NOTE
____
IN ORDER TO USE:
-> Import correct version into base agent (e.g. rl_agent_env)
-> Base Agent / Market Interface has to use the correct version of TradingEnvironment etc.
"""
# TODO: Backtest input variablen logik
# TODO: Mit SimpleTradingEnvironment Logik testen

from env.market import MarketState, Order, Trade
from env.replay import Episode
#from rlagent.rlagent_v2 import RLAgent


# general imports
import copy
import datetime
logging.basicConfig(level=logging.CRITICAL)
import pandas as pd
import random
random.seed(42)

import gym
from gym import spaces
import numpy as np


class TradingEnvironment(gym.Env):

    metadata = {'render.modes': ['human']}


    def __init__(self, env_config:dict):
    #def __init__(self, config: dict):

        # ... directly from config
        #agent = config.get("agent")
        #config_dict = config.get("config_dict")

        agent = env_config.get("config").get("agent") # instance if agent class
        config_dict = env_config.get("config").get("config_dict") # config params for replay

        # instantiate TradingSimulator (the replay/simulation class)
        self.simulator = TradingSimulator(agent=agent, config_dict=config_dict)
        #self.replay = replay
        #self.agent = agent

        self.action_space = spaces.Discrete(3)
        # TODO: plausible min and max ranges
        self.observation_space = spaces.Box(np.zeros(40), np.array([10_000]*40))

    def step(self,action):
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        #TODO: which is the correct order???
        # I would rather say 1) action, 2) step...(the action has to affect
        # the environment before we want to observe the new environment?
        self._take_action(action)
        obs = self.simulator.take_step()
        done = self.simulator.replay_data.done
        # TODO: proper info dict (e.g. with infos from simulator)
        info = {}
        # TODO: proper reward function
        reward = self.simulator.agent.market_interface.pnl_realized_total

        return obs, reward, done, info

    def reset(self):
        #todo: zusammenlegen
        self.simulator.replay_data.reset_before_run()
        self.simulator.reset_simulation()
        # This is necessary to provide the first observation
        # which is the input argument for the first action!
        self.simulator.take_step()
        print("..reset for new episode")
        # return first observation (as last_obs)
        #TODO: later also return the agent state part of the observation
        return self.simulator.market_obs.copy()

    def render(self):
        pass

    def seed(self):
        pass


    # TODO: implement properly (where is the best place? in simulator)
    def _take_action(self,action):

        if action == 0:  # 0:sell
            self.simulator.agent.market_interface.submit_order('Adidas', "sell", 100)
        elif action == 2:  # 2:buy
            self.simulator.agent.market_interface.submit_order('Adidas', "buy", 100)
        else:  # 1:wait
            pass

class ReplayData:

    def __init__(self,config_dict=None):
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
        self.generate_episode_start_list()

    # todo: random market_id
    def generate_episode_start_list(self):
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

    def reset_before_run(self):
        """
        Resets agent, market and builds next Episode.
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
        # return if episode could not be generated
        except:
            logging.info("(ERROR) could not run episode with the specified parameters")
            return  # do nothing

        # Make the episode iterable (call it "self.iterable_episode")
        self.iterable_episode = iter(self.episode)
        # TODO: This could be a bottleneck?
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


# Note: Call TradingSimulator() instead of Backtest() in BaseAgent

# TODO: ReplayData und Simulator zusammenlegen zu "Replay"
class TradingSimulator():# replay
    """
    Gym Environment for Backtest Engine.
    """
    timestamp_global = None

    def __init__(self, agent=None, config_dict=None):

        # TODO: default_agent:

        # from arguments
        self._agent = agent
        self.result_list = []
        self.display_interval=10

        # TODO: Organisieren wo agent instanziiert und resetted wird (extern? Agent class?)
        # TODO: agent trennen

        # TODO: keine agent instanz notwendig, weglassen
        # TODO: agent und replay direkt über config übergeben
        self.agent = agent

        # instantiate replay data class with config_dict
        self.replay_data = ReplayData(config_dict=config_dict)


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

    def _agent_step(self, source_id, either_update, timestamp, timestamp_next):
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
    # TODO: __len__ für episode
    # TODO: episode_steps innerhalb von der episode counten
    # TODO: Episode.length, episode.step (done-flag einbauen)
    def take_step(self):

        # note: replay data is responsible for step counting
        self.replay_data.step_counter += 1
        # note: replay is responsible for done flag
        if self.replay_data.step_counter >= self.replay_data.current_episode_length:
            self.replay_data.done = True
            print('(ENV) DONE')

        # self.iterable_episode is the iter object of self.episode
        # self.iterable_episode is assigned in reset_before_run()
        # -> self.iterable_episode = iter(self.episode)
        update_store = next(self.replay_data.iterable_episode)
        # update global timestamp
        self.__class__.timestamp_global = self.replay_data.episode.timestamp

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
        # TODO: buffer phase
        # during the buffer phase, do not inform agent about update
        #if episode.episode_buffering:
        #   continue

        # inform agent
        for source_id in source_list:

            self._agent_step(source_id=source_id,
                             either_update=update_store.get(source_id),
                             timestamp=self.replay_data.episode.timestamp,
                             timestamp_next=self.replay_data.episode.timestamp_next,
                             )


        # report the current state of the agent
        if not (self.replay_data.step_counter % self.display_interval):
            #print("STEP NUMBER: ", self.replay_data.step_counter)
            print(self.agent)

        # NEW (MARKET) OBSERVATION
        obs = self.market_obs.copy()

        return obs

    def reset_simulation(self):
        # reset the agent
        # TODO: agent.reset() method, e.g. in base_agent
        self.agent = copy.copy(self._agent)
        # reset the environment instances
        # reset market instances
        MarketState.reset_instances()
        Order.reset_history()
        Trade.reset_history()

        # take identifier list from ReplayData
        identifier_list = self.replay_data.identifier_list
        # str split
        identifier_list = set(identifier.split(".")[0] for identifier
            in identifier_list
        )
        # create new MarketState instances
        for market_id in identifier_list:
            _ = MarketState(market_id)
