# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym

# top-level ---

class Environment(gym.Env):

    def __init__(self, replay, agent, *args, **kwargs):

        #TODO: Question: but agent needs to be instantiate inside replay so that replay can use _agent_step???
        # So my approach would be to instantiate agent in replay and use replay as entry for TradingEnv.
        # Could I also instantiate agent in TradingEnv and pass it to replay (I guess so..)?

        self.replay = replay # TODO: replay should be instantiated with mode episode_generator, episode_broadcast, or episode_list
        self.agent = agent # TODO: 

    def step(self, action, *args, **kwargs):

        # get observation(t)
        # -> Replay.step() calls MarketState.update() and MarketState.match() 
        observation = self.replay.step(...)
        
        # get action(t)
        # -> Agent.step() calls MarketInterface.submit() or MarketInterface.cancel() to access order pool
        action = self.agent.step(observation, ...) 

        # -> note that observation(t) is updated in the background based on action(t-1)

    def reset(self, *args, **kwargs):
        
        # get next episode (if replay is not yet exhausted)
        # -> set next Episode, reset MarketState.instances, delete Order.history and Trade.history
        self.replay.reset() # update episode

        # get fresh agent 
        # -> reset agent attributes, the underlying models remain unchanged#
        # TODO:
        self.agent.reset() 

    def render(self, *args, **kwargs): 

        # use different LOB representations that can be rendered with each step
        # -> extending-window diagram with x-axis denoting time, and with y-axis denoting top-n prices per side + mid-point
        # -> snapshot diagram with x-axis price, and with y-axis volume
        # -> snapshot diagram in the form of a heat map
        # -> ...

        pass

# mid-level ---

# TODO: Q: What is the benefit when seperating agent step and market step?
class Replay: # used to go by 'Backtest'

    def __init__(self, *args, **kwargs):

        # prepare market environment
        self.market_list = ...
        self.order_list = ...
        self.trade_list = ...
        
        # set during instantiation
        self._episode_list = self._episode_generator(*args, **kwargs) # could be either of the below methods
        # set dynamically
        self._episode = None

    def step(self, *args, **kwargs):

        # update market environment (MarketState, Order, Trade) based on next update from episode
        try: 

            # retrieve updates
            update_store = self._episode.next()
            # update market_state instances
            for market_id, update in update_store:
                self.market_list[market_id].update(update)
                self.market_list[market_id].match()

            # retrieve observations
            observation_store = {}
            # ...
            for market_id, market in self.market_list:
                observation_store[market_id] = market.state
            # ...

        # stop if episode is exhausted
        except StopIteration: 
            pass

        return observation_store

    def reset(self, *args, **kwargs):
        
        # reset previously set market environment
        # ...

    # episode-related helper methods ---

    # - on Replay.__init__(), one out of the below methods determines the underlying _episode_list
    # - the _episode_list comprises a list of tuples [(start_buffer, start, end), *] 
    # - each tuple provides the parameters for an episode
    # - on Replay.reset(), the next episode is loaded automatically based on the next tuple (until replay is exhausted)

    def _episode_generator(self, *args, **kwargs): # produce similar episodes

        # option 1: generate episode_list based on episode_generator
        self._episode_list = None

    def _episode_broadcast(self, *args, **kwargs): # produce same-time episodes

        # option 2: generate episode_list based on episode_broadcast
        self._episode_list = None

    def _episode_manual(self, *args, **kwargs): # produce manual episodes from provided list

        # option 3: generate episode_list based on episode_manual
        self._episode_list = None

class Agent:

    def __init__(self, model, *args, **kwargs):
        
        # ...
        self.model = model

    def step(self, observation_store, *args, **kwargs):
        
        # ...
        action_store = {}
        # get action per market_id
        for market_id, observation in observation_store:
            action_store[market_id] = self.model.predict(observation)

        # implement action per market_id via MarketInterface
        for market_id, action in action_store:
            self.market_interface(market_id, action)

    def reset(self, *args, **kwargs):

        # keep somewhere a copy of the initial agent
        initial_copy = ...
        # then do something like this 
        self = initial_copy 

# bottom-level: episode ---

# TODO: I would import the normal Episode class without any changes..?
class Episode:
    """
    Episode is a generator that ...

    - loads and processes data (for multiple market_ids) 
    - allows iteration via __iter__ and __next__ method
    - throws StopIteration error when it is exhausted
    """

    def __init__(self, *args, **kwargs):
        pass  

    def __next__(self, *args, **kwargs): 
        pass

    # attributes ---
    # see original file ...

    # helper methods ---
    # see original file ...

# bottom-level: market environment ---

# TODO: I use the normal MarketState, Order, Trade classes from market.py
class MarketState: 

    def __init__(self, *args, **kwargs):
        pass  

    def update(self, *args, **kwargs):
        pass

    def match(self, *args, **kwargs):
        pass


class Order:

    history = list()

    def __init__(self, *args, **kwargs):
        pass  

    def execute(self, *args, **kwargs):
        pass

class Trade:

    history = list()

    def __init__(self, *args, **kwargs):
        pass  