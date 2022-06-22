# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: RL-Gym:
# TODO 1: Deliver Observation state to RL agent (return obs, done)
# TODO 2: Receive Action from RL agent
# TODO 3: Receive reward from environment (e.g. PnL) (return reward, info)
# TODO 4: Normalize observations (how?)
# TODO 8: define max_episode_steps (has to be consistent with replay.episode class)
# TODO 12: Run Script from backtest.run() e.g. with episode lists for the test version...
# TODO 13: Increase Observation state (e.g. last 100 book_states)
# TODO: Observation list and so on has to be reset after each episode
# TODO: Direkt Episoden in der Urzeit begrenzen anstatt die trading_phase flag zu verwenden
# TODO: done in der on_quote einführen wenn bestimmtes ereignis erfüllt ist, zB bankrott oder ziel erreicht

# TODO: Q: Should I Implement the entire RL in this script or in another one? e.g. the DDQN agent?
# TODO: Q: How to scale without knowing min, max, std or mean?
# TODO: Q: Does it make sense to utilize experience replay here?
# TODO: Q: Sollte das RL Model (zB build_model etc.) innerhalb von RLAgent stattfinden oder eine eigene Klasse bilden?
# TODO: Q: Sollte der agent steps on time oder on quote machen?

# TODO: Die Aufsplittung mit agent2, rl_replay ist zu aufwändig, einfach eine neue
# rl run episode methode im normalen backtest wäre besser...

# standard imports
# Import BaseAgent from agent2 since agent2 uses RLBacktest
from agent.agent import BaseAgent
from env.replay import Backtest
import datetime
import numpy as np
import pandas as pd

# RL imports
from ddqnmodel import DDQNModel
import logging
import gym
from gym import spaces
import warnings

warnings.filterwarnings('ignore')
from pathlib import Path
from time import time
from collections import deque
from random import sample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
# using tensorflow
import tensorflow as tf

import gym
from gym.envs.registration import register


class RLAgent(BaseAgent):

    def __init__(self, name: str,
                 quantity: int,
                 gamma=.99,
                 tau=100,
                 architecture=(256, 256),
                 learning_rate=0.0001,
                 l2_reg=1e-6,
                 replay_capacity=int(1e6),
                 batch_size=4096,
                 epsilon_start=1.0,
                 epsilon_end=.01,
                 epsilon_decay_steps=250,
                 epsilon_exponential_decay=.99,
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
        self.trading_phase = False
        # static rl attributes
        self.gamma = gamma,  # discount factor
        self.tau = tau  # target network update frequency
        self.architecture = architecture  # units per layer
        self.learning_rate = learning_rate  # learning rate
        self.l2_reg = l2_reg  # L2 regularization
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay

        # action
        self.pnl_old = 0
        self.last_obs = np.zeros(40, dtype='object')
        self.last_action = 1
        self.last_reward = 0

        # TODO
        self.trading_period_length = 1000

        # dynamic attributes
        self.book_state_list = []
        self.obs = np.array([], dtype='float32')

        self.step = 0
        self.actions = []  # better array?
        self.pnls = []
        self.positions = []  # should I store long/short?
        self.trades = []

        # initialize RL model
        self.config_tensorflow()
        self.initialize_model()
        self.set_up_gym()

        # check:
        print('Architecture: ', self.ddqn.architecture)
        print('Action Space: ', self.trading_environment.action_space)
        print('Observation Space Shape: ', self.trading_environment.observation_space.shape)

    def config_tensorflow(self):
        """
        Settings for tensorflow.
        """
        np.random.seed(42)
        tf.random.set_seed(42)
        # Use GPU if available
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if gpu_devices:
            print('Using GPU')
            tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        else:
            print('Using CPU')

    def set_up_gym(self):
        """
        Entry point has to refer to a class which inherits
        gym env: e.g. TradingEnvironment(gym.Env)
        """
        # TODO: get trading_period_length form lib (run_episodes)
        self.trading_period_length = 1000
        register(
            id='l2_trading-v0',
            # TODO: What would be the best entry point? tradingenv_v0?
            entry_point='tradingenv_v0:TradingEnvironment',
            max_episode_steps=self.trading_period_length)

        self.trading_environment = gym.make('l2_trading-v0')
        self.trading_environment.seed(42)

        # TODO: Should I assign the values here or in __init__?
        self.state_dim = self.trading_environment.observation_space.shape[0]
        self.num_actions = self.trading_environment.action_space.n
        # self.max_episode_steps = trading_environment.spec.max_episode_steps
        print('STATE DIM ', self.state_dim)
        print('Num_Actions ', self.num_actions)

    def initialize_model(self):
        """
        Initialize DDQN Model
        """
        # needs observation space and action space from trading_environment:
        # state_dim = trading_environment.observation_space.shape[0]
        # num_actions = trading_environment.action_space.n
        # hardcode for testing:
        # TODO: deduct dims from trading env
        state_dim = 40  # 10 levels*2 sides* 2 variables --> (40,)
        num_actions = 3  # 0,1,2

        tf.keras.backend.clear_session()

        self.ddqn = DDQNModel(state_dim=state_dim,
                              num_actions=num_actions,
                              learning_rate=self.learning_rate,
                              gamma=self.gamma,
                              epsilon_start=self.epsilon_start,
                              epsilon_end=self.epsilon_end,
                              epsilon_decay_steps=self.epsilon_decay_steps,
                              epsilon_exponential_decay=self.epsilon_exponential_decay,
                              replay_capacity=self.replay_capacity,
                              architecture=self.architecture,
                              l2_reg=self.l2_reg,
                              tau=self.tau,
                              batch_size=self.batch_size)

    def track_learning_progress(self):
        """
        Track some results during training,

        :return:
        """
        pass

    # TODO: How can I access self.markets[market_id] in agent?
    # TODO: Can I use on quote to iterate over the steps of the agent?
    # TODO: Logik von this state und next state... (Action und reward vergleichen...)
    def on_quote(self, market_id: str, book_state: pd.Series):

        # OBSERVATION SPACE
        # TODO: Normalize
        self.obs = np.array(book_state[1:], dtype='float32')  # without timestamp
        # print(len(self.obs))
        # normalization mit tickprize? / min-max
        # oder zscore mit 5 tagen

        # ACTION
        # pass the state to the ddqn to get the action
        # TODO: Which observation should be passed (current or last?)
        action = self.ddqn.epsilon_greedy_policy(self.obs.reshape(-1, self.state_dim))
        self.actions.append(action)
        # print('action: ',action)

        # DONE (Problem: Done is not necessary in our env...)
        # done = backtest.done
        # print('DONE', str(done))
        done = 0

        # take step
        # next_state, reward, done, _ = self.trading_environment.step(action)

        # REWARD (PnL)
        # pnl_realized = self.market_interface.pnl_realized_total
        # self.pnls.append(pnl_realized)
        # pnl_unrealized = self.market_interface.pnl_unrealized_total
        # print(pnl_realized, pnl_unrealized)
        pnl_new = (self.market_interface.pnl_unrealized_total + self.market_interface.pnl_realized_total)
        pnl_diff = pnl_new - self.pnl_old
        reward = pnl_diff
        # print(reward)
        # for next iteration
        self.pnl_old = pnl_new

        # INFO
        info = {'reward', reward}

        # memorize last action/reward/state together with new state
        self.ddqn.memorize_transition(self.last_obs,  # old state
                                      self.last_action,
                                      self.last_reward,
                                      # new state
                                      self.obs,
                                      0.0 if done else 1.0)

        # train
        if self.ddqn.train:
            # run experience replay until done
            self.ddqn.experience_replay()
        # if done:
        #    break

        # save observation for next iteration
        self.last_obs = self.obs
        self.last_action = action
        self.last_reward = reward

        # TAKE ACTION (Action has to be defined in RL model)
        if self.trading_phase:
            # actions space:
            # 0: short
            # 1: hold
            # 2: long

            # for testing purposes: generate random action:
            # action = np.random.randint(0, 3)
            # TODO: Close open position before taking the oposite side...
            if not self.market_interface.get_filtered_orders(
                    market_id, status="ACTIVE"):

                # action == 0: submit market sell order
                if action == 0:
                    self.market_interface.submit_order(market_id, "sell", self.quantity)

                # action == 2: submit market buy order
                elif action == 2:
                    self.market_interface.submit_order(market_id, "buy", self.quantity)

                # action == 1: wait
                else:
                    pass

    def on_trade(self, market_id: str, trades_state: pd.Series):
        pass

    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):
        """
         Provides the timestamps
         for both current and next iteration. The given interval may be used to
         submit orders before a specific point in time.
         """

        # reset rl trading env in the beginning of each new episode:

        # TODO: Better count the steps in on_qote... nothing happens in on_trade anyways
        # print(self.step)
        if self.step == 0:
            # TODO: Reset Trading Environment (Simulator stuff and Observation)
            # viel wird sowieso resetted da der agent nach jeder episode in backtest resetted wird.
            print('Reset Trading Environment')
        self.step = self.step + 1

        trading_time = timestamp.time() > self.start_time and \
                       timestamp.time() < self.end_time

        # Enter trading phase if
        # (1) current time in defined trading_time
        # (2) trading_phase is False up to now
        if trading_time and not self.trading_phase:
            print('Algo is now able to trade...')
            self.trading_phase = True

            # TODO: can trading_environment be resetted here before each new episode?
            # this_state = trading_environment.reset()

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


# run agent
if __name__ == "__main__":
    identifier_list = ["Adidas.BOOK", "Adidas.TRADES"]

    agent = RLAgent(
        name="RLAgent",
        quantity=100
    )

    # Instantiate RLBacktest here
    backtest = Backtest(agent=agent)

    # Option 1: run agent against a series of generated episodes, that is,
    # generate episodes with the same episode_buffer and episode_length
    # TODO: if status verstehen
    backtest.run_rl_trainingloop(identifier_list=identifier_list,
                                 date_start="2021-01-04",  # start date after which episodes are generated
                                 date_end="2021-01-05",  # end date before which episodes are generated
                                 episode_interval=20,  # e.g. 30 -> 30 min episodes, 60 -> 60 min episodes and so on
                                 episode_shuffle=True,  # ?
                                 episode_buffer=1,  # minutes
                                 # TODO: Can I use episode_length as trading_period_length?
                                 episode_length=5,  # minutes
                                 num_episodes=5,  # does this mean only n episodes should run?
                                 )