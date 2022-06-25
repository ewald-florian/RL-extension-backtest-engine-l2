"""
NOTE
- Add BaseAgent
- Add RLAgent
- to instantiate RLAgent directly in TradingEnv
- and to avoid circular imports
"""
# TODO: Backtest input variablen logik
# TODO: Mit SimpleTradingEnvironment Logik testen

#from env.market import MarketState, Order, Trade
from env.replay import Episode
from copy import deepcopy
#from rlagent.rlagent_v2 import RLAgent


# general imports
import copy
import datetime
import logging
logging.basicConfig(level=logging.CRITICAL)
import pandas as pd
import random
random.seed(42)

import gym
from gym import spaces
import numpy as np

#env_config = {"agent":agent,
#              "config_dict":None}

class TradingEnvironment(gym.Env):

    metadata = {'render.modes': ['human']}

    #def __init__(self, agent=None, config_dict=None):
    def __init__(self, config=None):

        #agent = config["agent"]
        #config_dict = env_config["config_dict"]

        agent = config.get("agent")
        config_dict = config.get("config_dict")

        # instantiate TradingEnv (the replay/simulation class)
        self.simulator = TradingSimulator()#agent=agent, config_dict=config_dict)
        # gym
        self.action_space = spaces.Discrete(3)
        # TODO: plausible min and max ranges
        self.observation_space = spaces.Box(np.zeros(40), np.array([10_000]*40))

    def step(self,action):
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        #TODO: which is the correct order???
        # I would rather say 1) action, 2) step...(the action has to affect
        # the environment before we want to observe the new environment?
        self.take_action(action)
        obs = self.simulator.take_step()
        done = self.simulator.replay_data.done
        # TODO: proper info dict (e.g. with infos from simulator)
        info = {}
        # TODO: proper reward function
        reward = self.simulator.agent.market_interface.pnl_realized_total

        return obs, reward, done, info

    def reset(self):
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

    # this is usually an external function...
    def compute_action(self):
        pass

    # TODO: implement properly (where is the best place? in simulator)
    def take_action(self,action):

        if action == 0:  # 0:sell
            self.simulator.agent.market_interface.submit_order('Adidas', "sell", 100)
        elif action == 2:  # 2:buy
            self.simulator.agent.market_interface.submit_order('Adidas', "buy", 100)
        else:  # 1:wait
            pass

class ReplayData:

    def __init__(self,config_dict=None

                 # TODO: backtest config file
                 #identifier_list: list = ["Adidas.BOOK", "Adidas.TRADES"],
                 #date_start: str = "2021-01-04",
                 #date_end: str = "2021-01-08",
                 #episode_interval: int = 5,
                 #episode_shuffle: bool = True,
                 #episode_buffer: int = 5,
                 #episode_length: int = 10,
                 #num_episodes: int = 10

                 ):
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

    # TODO: Könnte die erste obs returned werden?
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


# Note: Call TradingSimulator() instead of Backtest() in BaseAgent
class TradingSimulator():
    """
    Gym Environment for Backtest Engine.
    """
    timestamp_global = None

    def __init__(self): #, config_dict=None):#agent=None

        # TODO: default_agent:
        self.agent = RLAgent('name', 100)
        # from arguments
        # todo deepcopy
        self._agent = deepcopy(self.agent)
        self.result_list = []
        self.display_interval=10

        # TODO: Organisieren wo agent instanziiert und resetted wird (extern? Agent class?)

        # instantiate replay data class with config_dict
        self.replay_data = ReplayData(config_dict=None)



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


################################ agent_rl_env.py #############
from env.market import MarketState, Order, Trade
#from gym_linkage.tradingenv_v7 import TradingSimulator

# general imports
import abc
import pandas as pd
import textwrap


class BaseAgent(abc.ABC):

    def __init__(self, name):
        """
        Trading agent base class. Subclass BaseAgent to define how a concrete
        Agent should act given different market situations.

        :param name:
            str, agent name
        """

        # agent has market access via market_interface instance
        self.market_interface = MarketInterface(
            exposure_limit=1e6,  # ...
            latency=10,  # in us (microseconds)
            transaction_cost_factor=1e-3,  # 10 bps
        )

        # ...
        self.name = name

    # event management ---

    @abc.abstractmethod
    def on_quote(self, market_id: str, book_state: pd.Series):
        """
        This method is called after a new quote.

        :param market_id:
            str, market identifier
        :param book_state:
            pd.Series, including timestamp, bid/ask price/quantity for ten levels
        """

        raise NotImplementedError("To be implemented in subclass.")

    @abc.abstractmethod
    def on_trade(self, market_id: str, trade_state: pd.Series):
        """
        This method is called after a new trade.

        :param market_id:
            str, market identifier
        :param trade_state:
            pd.Series, including timestamp, price, quantity
        """

        raise NotImplementedError("To be implemented in subclass.")

    @abc.abstractmethod
    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):
        """
        This method is called with every iteration and provides the timestamps
        for both current and next iteration. The given interval may be used to
        submit orders before a specific point in time.

        :param timestamp:
            pd.Timestamp, timestamp recorded in this iteration
        :param timestamp_next:
            pd.Timestamp, timestamp recorded in next iteration
        """

        raise NotImplementedError("To be implemented in subclass.")

    def __str__(self):
        """
        String representation.
        """

        # read global timestamp from Backtest class attribute
        # TODO:...
        timestamp_global = TradingSimulator.timestamp_global

        # string representation
        string = f"""
        ---
        timestamp:      {timestamp_global} (+{self.market_interface.latency} ms)
        ---
        exposure:       {self.market_interface.exposure_total}
        pnl_realized:   {self.market_interface.pnl_realized_total}
        pnl_unrealized: {self.market_interface.pnl_unrealized_total}
        ---
        """

        return textwrap.dedent(string)

    def reset(self):
        """
        Reset agent.
        """

        # ...
        return self.__init__(self.name)


class MarketInterface:

    def __init__(self,
                 exposure_limit: float = 1e6,
                 latency: int = 10,  # in us (microseconds)
                 transaction_cost_factor: float = 1e-3,  # 10 bps
                 ):
        """
        The market interface is used to interact with the market, that is,
        using the following methods ...

        - `submit_order`: ...
        - `cancel_order`: ...
        - `get_filtered_orders`: ...
        - `get_filtered_trades`: ...

        ... to submit and cancel specific orders.

        :param latency:
            int, latency before order submission (in us), default is 10
        :param transaction_cost_factor:
            float, transcation cost factor per trade (in bps), default is 10
        """

        # containers for related class instances
        self.market_state_list = MarketState.instances
        self.order_list = Order.history
        self.trade_list = Trade.history

        # settings
        self.exposure_limit = exposure_limit  # ...
        self.latency = latency  # in microseconds ("U"), used only in submit method
        self.transaction_cost_factor = transaction_cost_factor  # in bps

    # order management ---

    def submit_order(self, market_id, side, quantity, limit=None):
        """
        Submit market order, limit order if limit is specified.

        Note that, for convenience, this method also returns the order
        instance that can be used for cancellation.

        :param market_id:
            str, market identifier
        :param side:
            str, either 'buy' or 'sell'
        :param quantity:
            int, number of shares ordered
        :param limit:
            float, limit price to consider, optional
        :return order:
            Order, order instance
        """

        # submit order
        order = Order(
            timestamp=TradingSimulator.timestamp_global + pd.Timedelta(self.latency, "us"),  # microseconds
            market_id=market_id,
            side=side,
            quantity=quantity,
            limit=limit,
        )

        return order

    def cancel_order(self, order):
        """
        Cancel an active order.

        :param order:
            Order, order instance
        """

        # cancel order
        order.cancel()

    # order assertion ---

    def _assert_exposure(self, market_id, side, quantity, limit):
        """
        Assert agent exposure. Note that program execution is supposed to
        continue.
        """

        # first, assert that market exists
        assert market_id in self.market_state_list, \
            "market_id '{market_id}' does not exist".format(
                market_id=market_id,
            )

        # calculate position value for limit order
        if limit:
            exposure_change = quantity * limit
        # calculate position value for market order (estimated)
        else:
            exposure_change = quantity * self.market_state_list[market_id].mid_point

        # ...
        exposure_test = self.exposure.copy()  # isolate changes
        exposure_test[market_id] = self.exposure[market_id] + exposure_change * {
            "buy": + 1, "sell": - 1,
        }[side]
        exposure_test_total = round(
            sum(abs(exposure) for _, exposure in exposure_test.items()), 3
        )

        # ...
        assert self.exposure_limit >= exposure_test_total, \
            "{exposure_change} exceeds exposure_left ({exposure_left})".format(
                exposure_change=exposure_change,
                exposure_left=self.exposure_left,
            )

    # filtered orders, trades ---

    def get_filtered_orders(self, market_id=None, side=None, status=None):
        """
        Filter Order.history based on market_id, side and status.

        :param market_id:
            str, market identifier, optional
        :param side:
            str, either 'buy' or 'sell', optional
        :param status:
            str, either 'ACTIVE', 'FILLED', 'CANCELLED' or 'REJECTED', optional
        :return orders:
            list, filtered Order instances
        """

        orders = self.order_list

        # orders must have requested market_id
        if market_id:
            orders = filter(lambda order: order.market_id == market_id, orders)
        # orders must have requested side
        if side:
            orders = filter(lambda order: order.side == side, orders)
        # orders must have requested status
        if status:
            orders = filter(lambda order: order.status == status, orders)

        return list(orders)

    def get_filtered_trades(self, market_id=None, side=None):
        """
        Filter Trade.history based on market_id and side.

        :param market_id:
            str, market identifier, optional
        :param side:
            str, either 'buy' or 'sell', optional
        :return trades:
            list, filtered Trade instances
        """

        trades = self.trade_list

        # trades must have requested market_id
        if market_id:
            trades = filter(lambda trade: trade.market_id == market_id, trades)
        # trades must have requested side
        if side:
            trades = filter(lambda trade: trade.side == side, trades)

        return list(trades)

    # symbol, agent statistics ---

    @property
    def exposure(self, result={}):
        """
        Current net exposure that the agent has per market, based statically
        on the entry value of the remaining positions.

        Note that a positive and a negative value indicate a long and a short
        position, respectively.

        :return exposure:
            dict, {<market_id>: <exposure>, *}
        """

        for market_id, _ in self.market_state_list.items():

            # trades filtered per market
            trades_buy = self.get_filtered_trades(market_id, side="buy")
            trades_sell = self.get_filtered_trades(market_id, side="sell")

            # quantity per market
            quantity_buy = sum(t.quantity for t in trades_buy)
            quantity_sell = sum(t.quantity for t in trades_sell)
            quantity_unreal = quantity_buy - quantity_sell

            # case 1: buy side surplus
            if quantity_unreal > 0:
                vwap_buy = sum(t.quantity * t.price for t in trades_buy) / quantity_buy
                result_market = quantity_unreal * vwap_buy
            # case 2: sell side surplus
            elif quantity_unreal < 0:
                vwap_sell = sum(t.quantity * t.price for t in trades_sell) / quantity_sell
                result_market = quantity_unreal * vwap_sell
            # case 3: all quantity is realized
            else:
                result_market = 0

            result[market_id] = round(result_market, 3)

        return result

    @property
    def exposure_total(self):
        """
        Current net exposure that the agent has across all markets, based on
        the net exposure that the agent has per market.

        Note that we use the absolute value for both long and short positions.

        :return exposure_total:
            float, total exposure across all markets
        """

        result = sum(abs(exposure) for _, exposure in self.exposure.items())
        result = round(result, 3)

        return result

    @property
    def pnl_realized(self, result={}):
        """
        Current realized PnL that the agent has per market.

        :return pnl_realized:
            dict, {<market_id>: <pnl_realized>, *}
        """

        for market_id, _ in self.market_state_list.items():

            # trades filtered per market
            trades_buy = self.get_filtered_trades(market_id, side="buy")
            trades_sell = self.get_filtered_trades(market_id, side="sell")

            # quantity per market
            quantity_buy = sum(t.quantity for t in trades_buy)
            quantity_sell = sum(t.quantity for t in trades_sell)
            quantity_real = min(quantity_buy, quantity_sell)

            # case 1: quantity_real is 0
            if not quantity_real:
                result_market = 0
            # case 2: quantity_real > 0
            else:
                vwap_buy = sum(t.quantity * t.price for t in trades_buy) / quantity_buy
                vwap_sell = sum(t.quantity * t.price for t in trades_sell) / quantity_sell
                result_market = quantity_real * (vwap_sell - vwap_buy)

            result[market_id] = round(result_market, 3)

        return result

    @property
    def pnl_realized_total(self):
        """
        Current realized pnl that the agent has across all markets, based on
        the realized pnl that the agent has per market.

        :return pnl_realized_total:
            float, total realized pnl across all markets
        """

        result = sum(pnl for _, pnl in self.pnl_realized.items())
        result = round(result, 3)

        return result

    @property
    def pnl_unrealized(self, result={}):
        """
        This method returns the unrealized PnL that the agent has per market.

        :return pnl_unrealized:
            dict, {<market_id>: <pnl_unrealized>, *}
        """

        for market_id, market in self.market_state_list.items():

            # trades filtered per market
            trades_buy = self.get_filtered_trades(market_id, side="buy")
            trades_sell = self.get_filtered_trades(market_id, side="sell")

            # quantity per market
            quantity_buy = sum(t.quantity for t in trades_buy)
            quantity_sell = sum(t.quantity for t in trades_sell)
            quantity_unreal = quantity_buy - quantity_sell

            # case 1: buy side surplus
            if quantity_unreal > 0:
                vwap_buy = sum(t.quantity * t.price for t in trades_buy) / quantity_buy
                result_market = abs(quantity_unreal) * (market.best_bid - vwap_buy)
            # case 2: sell side surplus
            elif quantity_unreal < 0:
                vwap_sell = sum(t.quantity * t.price for t in trades_sell) / quantity_sell
                result_market = abs(quantity_unreal) * (vwap_sell - market.best_ask)
            # case 3: all quantity is realized
            else:
                result_market = 0

            result[market_id] = round(result_market, 3)

        return result

    @property
    def pnl_unrealized_total(self):
        """
        Current unrealized pnl that the agent has across all markets, based on
        the unrealized pnl that the agent has per market.

        :return pnl_unrealized_total:
            float, total unrealized pnl across all markets
        """

        result = sum(pnl for _, pnl in self.pnl_unrealized.items())
        result = round(result, 3)

        return result

    @property
    def exposure_left(self):
        """
        Current net exposure left before agent exceeds exposure_limit.

        :return exposure_left:
            float, remaining exposure
        """

        # TODO: include self.pnl_realized_total?
        result = self.exposure_limit - self.exposure_total
        result = round(result, 3)

        return result

    @property
    def transaction_cost(self):
        """
        Current trading cost based on trade history, accumulated throughout
        the entire backtest.

        :transaction_cost:
            float, accumulated transaction cost
        """

        result = sum(t.price * t.quantity for t in self.trade_list)
        result = result * self.transaction_cost_factor
        result = round(result, 3)

        return result

############ RLAgent #############
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
        self.market_interface.transaction_cost_factor = 0

    def on_quote(self, market_id: str, book_state: pd.Series):
        pass

    def on_trade(self, market_id: str, trades_state: pd.Series):
        pass

    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):
        pass

