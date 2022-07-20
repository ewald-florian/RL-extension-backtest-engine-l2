
import numpy as np
import pandas as pd
import datetime
import copy
import random
import logging
import textwrap
logging.basicConfig(level=logging.CRITICAL)

from env.rlreplay import Episode
from env.market import MarketState, Order, Trade
from context.context import MarketContext, ObservationSpace, AgentContext
import gym
from gym import spaces

class TradingEnvironment(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, env_config:dict):
        """
        Trading environment for reinforcement learning,
        follows the openAI gym structure.
        :param env_config
                dict, contains configuration variables
        """
        self.replay = env_config.get("config").get("replay")
        #self.agent = env_config.get("config").get("agent")
        # todo: AgentInterface in config übergeben
        self.agent_interface = AgentInterface()

        self.action_space = spaces.Discrete(3)
        # TODO:
        self.observation_space = spaces.Box(np.zeros(40), np.array([10_000]*40))

    def step(self, action):
        """
        Executes a step in the environment by applying an action.
        Transitions the environment to the next state.
        Returns the new observation, reward, completion status, and other info.
        :param action:
            int, (format depends on action_space)
        :return: observation
            np.array, (format depends on observation space)
        :return: reward
            The reward from the environment after executing the action
            that was given as the input
        :return: done
            bool, True if episode is complete, False otherwise
        :return: info
            dict, contains further info on the environment, can be empty
        """
        assert self.action_space.contains(action), "Invalid Action"
        # 1) take action
        # self.agent.take_action(action=action)
        self.agent_interface.take_action(action=action)
        # 2) call replay.step()
        #reward, market_obs, update_store, timestamp, timestamp_next = self.replay.step()
        reward, market_obs = self.replay.step()
        # 4) returns:
        # done
        done = self.replay.done
        # info
        info = {}
        return market_obs, reward, done, info

    def reset(self):
        """
        Resets the environment to its initial state, and returns the
        observation of the environment corresponding to the initial state.
        Has to be called to start a new episode.
        :return: first_obs
            np.array, first observation of the new episode
            (format depends on observation space)
        """
        self.replay.reset()
        #self.agent.reset()
        self.replay.step()
        print("..reset for new episode")
        # return first observation
        # todo: first_obs should be returned from replay.reset()...
        first_obs = self.replay.market_state.copy()
        return first_obs

    def render(self):
        pass

    def seed(self):
        pass

class Replay:

    def __init__(self, config_dict=None):

        self.result_list = []
        self.display_interval = 10

        # TODO: should MarketContext be independent of replay?
        # instantiate market_context
        self.market_ctx = MarketContext()
        self.observation_space = ObservationSpace()
        self.agent_context = AgentContext()

        # default configuration for simulation
        config = {"identifier_list":
                            ["Adidas.BOOK", "Adidas.TRADES"],
                               "date_start": "2021-01-04",
                               "date_end": "2021-01-20",
                               "episode_interval": 5,
                               "episode_shuffle": True,
                               "episode_buffer": 5,
                               "episode_length": 10,
                               "num_episodes": 20
                               }
        # update config whith custom configuration
        if config_dict:
            config.update(config_dict)

        # todo: .get()
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

        #TODO: can I avoid this cross dependency?
        self.stats = TradingStatistics()
        #TODO: Should reward be independent instance?
        self.reward = Reward()

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
        self.episode_index = 0

    def reset(self):

        self._build_new_episode()
        self._reset_market()
        # reset stats
        self.stats.reset()

    # former reset_before_run()
    def _build_new_episode(self):
        """
        Builds next Episode.
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
            print("No Episode was build")
            return  # do nothing

        # set done flag to false (will be set true in Replay)
        # if step > max_steps
        self.done = False
        # Adds 1 to episode_counter after each Episode
        self.episode_counter += 1
        # TODO: episode_index if Bedingung (nur wenn Episode erfolgreich gebaut werden konnte...)
        # in reset before run?
        self.episode_index += 1
        # print (note that this actually prints the "next" episode...
        print('(ENV) EPISODE COUNTER:', self.episode_counter)

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

    def step(self):

        # use episode.step and episode.__len__ to manage done flag:
        if self.episode.step >= self.episode.__len__()-1:
            self.done = True
            print('(ENV) DONE')

        #TODO: Where is the best way to call tradingStats?
        # replay is convenient since it already contains a step counter which
        # can be used with display interval...
        if self.episode.step % self.stats.display_interval == 1:
            print(self.stats)

        # include try-except statement when building new episode:
        try:

            # call episode.__next__() to get next data update
            update_store = self.episode.__next__()
            # update global timestamp
            self.__class__.timestamp_global = self.episode.timestamp

            # Update MarketState
            market_list = set(identifier.split(".")[0] for identifier in update_store)
            source_list = list(update_store)

            # e.g. 'Adidas.BOOK', second would be sometimes 'Adidas.TRADES'
            source_id = source_list[0]
            # save book as array without timestamps and labels
            market_state = update_store.get(source_id).array[1:]
            # convert to float (for model)
            self.market_state = market_state.astype('float32')

            # update market
            for market_id in market_list:

                self._market_step(market_id=market_id,
                                  book_update=update_store.get(f"{market_id}.BOOK"),
                                  trade_update=update_store.get(f"{market_id}.TRADES", pd.Series([None] * 3)),
                                  # optional, default to empty pd.Series
                                  )
            # MARKET CONTEXT
            #TODO: should I let the agent wait for the first n
            # episodes until market_context is complete?

            # store market_state in to market_context
            self.market_ctx.store_market_context(market_state)
            #print('MARKET CONTEXT')
            #print('MC Length', len(self.market_ctx.market_context))
            #print(self.market_ctx.market_context)

            # MARKET OBSERVATION
            market_observation = self.observation_space.create_market_observation(self.market_ctx.market_context)
            #print("MARKET OBSERVATION")
            #print(market_observation)

            # REWARD
            reward = self.reward.compute_reward()
            #print('REWARD: ', reward)

            # AGENT CONTEXT
            agent_state = self.agent_context.get_agent_state()
            print("TYPE:", type(agent_state))
            print("AGENT STATE")
            print(agent_state)



            return reward, market_observation

        except StopIteration:
            print("Iteration exhausted")


# TODO: Implement proper solution of agent/agent interface...

# Ich brauche keinen base agent sondern nur das market interface
# Ich brauche das market interface

class AgentInterface:

    def __init__(self):
        # instantiate market interface
        self.market_interface = MarketInterface(latency=10)

    def take_action(self,action):
        # TODO: remove market_id hardcode...
        if action == 0:  # 0:sell
            self.market_interface.submit_order('Adidas', "sell", 100)
        elif action == 2:  # 2:buy
            self.market_interface.submit_order('Adidas', "buy", 100)
        else:  # 1:wait
            pass

    def reset(self):
        self.__init__()

    def __str__(self):
        pass

# TODO: Reset Method (actually not needed since the class does not really store anything..)
class MarketInterface:

    def __init__(self,
                 latency: int = 10,  # in us (microseconds)
                 ):
        """
        The market interface is used to interact with the market, that is,
        using the following methods ...
        - `submit_order`: ...
        - `cancel_order`: ...
        ... to submit and cancel specific orders.

        :param latency:
            int, latency before order submission (in us), default is 10
        """
        # settings
        self.latency = latency  # in microseconds ("U"), used only in submit method

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
            timestamp=Replay.timestamp_global + pd.Timedelta(self.latency, "us"),  # microseconds
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

class TradingStatistics: # basically the second part of MarketInterface

    def __init__(self,
                 display_interval = 100,
                 exposure_limit: float = 1_000_000,
                 # TODO: avoid double assigning of latency in interface and stats...
                 latency: int = 10,
                 transaction_cost_factor: float = 1e-3,  # 10 bps
                 ):

        # containers for related class instances
        self.market_state_list = MarketState.instances
        self.order_list = Order.history
        self.trade_list = Trade.history

        # settings
        self.exposure_limit = exposure_limit  # ...
        self.transaction_cost_factor = transaction_cost_factor  # in bps
        self.latency = latency
        self.display_interval = display_interval

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

    # basically the original agent string representation
    def __str__(self):
        """
        String representation.
        """
        # read global timestamp from Replay class attribute
        timestamp_global = Replay.timestamp_global

        # string representation
        string = f"""
           ---
           timestamp:      {timestamp_global} (+{self.latency} ms)
           ---
           exposure:       {self.exposure_total}
           pnl_realized:   {self.pnl_realized_total}
           pnl_unrealized: {self.pnl_unrealized_total}
           ---
           """
        return textwrap.dedent(string)

    def reset(self):
        """
        Reset TradingStats to the original state.
        Should be called before each new episode.
        """
        self.__init__(self.display_interval,
                     self.exposure_limit,
                     self.latency,
                     self.transaction_cost_factor)

class Reward:

    def __init__(self,
                 # TODO: avoid double assigning of latency in interface and stats...
                 latency: int = 10,
                 transaction_cost_factor: float = 1e-3,  # 10 bps
                 ):

        # containers for related class instances
        self.market_state_list = MarketState.instances
        self.order_list = Order.history
        self.trade_list = Trade.history

        # settings
        self.transaction_cost_factor = transaction_cost_factor  # in bps
        self.latency = latency

    def compute_reward(self):
        """
        Compute the reward according to individual reward function.
        :return: reward
            float, reward for RL environment
        """
        # for testing: just use pnl_realized as reward
        pnl_realized = self.pnl_realized()
        key = list(pnl_realized.keys())[0]
        reward = pnl_realized.get(key)
        return reward

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

    def reset(self):
        """
        Reset TradingStats to the original state.
        Should be called before each new episode.
        """
        self.__init__(self.latency,
                     self.transaction_cost_factor)