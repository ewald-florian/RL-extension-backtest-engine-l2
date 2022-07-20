import numpy as np
import pandas as pd

# TODO: Warum bekomme ich hier keinen error für zirkuläre importe?
from env.market import MarketState, Order, Trade

#TODO: Implement AgentContext
class AgentContext:

    def __init__(self, n_states=10):

        # containers for related class instances
        self.n_states = n_states
        self.market_state_list = MarketState.instances
        self.order_list = Order.history
        self.trade_list = Trade.history
        self.agent_state = {}
        self.agent_context = []

    def _get_filtered_orders(self, market_id=None, side=None, status=None):
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

    def _get_filtered_trades(self, market_id=None, side=None):
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

    # TODO: how should trades and orders be stored? only the specific market_id or the entire thing?
    def get_agent_state(self):

        for market_id, _ in self.market_state_list.items():
            # trades filtered per market
            trades_buy = self._get_filtered_trades(market_id, side="buy")
            trades_sell = self._get_filtered_trades(market_id, side="sell")

            orders_buy = self._get_filtered_orders(market_id, side="buy")
            orders_sell = self._get_filtered_orders(market_id, side="sell")

        # store agent_state to dictionary
        agent_state = {"trades_buy": trades_buy,
                       "trades_sell": trades_sell,
                       "orders_buy": orders_buy,
                       "orders_sell": orders_sell,
        }

        self.agent_state = agent_state

        return self.agent_state

    # TODO: is there even a usecase for storing several agent states?
    def store_agent_context(self):

        self.agent_context.append(self.agent_state)
        self.agent_context = self.agent_context[-self.n_states:]


    def reset(self):
        """
        Reset AgentContext
        """
        self.__init__()

class MarketContext():
    """
    Market context for RL-Training.
    :param n_market_states
        int, number of market states to be stored in market_context.
    """
    def __init__(self, n_market_states=20):
        self.n_states = n_market_states
        self.market_state_list = []
        self.market_context = np.array([])

    # TODO: __call__ Method?
    def store_market_context(self, book_state):
        """
        Store the last n_market_states into the market_context
        array. The market_context array can be used in
        ObservationSpace to generate observations.
        """
        #market_state = np.array(book_state[1:], dtype='float32') # without timestamp
        market_state = np.array(book_state, dtype='float32')
        self.market_state_list.append(market_state)
        self.market_state_list = self.market_state_list[-self.n_states:]
        self.market_context = np.array(self.market_state_list, dtype='float32')

    def reset(self):
        """
        Reset market_context.
        """
        self.market_state_list = []
        self.market_state = np.array([])

#TODO: Take the market context and design features based on it, do not provovide the
# entire market_context array as input to the network but only the most recent market_update
# together with some designed features (e.g. average spread over the last x updates)
class ObservationSpace(): # Space
    """
    Take market_context and agent_context as input and return
    observation in a format that fits the NN model as input.
    """
    def __init__(self):
        pass
    # TODO: check if the features are computed with the correct columns
    def create_market_observation(self, market_context):
        """
        Adjust market_state to a format that fits tf (2D array)
        if necessary.
        """
        # normalization
        # TODO: Apply proper normalization
        market_context_norm = market_context.copy()
        # 1) scale prices
        min_price = market_context_norm[:, ::2].min()
        max_price = market_context_norm[:, ::2].max()
        scaled_prices = (market_context_norm[:, ::2] - min_price) / (max_price - min_price)
        # 2) scale quantities
        min_q = market_context_norm[:, 1::2].min()
        max_q = market_context_norm[:, 1::2].max()
        scaled_quantities = (market_context_norm[:, 1::2] - min_q) / (max_q - min_q)
        # 3) overwrite context with scaled values
        market_context_norm[:, ::2] = scaled_prices
        market_context_norm[:, 1::2] = scaled_quantities

        # use only the scaled current market state as observation
        market_observation = market_context_norm[-1,:]
        market_observation = market_observation.astype('float32')

        # TODO: compute further features
        # "To make the orderbook comparable fordifferent stocks,heights are normalized by the price gap between the
        # tenth price and the mid-price." Raja Velu p.287
        # Problem: What if the difference is below 1?
        # midpoint
        midpoint = (market_context[:,0] + market_context[:,2]) / 2
        # relative spread
        spread = market_context[:,2] - market_context[:, 0]
        relative_spread = spread / midpoint
        # quote imbalance
        # note: (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)
        best_ask_vol = market_context[:,1]
        best_bid_vol = market_context[:,3]
        # TODO: can be negative --> adjust observation_space
        quote_imbalance = (best_bid_vol - best_ask_vol) + (best_bid_vol + best_ask_vol)
        # return only one context row for testing purposes
        return market_observation

    def create_agent_observation(self):
        """
        Adjust agent_state to a format that fits tf (2D array)
        if necessary.
        """
        pass

    def create_observation(self):
        """
        Combine market_obs and agent_obs to one array which
        can be fed into the NN.
        """
        pass




