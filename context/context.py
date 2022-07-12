import numpy as np
import pandas as pd
from agent.agent import BaseAgent
from env.replay import Backtest
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#TODO: Implement AgentState (dont use base agent..., use the original methods of market interface)
class AgentState(BaseAgent):
    # einzelne datenstruktur mit einzelner methode, zB update oder nur datenstruktur
    # pnl etc direkt im base agent oder in seperater metric class

    # AgentState: update einen array mit orders und trades (standing orders (ACTIVE), offener positionen

    def __init__(self):

        self.current_state = None # What is the best form? dict? list?
        pass

    def receive_filtered_trades(self):
        """
        Get the orders.
        :return:
        """
        # trade_list = self.market_interface.get_filtered_trades(market_id, side='buy')

        pass
    def receive_filtered_orders(self):
        """
        Get the trades.

        :return:
        """
        # order_list = self.market_interface.get_filtered_orders(market_id, side='buy', status=None)
        pass
    def current_state(self):
        """
        Save orders and trades in a current_agent_state variable.

        :return:
        """

        # self.current_state = None
        pass

#TODO: Implement AgentContext
class AgentContext(): # keine vererbung notwendig, observer pattern?
    """
    Manage agent context.
    """

    def __init__(self, n_agent_states=100):
        self.agent_state_list = []
        self.n_states = n_agent_states
        pass

    def store_agent_state(self):
        """
        Store a number of agent states to a list.

        :return:
        """
        #self.agent_state_list(self.current_state)
        #self.agent_state_list = self.agent_state_list[-self.n_agent_states:]

        pass

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




