import numpy as np
import pandas as pd
from agent.agent import BaseAgent
from env.replay import Backtest
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#TODO: Implement AgentState
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
    """
    def __init__(self, n_market_states=1):
        self.n_states = n_market_states
        self.market_state_list = []
        self.market_context = np.array([])

    # TODO: __call__ Method?

    def store_market_context(self, book_state):
        """
        Store the last n_market_states into the market_context
        array.
        """
        market_state = np.array(book_state[1:], dtype='float64')
        self.market_state_list.append(market_state)
        self.market_state_list = self.market_state_list[-self.n_states:]
        self.market_context = np.array(self.market_state_list, dtype='float64')
        # self.market_context = self.normalize_market_state(self.market_context)

    def normalize_market_state(self, data):
        """
        Normalize market_context array.
        """
        # TODO: Find proper normalization approach
        # For testing: use min max scaler (does not really make sense...)
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        return scaled_data


    def reset(self):
        """
        Reset market_context
        """
        self.market_state_list = []
        self.market_state = np.array([])

#TODO: Implement ObservationFilter
class ObservationFilter(): # Space
    """
    Take market state and agent state as input and return
    observation in a format that fits the NN model as input.
    """
    def __init__(self):
        pass

    def create_market_observation(self, market_context):
        """
        Adjust market_state to a format that fits tf (2D array)
        if necessary.
        """

        self.market_obs = market_context

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




