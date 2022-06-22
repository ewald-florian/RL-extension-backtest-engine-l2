from agent.agent import BaseAgent
from env.replay import Backtest
import datetime
import numpy as np

# RL imports
from model.model import DDQNModel
from context.context import MarketContext, ObservationFilter
from model.trainer import Trainer
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import tensorflow as tf



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
        self.gamma = gamma,  # discount factor (usually 1 for trading applications)
        self.tau = tau  # target network update frequency
        self.architecture = architecture  # units per layer
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg  # L2 regularization
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay

        self.context = MarketContext()

        # action
        self.pnl_old = 0
        self.last_obs = np.zeros(40, dtype='object')
        self.last_action = 1
        self.last_reward = 0

        # TODO: Solve Hardcode
        self.state_dim = 40
        self.num_actions = 3

        # dynamic attributes
        self.book_state_list = []
        self.obs = np.array([], dtype='float32')

        self.step = 0
        self.actions = []  # better array?
        self.pnls = []
        self.positions = []  # should I store long/short?
        self.trades = []

        # initialize RL model
        self.initialize_model()
        # TODO: Should trainer be initialized seperately?
        self.trainer = Trainer()
        # check:
        print('Architecture: ', self.ddqn.architecture)

    # TODO: Initialize Model in backtest not in agent (or it is resetted every episode)
    def initialize_model(self):
        """
        Initialize DDQN Model
        """
        # hardcode for testing:
        state_dim = 40  # 10 levels*2 sides* 2 variables --> (40,)
        num_actions = 3  # 0,1,2
        # free up memory
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

    #TODO: track learning process / logger
    def track_learning_progress(self):
        pass

    def on_quote(self, market_id: str, book_state: pd.Series):

        # TODO: Call market_context
        #self.context.store_market_context(book_state=book_state)
        #self.obs = self.context.market_context

        # OBSERVATION SPACE
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
        # TODO: Do we need done?

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
        # TODO: Do we need info?

        # memorize last action/reward/state together with new state
        self.ddqn.memorize_transition(self.last_obs,  # old state
                                      self.last_action,
                                      self.last_reward,
                                      self.obs, # new state
                                      0.0 if done else 1.0)

        # train
        if self.ddqn.train:
            # run experience replay until done
            self.ddqn.experience_replay()
        #if done:
        #    break

        # save observation for next iteration
        self.last_obs = self.obs
        self.last_action = action
        self.last_reward = reward

        # TODO: ActionFilter/ACTIONSPACE class
        # TAKE ACTION

        # actions space:
        # 0: short
        # 1: hold
        # 2: long

        # TODO: Close open position before taking the opposite side...
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
        pass


# run agent
if __name__ == "__main__":

    identifier_list = ["Adidas.BOOK", "Adidas.TRADES"]

    agent = RLAgent(
        name="RLAgent",
        quantity=100
    )

    # Instantiate RLBacktest here
    backtest = Backtest(agent=agent)

    # generate episodes with the same episode_buffer and episode_length
    backtest.run_episode_generator(identifier_list=identifier_list,
                                   date_start="2021-01-04",  # start date after which episodes are generated
                                   date_end="2021-01-05",  # end date before which episodes are generated
                                   episode_interval=20,  # start intervals: e.g. 30 -> one starting point every 30 mins
                                   episode_shuffle=True,
                                   episode_buffer=5, #
                                   episode_length=10, # min
                                   num_episodes=5,
                                   )







