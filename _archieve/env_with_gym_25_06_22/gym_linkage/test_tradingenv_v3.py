from gym_linkage.tradingenv_v3 import Backtest # wichtig: gleicher import wie bei base agent!
from rlagent.rlagent import RLAgent

identifier_list = ["Adidas.BOOK", "Adidas.TRADES"]

agent = RLAgent(
    name="RLAgent",
    quantity=100)

# Instantiate RLBacktest here
backtest = Backtest(agent=agent)
backtest.generate_episode_start_list()
backtest.reset_before_run()
#backtest.run_all_episodes()
for i in range(1000):
    backtest.step()

"""
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
                                   date_end="2021-01-08",  # end date before which episodes are generated
                                   episode_interval=10,  # start intervals: e.g. 30 -> one starting point every 30 mins
                                   episode_shuffle=True,
                                   episode_buffer=5, #
                                   episode_length=10, # min
                                   num_episodes=5,
                                   )
"""