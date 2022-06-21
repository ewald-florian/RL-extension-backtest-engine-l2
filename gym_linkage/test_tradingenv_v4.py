from gym_linkage.tradingenv_v4 import TradingEnvironment # wichtig: gleicher import wie bei base agent!
from rlagent.rlagent import RLAgent

identifier_list = ["Adidas.BOOK", "Adidas.TRADES"]

agent = RLAgent(
    name="RLAgent",
    quantity=100)

env = TradingEnvironment(agent=agent)
env.generate_episode_start_list()
env.reset_before_run()
#env.run_episode_steps_with_next()
# env.run_episode_steps()
# env.run_all_episodes()
# TODO: Das __str__ Problem liegt bei step, mit run_all_episodes geht es...
#
for i in range(1000):
   env.step()

