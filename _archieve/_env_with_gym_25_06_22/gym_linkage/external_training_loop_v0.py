from gym_linkage.tradingenv_v4 import TradingEnvironment # wichtig: gleicher import wie bei base agent!
from rlagent.rlagent import RLAgent

# initialize agent
agent = RLAgent(
    name="RLAgent",
    quantity=100)

# initialize trading environment (with agent)
env = TradingEnvironment(agent=agent)
# generate episode start list
env.generate_episode_start_list()

# EXTERNAL TRAINING LOOP

num_episodes = env.num_episodes # evtl nicht die tatsächlichen episoden sondern nur die "gewünschten" wenn eine episode nicht gebaut werden kann...


for episode in range(num_episodes):
    # call reset before run (reset env and build new episode)
    env.reset_before_run()
    print(episode)
    # the episode lengths can vary
    current_episode_length = env.current_episode_length

    #for step in range(current_episode_length:
        # take step
        #env.step()

"""
env.reset_before_run()
#env.run_episode_steps_with_next()
# env.run_episode_steps()
# env.run_all_episodes()
# TODO: Das __str__ Problem liegt bei step, mit run_all_episodes geht es...
#
for i in range(1000):
   env.step()
   
"""

