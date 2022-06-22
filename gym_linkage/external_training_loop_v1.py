
# Now, TradingEnvironment uses trading.simulator
from gym_linkage.tradingenv_v6 import TradingEnvironment  # wichtig: gleicher import wie bei base agent!
from rlagent.rlagent import RLAgent

agent = RLAgent(
    name="RLAgent",
    quantity=100)

env = TradingEnvironment(agent=agent)

env.simulator.replay_data.generate_episode_start_list()

num_episodes = env.simulator.replay_data.num_episodes


for episode_counter in range(num_episodes):
    # call reset before run (reset env and build new episode)
    #env.simulator.replay_data.reset_before_run()
    env.reset()
    print(episode_counter)
    # the episode lengths can vary
    current_episode_length = env.simulator.replay_data.current_episode_length

    for step in range(current_episode_length):
        env.step()




