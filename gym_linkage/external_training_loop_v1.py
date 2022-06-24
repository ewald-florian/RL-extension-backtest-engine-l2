
# Now, TradingEnvironment uses trading.simulator
from gym_linkage.tradingenv_v6 import TradingEnvironment  # wichtig: gleicher import wie bei base agent!
from rlagent.rlagent_v2 import RLAgent
from model.ddqn import DDQNModel
import numpy as np


# TODO: Action einbinden (Schritt 1: Extern, Schritt2: Intern)
# TODO: DDQN einbinden
agent = RLAgent(
    name="RLAgent",
    quantity=100)

env = TradingEnvironment(agent=agent)
# has to be changen when spaces become multi dimensional
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
ddqn = DDQNModel(state_dim=state_dim, num_actions=num_actions)
ddqn.online_network.summary() # should it be initiated inside env? actually no, env and model should be seperate

env.simulator.replay_data.generate_episode_start_list()

num_episodes = env.simulator.replay_data.num_episodes

for episode_counter in range(num_episodes):#(num_episodes):
    # call reset before run (reset env and build new episode)
    env.reset()
    print(episode_counter)
    # the episode lengths can vary
    current_episode_length = env.simulator.replay_data.current_episode_length
    env.liste = []
    last_obs = np.zeros(40)

    # external done flag for testing
    done = False

    for step in range(current_episode_length):

        if step == (current_episode_length-1):
            done = True

        env.step()

        new_obs = env.simulator.market_obs.copy()

        action = ddqn.epsilon_greedy_policy(new_obs)
        print(action)

        # for testing
        reward = 0

        if done:
            save_done = 1
        else:
            save_done = 0

        # store to ddqn memory
        ddqn.memorize_transition(last_obs, action, reward, new_obs, save_done)

        # store obs as last_obs
        # last_obs = new_obs












