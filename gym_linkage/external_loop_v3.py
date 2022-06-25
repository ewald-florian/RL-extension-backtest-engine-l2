# Idea: execute action inside the loop...

# Now, TradingEnvironment uses trading.simulator
from gym_linkage.tradingenv_v6 import TradingEnvironment  # wichtig: gleicher import wie bei base agent!
from rlagent.rlagent_v2 import RLAgent
from model.ddqn import DDQNModel
import numpy as np
import os


agent = RLAgent(
    name="RLAgent",
    quantity=100)

env = TradingEnvironment(agent=agent)
# to be changed if spaces become multi dimensional
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
ddqn = DDQNModel(state_dim=state_dim, num_actions=num_actions)
ddqn.online_network.summary()

num_episodes = env.simulator.replay_data.num_episodes

# statistics:
action_list_all_episodes = []
reward_list_all_episodes = []

print("(LOOP) NUMBER OF EPISODES", num_episodes)
for episode_counter in range(num_episodes):
    # call env.reset() -> return first observation
    last_obs = env.reset()
    # the episode lengths can vary
    current_episode_length = env.simulator.replay_data.current_episode_length

    # statistics:
    action_list = []
    reward_list = []

    for step in range(current_episode_length-1):

        # compute action according to last_obs
        action = ddqn.epsilon_greedy_policy(last_obs.reshape(-1, state_dim))

        # env.step(action)  according to action
        new_obs, reward, done, info = env.step(action=action)

        # store variables to ddqn memory
        ddqn.memorize_transition(last_obs, action, reward, new_obs, int(done))

        # train model
        if ddqn.train:
            ddqn.experience_replay()
        if done:
            break

        # store new_obs as last_obs for the next iteration
        last_obs = new_obs

        # step results

        reward_list.append(reward)
        action_list.append(int(action))

    # episode results
    action_list_all_episodes.append(action_list)
    reward_list_all_episodes.append(reward_list)

os.system('say "Training Loop over all Episodes is completed"')














