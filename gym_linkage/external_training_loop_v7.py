# Use while not done condition to iterate over steps in episode

import logging

logging.basicConfig(level=logging.CRITICAL)
from gym_linkage.tradingenv_v10 import Replay  # wichtig: gleicher import wie bei base agent!
from rlagent.rlagent_v2 import RLAgent
from gym_linkage.tradingenv_v10 import TradingEnvironment, AgentInterface
from model.ddqn import DDQNModel
import os
import numpy as np
import pandas as pd

sub_agent = RLAgent(
    name="RLAgent",
    quantity=100)

agent = AgentInterface(sub_agent)

replay = Replay()

env_config = {"config": {"agent": agent, "replay": replay}}

env = TradingEnvironment(env_config=env_config)

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

ddqn = DDQNModel(state_dim=state_dim, num_actions=num_actions)
ddqn.online_network.summary()

action_list_all_episodes = []
reward_list_all_episodes = []

# num_episodes = env.replay.num_episodes
num_episodes = 3
print('NUM EPISODES', num_episodes)

for episode_counter in range(num_episodes):
    # call env.reset() -> return first observation
    last_obs = env.reset()

    # statistics:
    action_list = []
    reward_list = []

    while not env.replay.done:

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

# stats
print('action means')
for i, action_list in enumerate(action_list_all_episodes):
    print(np.mean(action_list))

print('reward means')
for i, reward_list in enumerate(reward_list_all_episodes):
    print(np.mean(reward_list))


