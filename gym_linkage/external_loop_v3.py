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
# has to be changen when spaces become multi dimensional
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
ddqn = DDQNModel(state_dim=state_dim, num_actions=num_actions)
ddqn.online_network.summary()

num_episodes = env.simulator.replay_data.num_episodes
# statistics:
#action_list_all_episodes = []
#pnl_list_all_episodes = [] # reward

print("NUMBER OF EPISODES", num_episodes)
for episode_counter in range(1):#(num_episodes):
    # call reset before run (reset env and build new episode)
    env.reset()
    # take first step to get first obs
    # TODO: move into env.reset()
    #env.simulator.take_step()
    # return the first observation:
    #first_obs = env.simulator.market_obs.copy()
    #print(first_obs)

    print(episode_counter)
    # the episode lengths can vary
    current_episode_length = env.simulator.replay_data.current_episode_length
    # initialize for first iteration (todo: proper solution)

    # statistics:
    action_list = []
    #pnl_list = 0 # reward

    # external done flag for testing
    done = False

    for step in range(1000):

        #if step == (current_episode_length-1):
        #    done = True
        #    # store final reward
        #    result = env.simulator.agent.market_interface.pnl_realized_total
        #    pnl_list_all_episodes.append(result)
            # store number of trades

        # provisorische Lösung
        #if step==0:
        #    new_obs = np.zeros(40)
        #else:
        new_obs = env.simulator.market_obs.copy()
        print(new_obs)

        # TODO: How to execute the action in rl_agent?
        action = ddqn.epsilon_greedy_policy(new_obs.reshape(-1, state_dim))
        action_list.append(action)

        env.step(action=action)

        # dense reward
        #reward = env.simulator.agent.market_interface.pnl_realized_total
        #print('reward', reward)

        #if done:
        #    save_done = 1
        #else:
        #    save_done = 0

        # store to ddqn memory
        #ddqn.memorize_transition(last_obs, action, reward, new_obs, save_done)

        # train model
        #if ddqn.train:
        #    ddqn.experience_replay()
       # if done:
        #    break

        # store obs as last_obs
        #last_obs = new_obs

    #action_list_all_episodes.append(action_list)

os.system('say "Training Loop over all Episodes is finished"')














