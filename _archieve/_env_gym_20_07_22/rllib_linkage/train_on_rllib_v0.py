"""
$ conda create -n rllib python=3.9
$ conda activate rllib
$ pip install cmake "ray[rllib,serve]" recsim jupyterlab tensorflow torch
$ pip install grpcio # <- extra install only on apple M1 mac
$ # **Note:** In case you are getting a "requires TensorFlow version >= 2.8" error at some point in the notebook, try the following:
$ pip uninstall -y tensorflow
$ python -m pip install tensorflow-macos --no-cache-dir
"""


from gym_linkage.tradingenv_v9 import TradingEnvironment
from rlagent.rlagent_v2 import RLAgent
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
import pprint
import ray

# Start a new instance of Ray
ray.init()

custom_config_dict = {
        "episode_buffer": 2,
        "episode_length": 12,
        "num_episodes": 8
        }

agent = RLAgent(
    name="RLAgent",
    quantity=100)

# make config for env (required format)
config = {
    "env": TradingEnvironment,
    "env_config": {
        "config": {
            "agent": agent,
            "config_dict": custom_config_dict,
        },
    }
}

# instantiate ppo trainer
rllib_trainer = PPOTrainer(config=config)
print(rllib_trainer)

# TODO: reduce logging/info of the backtest engine

# train
for _ in range(10):
    results = rllib_trainer.train()
    print(f"Iteration={rllib_trainer.iteration}: R(\"return\")={results['episode_reward_mean']}")

del results["config"]
# show results
pprint.pprint(results)
