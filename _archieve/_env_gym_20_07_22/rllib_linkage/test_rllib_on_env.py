
#TODO: Ich habe wieder das Problem, dass MarketState nicht funktioniert (loggt 0en),
# irgendwo muss etwas mit den imports schief gegangen sein wie letztes mal...

import ray
from gym_linkage.tradingenv_v8 import TradingEnvironment
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
import pprint


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

# TODO: Tune registration

config = {
    "env": TradingEnvironment,
    #"env_config": {
    #    "config": {
    #        "agent": agent,
    #        "config_dict": None,
    #    },
    #}
}


# instantiate ppo trainer
rllib_trainer = PPOTrainer(config=config)
rllib_trainer

# run a training loop
results = rllib_trainer.train()
# del config info
del results["config"]
# show results
pprint.pprint(results)

#ray.shutdown()

"""
# We use the `Trainer.save()` method to create a checkpoint.
checkpoint_file = rllib_trainer.save()
print(f"Trainer (at iteration {rllib_trainer.iteration} was saved in '{checkpoint_file}'!")

new_trainer = PPOTrainer(config=config)
new_trainer.restore(checkpoint_file)

rllib_trainer.stop()
new_trainer.stop()
"""