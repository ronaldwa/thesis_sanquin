"""Script can be used to test the trained model"""


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2
import numpy as np

# Own python files
import sanquin_blood as sq_blood
import environment
import visualisations

# ANTIGENS INCLUDED
#                         A   B   D   C   c   E   e   K   k   M   N   S   s
antigens_incl = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          # C(w) Fy(a) Fy(b) Jk(a) Jk(b) Kp(a) Kp(b) Lu(a) Lu(b) Wr(a)
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# ----- DATA
# get demand
demand_distribution = sq_blood.get_bloodgroup_distribution(antigens_incl, 'demand')
# get supply
supply_distribution = sq_blood.get_bloodgroup_distribution(antigens_incl, 'supply')

# Initialize variables
MAX_AGE = 35
INVENTORY_SIZE = 50
MAX_DAY = 500

# Get model ready
env = environment.Env(supply_distribution[0], demand_distribution[0], MAX_AGE, INVENTORY_SIZE, max_day=MAX_DAY, test=True, obs_method=2)
env = DummyVecEnv([lambda: env])
model = PPO2.load('3c_bloodgroup_5m_method2', env=env)


# Initialize testing variables
episodes = 1
for i in range(episodes):
    obs = env.reset()
    # Do some testing
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs_next, rewards, done, info = env.step(action)
        obs = obs_next
print('-----')
eval = env.env_method('render_blood_specific')  # get evaluation metrics
visualisations.export_results(eval[0], list(demand_distribution[0].keys()), 'test')  # print visuals


# todo: test piece below
# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
