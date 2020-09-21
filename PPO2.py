"""
TRAIN THE ALGORITHM
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import numpy as np

# Own python files
import sanquin_blood as sq_blood
import environment
import data_extract

from datetime import datetime
time_string = datetime.now().strftime("%Y_%m_%d_%H_%M")


# ANTIGENS that will be selected during training
#                         A   B   D   C   c   E   e   K   k   M   N   S   s
antigens_incl = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          # C(w) Fy(a) Fy(b) Jk(a) Jk(b) Kp(a) Kp(b) Lu(a) Lu(b) Wr(a)
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# ----- DATA
# get demand
demand_distribution = sq_blood.get_bloodgroup_distribution(antigens_incl, 'demand')
# get supply
supply_distribution = sq_blood.get_bloodgroup_distribution(antigens_incl, 'supply')


# Initialize parameters
MAX_AGE = 35
INVENTORY_SIZE = 50
observation_method = 2
name = '_3c_bloodgroup_5m_method2_withreset' # name of the file
file_name = time_string+name

# Create environment
env = environment.Env(supply_distribution[0], demand_distribution[0], MAX_AGE, INVENTORY_SIZE, observation_method, file_name=file_name)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: env])

# Train the model
model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log="./tensorboard_data/ppo_v6/")  # create model
model.learn(total_timesteps=5000000, tb_log_name=file_name)  # train the model and run tensorboard
# TB- run: tensorboard --logdir ./ppo_v6/

# Export
model.save(file_name)  # save the model

# Combine results and save
# might take a while!
data_extract.merge_results(file_name)
