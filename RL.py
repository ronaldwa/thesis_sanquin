"""
PPO2 contains three function

- train, to train the agent
- train_decaying, to train the agent with decaying reward threshold
- solve, to test the model
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from datetime import datetime
from typing import Tuple

# Own python files
import environment
import data_extract
from custom_callback import StopTrainingOnDecayingRewardThreshold


def train(supply_distribution: Tuple[dict, list], demand_distribution: Tuple[dict, list], model_name: str, demand: int,
          max_day: int, training_timesteps_list: str, tblog: str, max_age: int = 35, obs_method: int = 1,
          doi: int = 4) -> str:
    """
    Train the agent
    First train without evaluation
    Second train with in-training evaluation

    :param demand_distribution: dict of {blood group : prevalence }, list of antigens included of the demand
    :param supply_distribution: dict of {blood group : prevalence }, list of antigens included of the supply
    :param model_name: str: name of the model to be stored
    :param demand: int: number of blood that is supplied / requested
    :param max_day: int: number of days per episode
    :param training_timesteps_list: list: [number of episodes without evaluation, number of episodes with evaluation]
    :param tblog: str, name of the tensorboard log
    :param max_age: int, max age of the RBCs
    :param obs_method: int, 1 or 2: item requested one-hot-encoded (1) or binary (2)
    :param doi: int, number of days of inventory
    :return: file name: str, name of the model that is stored
    """
    # Initialize parameters
    GAMMA = round(1 - (1 / (35 * demand)), 5)  # 0.993
    state_type = 'custom_category'

    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    file_name = time_string + model_name

    max_reward = max_day * demand * 0.1

    # Create environment
    env = environment.Env(supply_distribution[0], demand_distribution[0], max_age, demand, doi=doi,
                          obs_method=obs_method, state_type=state_type, max_day=max_day, file_name=file_name,
                          verbose=0)
    env = DummyVecEnv([lambda: env])
    model = PPO2(MlpPolicy, env, gamma=GAMMA, verbose=0,
                 tensorboard_log="results/tensorboard_data/" + tblog + "/")  # create model

    # Train the model without evaluation (=faster)
    print('start phase 1, without evaluation')
    model.learn(total_timesteps=training_timesteps_list[0],
                tb_log_name=file_name)
    # TB- run: tensorboard --logdir ./tblog/

    # Export
    model.save('results/model/' + file_name)  # Save for backup

    callback_on_best = StopTrainingOnDecayingRewardThreshold(max_reward=max_reward,
                                                             episode_decay=training_timesteps_list[2],
                                                             reward_decay=0.05,
                                                             no_reward_episodes=training_timesteps_list[0],
                                                             verbose=1)

    # Callback for evaluation
    eval_callback = EvalCallback(env,  # callback_on_new_best=callback_on_best,
                                 best_model_save_path='results/model/' + file_name,
                                 eval_freq=50000, verbose=1,
                                 n_eval_episodes=5)

    # Train the model with eval every 50000 steps
    print('start phase 2 with evaluation')
    model.learn(total_timesteps=training_timesteps_list[1], tb_log_name=file_name, callback=eval_callback,
                reset_num_timesteps=False)  # train the model and run tensorboard 5000000 1500000

    # Export
    model.save('results/model/' + file_name + 'end')  # Save for backup

    # Extract the tensorboard data
    data_extract.extract_tb(file_name)

    return file_name

def train_decaying(supply_distribution: Tuple[dict, list], demand_distribution: Tuple[dict, list], model_name: str, demand: int,
          max_day: int, training_timesteps_list: str, tblog: str, max_age: int = 35, obs_method: int = 1,
          doi: int = 4) -> str:
    """
    Train the agent
    First train without evaluation
    Second train with in-training evaluation

    :param demand_distribution: dict of {blood group : prevalence }, list of antigens included of the demand
    :param supply_distribution: dict of {blood group : prevalence }, list of antigens included of the supply
    :param model_name: str: name of the model to be stored
    :param demand: int: number of blood that is supplied / requested
    :param max_day: int: number of days per episode
    :param training_timesteps_list: list: [number of episodes without evaluation, number of episodes with evaluation,
    interval of episodes after which the decay is used]
    :param tblog: str, name of the tensorboard log
    :param max_age: int, max age of the RBCs
    :param obs_method: int, 1 or 2: item requested one-hot-encoded (1) or binary (2)
    :param doi: int, number of days of inventory
    :return: file name: str, name of the model that is stored
    """
    # Initialize parameters
    GAMMA = round(1 - (1 / (35 * demand)), 5)  # 0.993
    state_type = 'custom_category'

    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    file_name = time_string + model_name

    max_reward = max_day * demand * 0.1

    # Create environment
    env = environment.Env(supply_distribution[0], demand_distribution[0], max_age, demand, doi=doi,
                          obs_method=obs_method, state_type=state_type, max_day=max_day, file_name=file_name,
                          verbose=0)
    env = DummyVecEnv([lambda: env])
    model = PPO2(MlpPolicy, env, gamma=GAMMA, verbose=0,
                 tensorboard_log="results/tensorboard_data/" + tblog + "/")  # create model

    # Train the model without evaluation (=faster)
    print('start phase 1, without evaluation')
    model.learn(total_timesteps=training_timesteps_list[0],
                tb_log_name=file_name)
    # TB- run: tensorboard --logdir ./tblog/

    # Export
    model.save('results/model/' + file_name)  # Save for backup

    callback_on_best = StopTrainingOnDecayingRewardThreshold(max_reward=max_reward,
                                                             episode_decay=training_timesteps_list[2],
                                                             reward_decay=0.05,
                                                             no_reward_episodes=training_timesteps_list[0],
                                                             verbose=1)

    # Callback for evaluation
    eval_callback = EvalCallback(env,  callback_on_new_best=callback_on_best,
                                 best_model_save_path='results/model/' + file_name,
                                 eval_freq=50000, verbose=1,
                                 n_eval_episodes=5)

    # Train the model with eval every 50000 steps
    print('start phase 2 with evaluation')
    model.learn(total_timesteps=training_timesteps_list[1], tb_log_name=file_name, callback=eval_callback,
                reset_num_timesteps=False)  # train the model and run tensorboard 5000000 1500000

    # Export
    model.save('results/model/' + file_name + 'end')  # Save for backup

    # Extract the tensorboard data
    data_extract.extract_tb(file_name)

    return file_name

def solve(supply_distribution: Tuple[dict, list], demand_distribution: Tuple[dict, list], model_name: str,
          export_model: str, max_age: int, demand: int, doi: int, n_warm_start_days: int, n_days: int,
          obs_method: int, state_type: str) -> dict:
    """

    :param demand_distribution: Tuple[dict, list] containing a dict with {blood_group : distribution}, list of
    included antigens
    :param supply_distribution: Tuple[dict, list] containing a dict with {blood_group : distribution}, list of
    included antigens
    :param model_name: str, name of the model that is used to store the results
    :param export_model: str, name of hte model that is trained
    :param max_age: int, max age of the RBCs
    :param demand: int, number of demand / supply per day
    :param doi: days of inventory, the number of days the inventory is filled before first supply
    :param n_warm_start_days: int, number of days of warm start
    :param n_days: int, number of days for evaluation
    :param obs_method: int, 1 or 2: item requested one-hot-encoded (1) or binary (2)
    :param state_type: type of state that is used 'custom category'
    :return:
    """
    # Get model ready
    env = environment.Env(supply_distribution[0], demand_distribution[0], max_age, demand, doi,
                          obs_method=obs_method, state_type=state_type, file_name=model_name)
    env = DummyVecEnv([lambda: env])
    model = PPO2.load(export_model, env=env)

    # Run model
    obs = env.reset()

    # Warm start
    print('warm start - started')
    env.env_method('set_days', n_warm_start_days)
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs_next, rewards, done, info = env.step(action)
        obs = obs_next
    print('warm start - ended')


    # Testing
    print('Testing - started')
    env.env_method('set_days', n_days)
    env.env_method('change_eval_boolean', True)

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs_next, rewards, done, info = env.step(action)
        obs = obs_next

    results = env.env_method('render_blood_specific')  # get evaluation metrics
    print('Testing - ended')

    return results
