"""
This file provides 5 different functions to get started:
- train_rl
- test_rl
- test_fp1
- test_fp2
- test_mrol
The file begins by importing the right packages and setting the parameters.
Note that the Fixed Policy models only work when only A,B & D are selected as antigens.

At the end of the file a short example pipeline is provided
"""
# Imports general packages
import numpy as np

# Import own packages
import sanquin_blood as sq_blood
import visualisations

# Import the algorithm packages
import FIXED_POLICY
import FIXED_POLICY_2
import MROL
import RL

# Select the antigens
#                         A   B   D   C   c   E   e   K   k   M   N   S   s
antigens_incl = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          # C(w) Fy(a) Fy(b) Jk(a) Jk(b) Kp(a) Kp(b) Lu(a) Lu(b) Wr(a)
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# ----- DATA
# get demand
demand_distribution = sq_blood.get_bloodgroup_distribution(antigens_incl, 'demand')
# get supply
supply_distribution = sq_blood.get_bloodgroup_distribution(antigens_incl, 'supply')
# Blood group binary keys in a list
blood_groups_str = list(supply_distribution[0].keys())

parameters = {
    'include_visuals': False,
    'max_age': 35,
    'demand': 20,
    'doi': 4,
    'n_warm_start_days': 365 * 10,  # 10 years
    'n_days': 365 * 50,  # 50 years
    'rl': {'state_type': 'custom_category',  # State type, in research only custom_category is used
           'max_day': 100,  # number of days in an episode
           'obs_method': 1,  # 1 is one-hot-encoded, 2 is binary representation of item requested
           'tb_log': 'log_1',  # Tensorboard log
           'training_interval': [10000000, 5000000, 500000]  # [nochecks, checks, decay reward over]
           },
    'mrol': {'gamma': 0.5},  # Hyperparameter to balance MROL and FIFO
    'fp': {'policy': {
        '100': ['100', '000'],
        '101': ['101', '100', '001', '000'],
        '110': ['110', '100', '010', '000'],
        '111': ['111', '101', '011', '100', '010', '001', '000'],
        '010': ['010', '000'],
        '011': ['011', '010', '001', '000'],
        '000': ['000'],
        '001': ['001', '000']}
    }
}


def train_rl(model_name: str) -> str:
    """
    Trains the model using the parameters defined
    :param model_name: name of the model to save
    :return: str, name of the stored model
    """
    # model name
    model_name = model_name + "_RL"

    print('- start training RL model')
    trained_model_name = RL.train(supply_distribution=supply_distribution,  # global
                                  demand_distribution=demand_distribution,  # global
                                  model_name=model_name,  # in loop
                                  max_age=parameters['max_age'],
                                  demand=parameters['demand'],
                                  max_day=parameters['rl']['max_day'],
                                  obs_method=parameters['rl']['obs_method'],
                                  doi=parameters['doi'],
                                  training_timesteps_list=parameters['rl']['training_interval'],
                                  tblog=parameters['rl']['tb_log']
                                  )
    print('- Complete training RL model')
    return trained_model_name


def test_rl(model_name: str, trained_model_name: str) -> dict:
    """
    Tests the RL agent
    Note that the parameters of the trained and tested RL agent need to be the same for these parameters:
    - Antigens included
    - Max age
    - state type
    - obs method
    :param model_name: name of the model to be stored
    :param trained_model_name: name of the agent trained to be evaluated
    :return: dict containing all evaluation metrics
    """
    model_name = model_name + "_RL"

    print('- start testing RL model')
    results = RL.solve(supply_distribution=supply_distribution,
                       demand_distribution=demand_distribution,
                       model_name=model_name,
                       export_model='results/model/' + trained_model_name + '/best_model',
                       max_age=parameters['max_age'],
                       demand=parameters['demand'],
                       doi=parameters['doi'],
                       n_warm_start_days=parameters['n_warm_start_days'],
                       n_days=parameters['n_days'],
                       obs_method=parameters['rl']['obs_method'],
                       state_type=parameters['rl']['state_type']
                       )
    print('- complete testing RL model')
    return results[0]


def test_fp1() -> dict:
    """
    Running fixed policy 1 issuing policy based on parameters
    ONLY ABO-D!
    :return: dict containing all evaluation metrics
    """
    print('- start testing FP1 model')
    results = FIXED_POLICY.solve(demand_distribution=demand_distribution,  # global
                                 supply_distribution=supply_distribution,  # global
                                 policy_dict=parameters['fp']['policy'],
                                 demand=parameters['demand'],
                                 doi=parameters['doi'],
                                 max_age=parameters['max_age'],
                                 n_warm_start_days=parameters['n_warm_start_days'],
                                 n_days=parameters['n_days']
                                 )
    print('- complete testing FP1 model')
    return results


def test_fp2() -> dict:
    """
    Running fixed policy 2 issuing policy based on parameters
    ONLY ABO-D!
    :return: dict containing all evaluation metrics
    """
    print('- start testing FP2 model')
    results = FIXED_POLICY_2.solve(demand_distribution=demand_distribution,  # global
                                   supply_distribution=supply_distribution,  # global
                                   policy_dict=parameters['fp']['policy'],
                                   demand=parameters['demand'],
                                   doi=parameters['doi'],
                                   max_age=parameters['max_age'],
                                   n_warm_start_days=parameters['n_warm_start_days'],
                                   n_days=parameters['n_days']
                                   )
    print('- complete testing FP2 model')
    return results


def test_mrol() -> dict:
    """
    Running FIFO/MROL issuing policy based on parameters
    :return: dict containing all evaluation metrics
    """
    print('- start testing MROL model')
    results = MROL.solve(demand_distribution=demand_distribution,
                         supply_distribution=supply_distribution,
                         demand=parameters['demand'],
                         doi=parameters['doi'],
                         max_age=parameters['max_age'],
                         n_warm_start_days=parameters['n_warm_start_days'],
                         n_days=parameters['n_days'],
                         gamma=parameters['mrol']['gamma'])
    print('- complete testing MROL model')
    return results


# Example pipeline

# Train and test RL-agent
model_name = 'test_model'
trained_model_name = train_rl(model_name=model_name)
rl_results = test_rl(model_name=model_name, trained_model_name=trained_model_name)
visualisations.export_results(rl_results, blood_groups_str, model_name, include_visuals=parameters['include_visuals'])

# Test other models
fp1_results = test_fp1()
model_name_fp1 = model_name + '_fp1'
fp2_results = test_fp2()
model_name_fp2 = model_name + '_fp2'
mrol_results = test_mrol()
model_name_mrol = model_name + '_mrol'

# visualize
visualisations.export_results(rl_results, blood_groups_str, model_name_fp1,
                              include_visuals=parameters['include_visuals'])
visualisations.export_results(rl_results, blood_groups_str, model_name_fp2,
                              include_visuals=parameters['include_visuals'])
visualisations.export_results(rl_results, blood_groups_str, model_name_mrol,
                              include_visuals=parameters['include_visuals'])
