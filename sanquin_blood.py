"""
This file handles the blood groups

Global variable:
- data_folder
  Refers to where the data is stored

Functions:
- calculate_prevalence_blood_group
- recalculate_system_2
- calculate_distribution_ethnicity
- get_bloodgroup_distribution

JSON files used:
- antigen_prevalence.json
    Known prevalence of (all) antigens in population
- antigen_names.json
    List of the names of the antigens in the correct order
- antigen_groups.json
    List of the groups in the same order as the antigen names
- prevalence_demand.json
    Given prevalence of the demand of ABD antigens by Sanquin in 2012-2019
- ABD_supply.json
    Given prevalence of the supply of ABD antigens by Sanquin in 2012-2019



When referring tot blood systems, this system is used:
blood_system = {1: ['A', 'B'],
                2: ['D', 'C', 'c', 'E', 'e'],
                3: ['K', 'k'],
                4: ['M', 'N', 'S', 's'],
                5: ['C(w)'],
                6: ['Fy(a)', 'Fy(b)'],
                7: ['Jk(a)', 'Jk(b)'],
                8: ['Kp(a)', 'Kp(b)'],
                9: ['Lu(a)', 'Lu(b)'],
                10: ['Wr(a)']}
"""

import json
import logging
import numpy as np

# Setup logging
logger_blood = logging.getLogger(__name__)
logging.basicConfig(filename='inventory.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %('
                                                                                 'message)s')

# Data
data_folder = 'data/'


def calculate_prevalence_bloodgroup(distribution, remove_no_prevalence=True):
    """
    Calculates the prevalence of blood groups by combining the different systems
    :param distribution:(dict) distribution of the blood systems. Format: {'system_idx':{'bloodgroup':prevalence},{}}
    :param remove_no_prevalence:(boolean) default=True, removes the blood groups which have a prevalence equals 0
    :return: blood_dict: (dict) blood groups with prevalence. Format: {'bloodgroup': prevalence}
    """
    # initialize
    blood_dict = {}

    # Add the prevalence to the blood_dict per antigen system
    for system in distribution:
        # If nothing in distribution yet, start with first system in blood_dict
        if blood_dict == {}:
            blood_dict = distribution[system]

        # Add the system to the blood groups already in blood_dict
        else:
            previous_blood_dict = blood_dict
            blood_dict = dict()

            # Loop over the blood groups and add antigens to the blood groups,
            # thus increasing complexity in blood groups
            for key1 in previous_blood_dict.keys():
                for key2 in distribution[system]:
                    blood_dict[key1 + key2] = previous_blood_dict[key1] * distribution[system][key2]

    # remove prevalence that is zero
    if remove_no_prevalence is True:
        old_blood_dict = blood_dict.copy()
        for key in old_blood_dict:
            if blood_dict[key] == 0:
                del blood_dict[key]

    return blood_dict


def recalculate_system_2(distribution_system_2, distribution_d):
    """
    When using current existing data about blood groups, ABD is used. This function changes the antigens in system 2
    according to the prevalence of the rhesus in the current blood group systems.
    :param distribution_system_2: (dict) distribution of the blood system 2.
    Format: {'system_idx':{'bloodgroup':prevalence},{}}
    :param distribution_d: distribution of antigen D. Format: {'0':prevalence_0, '1':prevalence_1}
    :return: new distribution: (dict) new distribution of system 2. Same format as distribution_system_2
    """
    # initialize
    new_distribution = {}

    # Get the sum of blood groups with antigen D (idx==0 in string)
    sum_0 = sum([distribution_system_2[key] for key in distribution_system_2 if key[0] == '0'])
    sum_1 = sum([distribution_system_2[key] for key in distribution_system_2 if key[0] == '1'])

    # Recalculate the prevalence of the blood groups according to the provided antigen D values
    for blood_group in distribution_system_2:
        if blood_group[0] == '0':
            new_distribution[blood_group] = (1 / sum_0) * distribution_d['0'] * distribution_system_2[blood_group]
        if blood_group[0] == '1':
            new_distribution[blood_group] = (1 / sum_1) * distribution_d['1'] * distribution_system_2[blood_group]
    return new_distribution


def calculate_distribution_ethnicity(distribution, c=1.0, n=0, a=0):
    """
    Given the distribution of ethnicity in population (variables c, n, and a) returns the distribution
    :param distribution: (dict) distribution of the blood systems.
    Format: {'system_idx':{'bloodgroup': {caucasian : prevalence_caucasian, negroid:prevalence_negroid, ..},{}}
    :param c: (float) provided fraction of caucasian in population
    :param n: (float) provided fraction of negroid in population
    :param a: (float) provided fraction of asian in population
    :return: distribution_dict: (dict) istribution of the blood systems.
    Format: {'system_idx':{'bloodgroup':prevalence},{}}
    """
    assert c + n + a == 1, "The sum of the faction needs to be 1"

    # initialize
    distribution_dict = {}

    # Per blood group system calculate the distribution
    for system_idx in distribution.keys():
        blood_dict = dict()
        for antigen_combination in distribution[system_idx].keys():
            prevalence_key = (c * distribution[system_idx][antigen_combination]['caucasian']
                              + n * distribution[system_idx][antigen_combination]['negroid']
                              + a * distribution[system_idx][antigen_combination]['asian']) / 100

            blood_dict[antigen_combination] = prevalence_key
        distribution_dict[system_idx] = blood_dict

    logger_blood.info(f"Blood distribution created, ethnicity: c:{c}, n:{n}, a:{a}")
    return distribution_dict


def filter_distribution_bloodsystem(system_distribution, filter_array):
    """ System distribution: {bloodgroup : distribution bloodgroup}"""
    assert len(filter_array) == len(
        list(system_distribution.keys())[0]), 'Filter array needs as much values as antigens in bloodgroup'

    # If nothing to filter
    if sum(filter_array) == len(filter_array):
        return system_distribution
    if sum(filter_array) == 0:
        return {}

    filtered_distribution = dict()

    while 0 in filter_array:
        # initialize
        filtered_distribution = dict()

        # Find the index of the antigen to remove in the blood binary
        idx = filter_array.index(0)

        # loop through all blood types
        for key in system_distribution.keys():

            # remove the antigen from the binary
            new_key = key[:idx] + key[idx + 1:]

            # Add the new_key to the new dict
            if new_key not in filtered_distribution:
                filtered_distribution[new_key] = system_distribution[key]

            # If the key already in the new_prevalence dict, add the prevalence
            else:
                filtered_distribution[new_key] += system_distribution[key]

        # remove the 0 from the filter array
        del filter_array[idx]
        system_distribution = filtered_distribution.copy()

    return filtered_distribution  # format: {bloodgroup: distribution bloodgroup} with sum = 1


def get_bloodgroup_distribution(antigens_incl, data_choice=None, c=1.0, n=0, a=0,):
    """
    Main function to return the distribution.

    :param antigens_incl: (NP.Array) An array containing 0 when antigen is excluded, 1 when antigen is included.
    Format:[A, B, D , C, c, E, e, K, k, M, N, S, s, C(w), Fy(a), Fy(b), Jk(a), Jk(b), Kp(a), Kp(b), Lu(a), Lu(b), Wr(a)]
    :param data_choice: (Str), default = None. If value is supply or demand, the distribution of blood groups in known
    supply or demand by Sanquin will be used. Period 2012-2019
    :param c: (float) provided fraction of caucasian in population
    :param n: (float) provided fraction of negroid in population
    :param a: (float) provided fraction of asian in population
    :return: bloodgroup_prevalence: (dict) distribution of blood groups. Format:  {'bloodgroup':prevalence},{}}
    :return: included_antigen_names: (list) list with the names of antigens in correct order as antigens in the str.
    """
    logger_blood.info(f"Request received for new bloodgroup distribution")

    # Import necessary files
    with open(data_folder + 'antigen_prevalence.json') as json_file:
        antigen_prevalence = json.load(json_file)
    with open(data_folder + 'antigen_names.json') as json_file:
        antigen_names = json.load(json_file)
    with open(data_folder + 'antigen_groups.json') as json_file:
        antigen_groups = json.load(json_file)

    # Get the right distribution according to ethnicity
    distribution = calculate_distribution_ethnicity(antigen_prevalence, c, n, a)

    # What systems to use, others can be skipped
    include_systems = set()
    for index, antigen in enumerate(antigens_incl):
        if antigen == 1:
            include_systems.add(antigen_groups[index])

    # Filter the systems
    # A. Adapt the distribution when using these data sets if necessary
    if data_choice == 'supply':
        with open(data_folder + 'ABD_supply.json') as json_file:
            antigen_prevalence_supply = json.load(json_file)
        distribution['1'] = filter_distribution_bloodsystem(antigen_prevalence_supply, [1, 1, 0])
        distribution['2'] = recalculate_system_2(distribution['2'],
                                                 filter_distribution_bloodsystem(antigen_prevalence_supply, [0, 0, 1]))
        logger_blood.info(f"Other dataset used: supply")

    if data_choice == 'demand':
        with open(data_folder + 'prevalence_demand.json') as json_file:
            antigen_prevalence_demand = json.load(json_file)
        distribution['1'] = filter_distribution_bloodsystem(antigen_prevalence_demand, [1, 1, 0])
        distribution['2'] = recalculate_system_2(distribution['2'],
                                                 filter_distribution_bloodsystem(antigen_prevalence_demand, [0, 0, 1]))
        logger_blood.info(f"Other dataset used: demand")

    # B. Filter the data on the provided antigens
    distribution_use = {}
    for system in include_systems:
        system_incl = []
        for idx in [i for i, x in enumerate(antigen_groups) if x == system]:
            system_incl.append(antigens_incl[idx])
        distribution_use[system] = filter_distribution_bloodsystem(distribution[str(int(system))], system_incl)

    # create a list with the names
    included_antigen_names = [antigen_names[index] for index, value in enumerate(antigens_incl) if value == 1]
    logger_blood.info(f"Distribution returned with antigens: {included_antigen_names}")

    # Combine in blood groups
    bloodgroup_prevalence = calculate_prevalence_bloodgroup(distribution_use)
    return bloodgroup_prevalence, included_antigen_names




