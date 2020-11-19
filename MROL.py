"""
This function calculates the results for the FIFO/MROL policy
This is a daily optimization and is solved using pulp
"""

from sanquin_inventory import find_compatible_blood_list
import sanquin_inventory as sq_inv
from pulp import *
from typing import Tuple


def create_uf_dict(distribution: Tuple[dict, list], blood_groups_str: list) -> dict:
    """
    Create usability factor dict per blood group
    :param distribution: dict of {blood group : prevalence }, list of antigens included
    :param blood_groups_str: list of blood group strings
    :return: dictionary consisting of {blood group : usability factor}
    """
    uf_dict = {}

    # add the usability factor per blood group
    for idx, blood_group_str in enumerate(blood_groups_str):
        uf = sum([distribution[0][blood_str] for blood_str in
                  find_compatible_blood_list(blood_groups_str, idx, factor_type='usability')[0]])
        uf_dict[blood_group_str] = uf

    return uf_dict


def get_rol(blood_groups_str: list, requested_blood_group: str, uf_sup_dict: dict, uf_dem_dict: dict) -> dict:
    """
    Get a dict of the relative opportunity loss values per blood group
    :param blood_groups_str: list of blood group strings
    :param requested_blood_group: str: of the binary of the blood group
    :param uf_sup_dict: dict of the usability factor per blood group of supplied blood groups
    :param uf_sup_dict: dict of the usability factor per blood group of supplied blood groups
    :return: rol_dict: dict consisting of the rol values per blood group
    """
    blood_group_idx = blood_groups_str.index(requested_blood_group)

    rol_dict = {}

    # Find compatible blood
    comp_blood_str, comp_blood_int = find_compatible_blood_list(blood_groups_str, blood_group_idx)
    # Loop through compatible blood
    for idx, blood_group_idx in enumerate(comp_blood_int):
        # calculate rol
        uf_sup = uf_sup_dict[comp_blood_str[idx]]
        uf_req = uf_dem_dict[requested_blood_group]
        rol = (uf_sup - uf_req) / uf_sup
        rol_dict[comp_blood_str[idx]] = rol

    return rol_dict


def calculate_cost(blood_groups_str: list, uf_sup_dict: dict, uf_dem_dict: dict, i: str, r: int, j: str,
                   max_age: int = 35, gamma: float = 0.5) -> float:
    """
    Calculate the cost according to the FIFO/MROL function
    :param blood_groups_str: list of string of blood groups
    :param uf_sup_dict: dict of the usability factor per blood group of supplied blood groups
    :param uf_sup_dict: dict of the usability factor per blood group of supplied blood groups
    :param i: str: binary blood group item in inventory
    :param r: int: age of item in inventory
    :param j: str: binary blood group in demand
    :param max_age: int, max age of RBC, after which it is perished
    :param gamma: flaot hyperparameter between FIFO and MROL
    :return: cost as float
    """
    delta_ij = get_rol(blood_groups_str, j, uf_sup_dict, uf_dem_dict)[i]
    return gamma * delta_ij + (1 - gamma) * (r / max_age)


def lp_solve(blood_groups_str: list, inv: object, queue: object, uf_sup_dict: dict, uf_dem_dict: dict,
             max_age: int = 35, gamma: float= 0.5) -> Tuple[list, list, dict]:
    """
    Structure of LP solver
    1. Create nodes
    2. Create edges
    3. Initialize LP
    4. Solve the problem

    :param blood_groups_str: list of string with blood groups
    :param inv: the inventory object
    :param queue: the queue object
    :param uf_sup_dict: dict containing the usability factors of the supplied blood groups
    :param uf_dem_dict: dict containing the usability factors of the demanded blood groups
    :param max_age: int: max age a RBC can become
    :param gamma: hyperparameter to balance FIFO and MROL
    :return:
    inv_removal_list : list, list with items removed from inventory (outdating)
    queue_removal_list: list, list with items removed from queue (shortage)
    blood_matches: dict: { (blood group requested, blood group supplied) : amount}
    """
    # 1. Create nodes
    ## Supply
    supply_nodes_info = {}
    for idx, i in enumerate(blood_groups_str):
        for idx, amount in enumerate(inv.inventory[idx]):
            if amount > 0:
                r = max_age - idx
                supply_nodes_info[(i, r)] = amount  # {(i,r) : s_ir}

    ## Demand
    demand_nodes_info = {}
    for idx, j in enumerate(blood_groups_str):
        demand_nodes_info[j] = queue.queue[idx]  # {j:d_j}

    ## P & Q
    nodes_pq = ['p', 'q']

    ## All nodes togther
    all_nodes = [list(supply_nodes_info.keys()), list(demand_nodes_info.keys()), nodes_pq]
    all_nodes = [item for sublist in all_nodes for item in sublist]

    # 2. Create edges
    ## Tau: v1 -> v2
    edges_tau = {}
    max_cost_tau = -999  # placeholder
    for node_v1 in supply_nodes_info:
        i = node_v1[0]
        i_index = blood_groups_str.index(node_v1[0])
        r = node_v1[1]
        compatible_blood = find_compatible_blood_list(blood_groups_str, i_index, factor_type='usability')[0]
        for node_v2 in demand_nodes_info:
            if node_v2 in compatible_blood:
                cost = calculate_cost(blood_groups_str, uf_sup_dict, uf_dem_dict, i, r, j=node_v2, max_age=max_age, gamma=gamma)
                if cost > max_cost_tau:
                    max_cost_tau = cost
                capacity = min(supply_nodes_info[node_v1], demand_nodes_info[node_v2])
                edges_tau[(node_v1, node_v2)] = (cost, capacity)

    ## Tau_p: p -> v1
    edges_tau_p = {}
    for node in supply_nodes_info:
        capacity = supply_nodes_info[node]
        cost = 0
        edges_tau_p[('p', node)] = (cost, capacity)

    ## Tau_q: v2 -> q
    edges_tau_q = {}
    for node in demand_nodes_info:
        capacity = demand_nodes_info[node]
        cost = 0
        edges_tau_q[(node, 'q')] = (cost, capacity)

    ## Edqe[q,p]: q -> p
    capacity = sum(demand_nodes_info.values())
    cost = -max_cost_tau - 1
    edges_pq = {('q', 'p'): (cost, capacity)}  # to do

    ## Tau_pq: combine all edges
    all_edges = {**edges_tau, **edges_tau_p, **edges_tau_q, **edges_pq}

    # 3. Initialize LP
    ## Create variables - edges
    variables = LpVariable.dicts('m', all_edges, 0, cat=LpInteger)
    ## Creates the upper and lower bounds on the variables
    for edge in all_edges:
        variables[edge].bounds(0, all_edges[edge][1])  # Lowerbound = 0, upperBound is the capacity

    ## Create the 'prob' variable to contain the problem data
    problem = LpProblem("MROL", LpMinimize)

    ## Create the objective function
    problem += lpSum([variables[edge] * all_edges[edge][0] for edge in all_edges])

    ## Create all problem constraints

    ### 1 Maximum flow (circulation problem)
    problem += lpSum(variables[x] for x in edges_tau) == lpSum(variables[x] for x in edges_pq)

    ### 2 Capacity constraint
    for tau in all_edges:
        problem += variables[tau] <= all_edges[tau][1]

    ### 3 Demand constraint
    for j in demand_nodes_info:
        problem += lpSum(variables[x] for x in edges_tau if x[1] == j) <= demand_nodes_info[j]

    ### 4 Supply constraint
    for i in supply_nodes_info:
        problem += lpSum(variables[x] for x in edges_tau if x[0] == i) <= supply_nodes_info[i]

    # 4. Solve the problem
    problem.solve()

    ## Problem must be optimal
    assert LpStatus[problem.status] == 'Optimal'

    ## Return values so that the inventory model can work with it
    inv_removal_list = []
    queue_removal_list = [0 for blood_group in blood_groups_str]
    blood_matches = {}
    for x in variables:
        if (variables[x].varValue > 0) & (x != ('q', 'p')):
            sup = x[0][0]
            req = x[1]
            amount = variables[x].varValue
            age = x[0][1]

            # For removing from inventory
            remove_item = (blood_groups_str.index(sup), age, amount)
            inv_removal_list.append(remove_item)

            # For removing from queue
            queue_removal_list[blood_groups_str.index(req)] += amount

            # For keeping track of matches
            if (req, sup) in blood_matches:
                blood_matches[(req, sup)] += amount
            else:
                blood_matches[(req, sup)] = amount

    return inv_removal_list, queue_removal_list, blood_matches


def solve(demand_distribution: Tuple[dict, list], supply_distribution: Tuple[dict, list], demand: int = 50,
          doi: int = 6, max_age: int = 35, n_warm_start_days: int = 50, n_days: int = 50, gamma: float = 0.5) -> dict:
    """
    Solve the issuing problem
    :param demand_distribution: dict of {blood group : prevalence }, list of antigens included of the demand
    :param supply_distribution: dict of {blood group : prevalence }, list of antigens included of the supply
    :param demand: int, daily demand /supply per day
    :param doi: int days of inventory, how many days are filled before the day begins
    :param max_age: int max age an RBC can reach before perish
    :param n_warm_start_days: int, number of days to warm start
    :param n_days: int, number of days to do apply the issuing strategy to
    :param gamma: float, hyperparameter to balance FIFO and MROL
    :return: eval metrics, dict with all data
    """

    # Blood group strings
    blood_groups_str = list(supply_distribution[0].keys())

    # Usability factors
    uf_sup_dict = create_uf_dict(supply_distribution, blood_groups_str)
    uf_dem_dict = create_uf_dict(demand_distribution, blood_groups_str)

    # Inititalize counters
    infeasible = 0

    # Initialize invenotry and queue
    queue = sq_inv.Request(demand_distribution[0])
    inv = sq_inv.Inv(supply_distribution[0], max_age, eval_boolean=False)

    # Start the invenotry
    inv.reset_inventory()
    inv.start_inventory(doi, demand)

    # Warm start
    inv.eval_boolean = False  # do not record information
    for i in range(1, n_warm_start_days + 1):
        sys.stdout.write(f"Warm start day: {i} ({(round(i / n_days, 1)) * 100}%)  \r")
        sys.stdout.flush()

        queue.new_request(demand)
        inv.new_blood_supply(demand)  # the RL model starts the day with an new supply.

        inv_removal_list, queue_removal_list, blood_matches = lp_solve(blood_groups_str, inv, queue, uf_sup_dict,
                                                                       uf_dem_dict, max_age, gamma)

        # Deduct queue
        new_queue = []
        for Z_queue, Z_inv_removal_list in zip(queue.queue, queue_removal_list):
            new_queue.append(Z_queue - Z_inv_removal_list)
        queue.queue = new_queue

        # Deduct inventory
        for item in inv_removal_list:
            inv.inventory[item[0]][-item[1]] -= item[2]

        inv.increase_age()

    inv.eval_boolean = True  # record information
    for i in range(1, n_days + 1):
        sys.stdout.write(f"Day: {i} ({(round(i / n_days, 1)) * 100}%)  \r")
        sys.stdout.flush()

        queue.new_request(demand)
        inv.inventory_eval_queue(queue.queue)
        inv.new_blood_supply(demand)  # the RL model starts the day with an new supply.

        inv_removal_list, queue_removal_list, blood_matches = lp_solve(blood_groups_str, inv, queue, uf_sup_dict,
                                                                       uf_dem_dict, max_age, gamma)

        # Deduct queue
        new_queue = []
        for Z_queue, Z_inv_removal_list in zip(queue.queue, queue_removal_list):
            new_queue.append(Z_queue - Z_inv_removal_list)
        queue.queue = new_queue

        # If queue not empty remove from queue and keep track!
        if sum(queue.queue) > 0:
            infeasible += sum(queue.queue)
            for blood_idx, blood_amount in enumerate(queue.queue):
                inv.eval['infeasible'][blood_groups_str[blood_idx]] += blood_amount
            # queue does not need to be emptied, a new request will simply overwrite

        # Deduct inventory
        for item in inv_removal_list:
            inv.inventory[item[0]][-item[1]] -= item[2]

        # Keep track of all blood that is provided
        for item in inv_removal_list:
            for count in range(int(item[2])):
                inv.eval['provided'][blood_groups_str[item[0]]] += 1

        # Keep track of the issued age of the blood
        for item in inv_removal_list:
            for count in range(int(item[2])):
                age = 35 - item[1]
                inv.inventory_eval_age_match(blood_groups_str[item[0]], age)

        # Keep track of all matches
        for match in blood_matches:
            for count in range(int(blood_matches[match])):
                inv.inventory_eval_match(match[0], match[1])

        # End of the day
        inv.increase_age()

    return inv.eval