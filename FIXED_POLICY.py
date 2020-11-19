"""
This function calculates the results for the fixed policy 1
The fixed policy can be prodivded as parameter, however is also provided in the dictionary below.
"""
import sys
import sanquin_inventory as sq_inv

default_policy = {
    '100' : ['100','000'],
    '101' : ['101', '100', '001', '000'],
    '110' : ['110', '100', '010', '000'],
    '111' : ['111', '101', '011', '100', '010', '001', '000'],
    '010' : ['010', '000'],
    '011' : ['011', '010', '001', '000'],
    '000' : ['000'],
    '001' : ['001', '000'],
}


def solve(demand_distribution, supply_distribution, demand : int , doi : int, n_warm_start_days : int, n_days : int,
          max_age: int = 35, policy_dict : dict = default_policy) -> dict:
    """
    This function used the fixed policy to make issuing decisions for a epriod of n_warm_start days and n_days
    :param demand_distribution: Tuple[dict, list] containing a dict with {blood_group : distribution}, list of
    included antigens
    :param supply_distribution: Tuple[dict, list] containing a dict with {blood_group : distribution}, list of
    included antigens
    :param demand: int: number of demand / supplied items per day (i.e. 10,20 or 50 RBCs per day)
    :param doi: int: days of inventory, how filled the inventory is before the first supply
    :param n_warm_start_days: int: number of warm start days before starting the real simulation
    :param n_days: int: number of days to run the issuing process
    :param max_age: int: default = 35, max age of the RBCs after which they perish
    :param policy_dict: dict: the dict containing the policy decisions.
    :return: eval metrics
    """

    # Blood group strings
    blood_groups_str = list(supply_distribution[0].keys())

    # Inititalize counters <<<< CHECK
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
        sys.stdout.write(f"Warm start day: {i} ({(round(i / n_warm_start_days, 1)) * 100}%)  \r")
        sys.stdout.flush()
        inv.new_blood_supply(demand)  # the RL model starts the day with an new supply.
        queue.new_request(demand)


        while not queue.is_empty():
            feasible_action = False  # Initialize when infeasible action

            item = queue.item_requested(inv.inventory, blood_groups_str)  # Get item from queue
            blood_binary = blood_groups_str[item]  # Get the binary (e.g. '000') for the index
            find = False  # ?

            # Loop through the inventory starting witht he oldest
            for i in list(range(1, max_age+1)):
                for blood_match in policy_dict[blood_binary]:
                    blood_match_idx = blood_groups_str.index(
                        blood_match)  # Get the idx for the binary
                    if inv.inventory[blood_match_idx][-i] > 0:  # when in stock
                        queue.remove_from_request(item)
                        inv.remove_from_inventory(blood_match_idx)
                        feasible_action = True
                        find = True
                        break
                if find is True:
                    break
            if feasible_action is False:
                queue.remove_from_request(item)
        inv.increase_age()  # new day

    # Real stuff
    # Loop over the days
    print(inv.inventory)
    inv.eval_boolean = True  # do record information
    for i in range(1, n_days + 1):
        sys.stdout.write(f"Day: {i} ({(round(i / n_days, 1)) * 100}%)  \r")
        sys.stdout.flush()

        inv.new_blood_supply(demand)  # the RL model starts the day with an new supply.
        queue.new_request(demand)
        inv.inventory_eval_queue(queue.queue)

        while not queue.is_empty():
            feasible_action = False  # Initialize when infeasible action
            item = queue.item_requested(inv.inventory, blood_groups_str)  # Get item from queue
            blood_binary = blood_groups_str[item]  # Get the binary (e.g. '000') for the index
            find = False

            # Loop through the day starting witht he oldest
            for i in list(range(1, max_age+1)):
                # Loop through the possible options in the fixed policy
                for blood_match in policy_dict[blood_binary]:
                    blood_match_idx = blood_groups_str.index(
                        blood_match)  # Get the idx for the binary
                    if inv.inventory[blood_match_idx][-i] > 0:  # when in stock
                        queue.remove_from_request(item)
                        inv.remove_from_inventory(blood_match_idx)
                        feasible_action = True
                        #                     inv.log_match_result(blood_binary, blood_match)
                        inv.inventory_eval_match(blood_binary, blood_match)
                        find = True
                        break
                if find is True:
                    break
            if feasible_action is False:
                queue.remove_from_request(item)
                inv.inventory_eval_infeasible(blood_binary)
                infeasible += 1

        inv.increase_age()

    return inv.eval
