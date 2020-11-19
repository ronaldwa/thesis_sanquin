"""
This file contains two classes:
1. The inventory
2. The queue
"""

# ----- Import packages
import numpy as np
import random


# ----- Global functions
def sample(distribution_dict, n_samples):
    """
    Creates a sample given a certain distribution.
    Returns a list with the number of samples per given index distribution.
    :param distribution_dict: (dict) a dict containing the distribution. Format: {blood_group : prevalence}
    :param n_samples: (int) number of samples that are requested
    :return: (dict) the provided samples. Format: {blood group : number of items, etc.}
    """
    keys, probs = zip(*distribution_dict.items())
    sample_np = np.random.choice(keys, n_samples, p=probs)
    return {key: list(sample_np).count(key) for key in keys}


def find_compatible_blood_list(blood_groups_binary, requested_bloodgroup_idx, factor_type='receivability'):
    """
    Finds compatible blood given the blood groups as binary strings.
    :param blood_groups_binary: (list) All the blood groups in binary string format. (e.g. ['01', '11'] for B and AB.
    :param requested_bloodgroup_idx: (int) index for the asked blood. In the above example, 0 would be '01'
    :param factor_type: string: 'receivability' or 'usability'
    :return: tuple of two lists:
        blood_list: (list): All compatible blood in binary string format: (e.g. ['00', '01', '11'])
        idx_list: (list): Index of all compatible blood: [0, 1, 2].
    """
    # Transform input
    compatible_blood = np.array([list(bloodgroup) for bloodgroup in blood_groups_binary], dtype=np.int)
    requested_bloodgroup = list(blood_groups_binary)[requested_bloodgroup_idx]

    # Loop through antigens and determine whether the blood group is compatible
    if factor_type == 'receivability':
        for i, v in enumerate(requested_bloodgroup):
            # If antigen not in blood binary (0) than the provided blood groups cannot contain this antigen
            if v == '0':
                compatible_blood = compatible_blood[compatible_blood[:, i] != 1]

    if factor_type == 'usability':
        for i, v in enumerate(requested_bloodgroup):
            # If antigen in blood binary (1) than the provided blood groups cannot contain this antigen
            if v == '1':
                compatible_blood = compatible_blood[compatible_blood[:, i] == 1]

    # Create export lists
    blood_list = [''.join(str(e) for e in list(bloodgroup)) for bloodgroup in compatible_blood]
    idx_list = [list(blood_groups_binary).index(blood) for blood in blood_list]

    return blood_list, idx_list


# ---- CLASSES
class Inv:

    def __init__(self, distribution, max_age, eval_boolean=False):
        """
        :param distribution: dict, distribution of blood groups. Format {'bloodgroup':prevalence, ..}
        :param max_age: int, the max age blood can reach
        :param eval_boolean: boolean, False if not a
        """
        # Initialize variables
        self.distribution = distribution
        self.blood_groups = list(distribution.keys())
        self.eval_boolean = eval_boolean
        self.max_age = max_age
        self.compatible_match = {idx: find_compatible_blood_list(self.blood_groups, idx)
                                 for idx, blood in enumerate(self.blood_groups)}

        # If evaluation, keep track of all kinds of values
        self.eval = {'donated': {},
                     'requested': {},
                     'provided': {},
                     'removed': {},
                     'match': {},
                     'age': {},
                     'infeasible': {}}
        for key in self.eval.keys():
            for blood_group in self.blood_groups:
                if key not in ['match', 'age']:
                    self.eval[key][blood_group] = 0
        for blood_group in self.blood_groups:
            self.eval['match'][blood_group] = {}
            self.eval['age'][blood_group] = {}

        # Keep track of all removed values
        self.removed = np.zeros((len(self.blood_groups)))

        # Initialize inventory by running reset
        self.reset_inventory()

    def reset_inventory(self):
        """
        Initializes and reset inventory
        inventory is basically a matrix with size of number of blood groups * the max age of the blood.
        """
        # Initialize
        self.inventory = np.zeros((len(self.blood_groups), self.max_age))

        # evaluation metrics
        self.total_removed = 0
        self.infeasible_match = 0
        self.mismatch = 0
        self.match = 0
        self.notinstock = 0

    def start_inventory(self, doi, amount):
        """
        Start the inventory by randomly filling a doi number of days of inventory
        :param doi: int, number of days in inventory that need to be supplied
        :param amount: int, number of RBCs that are supplied
        """
        for i in range(doi):
            self.new_blood_supply(amount)
            self.increase_age()  # Note that the first day will be empty. This is filled by the day
            # (day starts with adding new supply)

    def new_blood_supply(self, amount, mode='not_add'):
        """
        Provide a new blood supply of size amount
        :param amount: int, number of RBCs that are requested
        :param mode: string, 'add', the number of RBCs that are removed from the total inventory are added.
        """
        new_blood_amount = list(sample(self.distribution, amount).values())
        self.inventory[:, 0] = new_blood_amount

        # evaluation metrics
        if self.eval_boolean:
            for blood_group_index, quantity in enumerate(new_blood_amount):
                blood_group_binary = self.blood_groups[blood_group_index]
                self.eval['donated'][blood_group_binary] += quantity

    def increase_age(self):
        """
        By rolling the inventory, the first column becomes the column to remove.
        :return self.removed; numpy.array, an array containing the removed RBCs.
        """
        self.inventory = np.roll(self.inventory, 1)  # last row will become first row
        self.removed = np.copy(self.inventory[:, 0])  # return what was removed
        self.inventory[:, 0] = 0  # Replace first row with 0

        # evaluation metrics
        self.total_removed += int(sum(self.removed))
        if self.eval_boolean:
            for blood_group_index, quantity in enumerate(self.removed):
                blood_group_binary = self.blood_groups[blood_group_index]
                self.eval['removed'][blood_group_binary] += quantity
        return self.removed

    def get_inventory_state(self, amount, state_type='three_categories', custom_cat=[8, 9, 9, 9]):
        """
        :param custom_cat:
        :param amount: int, number of blood supplied
        :param state_type: string, category of how to get the inventory state. 'sum', 'three_categories' or 'average_age'
        - Three categories, this provides a representation of the inventory by including three categories (fresh,
        core and danger age)
        - Average age, representation of blood where the average age and the mean age are provided.
        :param custom_cat:
        :return: the state
        """
        # SUM - only the sum of the number of blood
        if state_type == 'sum':
            state = self.inventory[:, :].sum(axis=1)
            state = (state - state.min()) / (state.max() - state.min() + 0.001)
            return state

        # 3C - three categories of a certain size, that provide the sum in three categories.
        if state_type == 'three_categories':
            # Create the three categories:
            size_category = 5
            s0 = self.inventory[:, :size_category].sum(axis=1) / len(
                self.inventory[:, :size_category][0])  # age = 0-size_category
            s1 = self.inventory[:, size_category:-size_category].sum(axis=1) / len(
                self.inventory[:, size_category:-size_category][0])  # middle ages
            s2 = self.inventory[:, -size_category:].sum(axis=1) / len(
                self.inventory[:, -size_category:][0])  # last categories

            # Create the state
            state = np.zeros((len(self.blood_groups), 3))  # 3 is the number of categories
            state[:, 0] = s0
            state[:, 1] = s1
            state[:, 2] = s2

            # Normalize, so that all values are withing 0-1.
            # We assume that there is on average in the category never more than 2 times the inventory
            blood_distr = list(self.distribution.values())  # IS FOUT, moet SUPPLY zijn

            for index, row in enumerate(state):
                stock = blood_distr[index] * amount * 2
                state[index] = row / stock

            return state

        if type(state_type) == 'custom_category':
            # If custom cateogry, the category exists of a provided list of bins that the ages belong to
            # The total number bins must equal the max age of the RBCs
            assert sum(
                custom_cat) == self.max_age, "The sum of the categories are not equal to the maximum age of the " \
                                             "products "

            # Fill the bins
            state = np.zeros((len(self.blood_groups), len(custom_cat)))
            cat_prev = 0
            for idx, cat in enumerate(custom_cat):
                cat = cat_prev + cat
                state[:, idx] = self.inventory[:, cat_prev:cat].sum(axis=1) / len(
                    self.inventory[:, cat_prev:cat][0])
                cat_prev = cat

            # Normalize, so that all values are withing 0-1.
            # We assume that there is on average in the category never more than 2 times the inventory
            blood_distr = list(self.distribution.values())  # IS FOUT, moet SUPPLY zijn

            for index, row in enumerate(state):
                stock = blood_distr[index] * amount * 2
                state[index] = row / stock

            return state

        if state_type == 'average_age':
            # Legacy, not used.
            # Idea behind average age: provide a columns with the average age per blood group and a
            # column with the quantity per blood group.

            # Get the sum of the inventory
            s0 = self.inventory[:, :].sum(axis=1)
            s0 = (s0 - s0.min()) / (s0.max() - s0.min())

            # Step 2 get the average age from 0-1
            s1 = []
            for blood_group in self.inventory:
                blood_age = 0
                blood_sum = 0
                for day, quantity in enumerate(blood_group):
                    day = day + 1  # to correct for starting at 0
                    blood_age += day * quantity
                    blood_sum += quantity
                s1.append(blood_age / (blood_sum + 0.0001))
            s1 = np.asarray(s1)
            s1 = (s1 - s1.min()) / (s1.max() - s1.min())

            state = np.zeros((len(self.blood_groups), 2))
            state[:, 0] = s0
            state[:, 1] = s1

            return state

    def in_stock(self, blood_idx):
        """
        Checks whether the provided index of the blood group is in stock.
        :param blood_idx: int, index of the blood group
        :return: boolean, TRUE if in stock, False if not
        """
        if sum(self.inventory[blood_idx]) > 0:
            return True
        else:
            False

    def remove_from_inventory(self, blood_idx):
        """
        Removes an item from the inventory for the provided index of the blood group
        :param blood_idx: int, index of the blood group
        :return: the age of the RBC that is removed
        """
        for idx, value in enumerate(self.inventory[blood_idx][::-1]):
            if value > 0:
                self.inventory[blood_idx][-idx - 1] -= 1
                # eval:
                if self.eval_boolean:
                    blood_group_binary = self.blood_groups[blood_idx]
                    self.eval['provided'][blood_group_binary] += 1
                    self.inventory_eval_age_match(blood_group_binary, self.max_age - idx)
                return int(self.max_age - idx)

    def inventory_performance(self, action_type):
        """
        Get the action type of the action, so that the algorithm can keep track of the number of type results.
        :param action_type: string, information on the type of result of an action for evaluation
        :return:
        """
        if action_type == 'mismatch':
            self.mismatch += 1
        elif action_type == 'infeasible':
            self.infeasible_match += 1
        elif action_type == 'notinstock':
            self.notinstock += 1
        elif action_type == 'match':
            self.match += 1
        else:
            print('Error when logging the action type')

    def inventory_eval_queue(self, request_list):
        """
        EVALUATION helper function
        provided the queue, it tracks all requested items
        :param request_list: list, equal to the queue
        """
        # For every blood group add the requested numbers to the eval variables
        for blood_group_index, quantity in enumerate(request_list):
            blood_group_binary = self.blood_groups[blood_group_index]
            self.eval['requested'][blood_group_binary] += quantity

    def inventory_eval_match(self, req_blood, sup_blood):
        """
        EVALUATION helper function
        Keeps track of all matches made.
        :param req_blood: int or string, the requested blood group
        :param sup_blood: int or string, the supplied blood group
        """

        if type(req_blood) == int:
            req_blood_group_binary = self.blood_groups[req_blood]
            sup_blood_group_binary = self.blood_groups[sup_blood]
        else:
            req_blood_group_binary = req_blood
            sup_blood_group_binary = sup_blood

        # Keep track of the match
        if req_blood_group_binary not in self.eval['match']:
            self.eval['match'][req_blood_group_binary] = {sup_blood_group_binary: 1}
        elif sup_blood_group_binary not in self.eval['match'][req_blood_group_binary]:
            self.eval['match'][req_blood_group_binary][sup_blood_group_binary] = 1
        else:
            self.eval['match'][req_blood_group_binary][sup_blood_group_binary] += 1

    def inventory_eval_age_match(self, sup_blood, age):
        """
        EVALUATION helper function
        Keeps track of the age of the provided blood
        :param sup_blood: string, binary of the blood that is supplied
        :param age: int, age of the blood (up to 35)
        """
        if sup_blood not in self.eval['age']:
            self.eval['age'][sup_blood] = {age: 1}
        elif age not in self.eval['age'][sup_blood]:
            self.eval['age'][sup_blood][age] = 1
        else:
            self.eval['age'][sup_blood][age] += 1

    def inventory_eval_infeasible(self, req_blood):
        self.eval['infeasible'][req_blood] += 1


class Request:

    def __init__(self, distribution):
        """
        :param distribution: (dict) a dict containing the distribution. Format: {blood_group : prevalence}
        """
        self.distribution = distribution
        self.queue = np.zeros((len(self.distribution.keys())))
        self.queue_size = len(self.distribution.keys())

        self.compatible_match = {idx: find_compatible_blood_list(self.distribution.keys(), idx) for idx, blood in
                                 enumerate(self.distribution.keys())}  # List with compatible matches

        # Evaluation
        self.eval_requested = {}
        for blood_group in list(self.distribution.keys()):
            self.eval_requested[blood_group] = 0

    def new_request(self, amount):
        """
        Creates a new request
        :param amount: number of new items to sample
        """
        self.queue[:] = list(sample(self.distribution, amount).values())

    def remove_from_request(self, blood_idx):
        """
        A specific item to remove from the queue
        :param blood_idx: int, index of the blood group that needs to be removed.
        """
        assert self.queue[
                   blood_idx] - 1 >= 0, f"The value was already 0, cannot deduct from zero. {blood_idx}, {self.queue}"
        self.queue[blood_idx] -= 1  # when matched, remove

    def is_empty(self):
        """
        Returns whether the queue is empty
        :return: boolean, True if empty, False if not
        """
        if sum(self.queue) == 0:
            return True
        else:
            return False

    def item_requested(self, inventory, type_queue='difficulty'):
        """
        Returns the item that is requested. There are three ways to do this:
        1. First, simply running through the queue from left to right (which is by blood group)
        2. Random, randomly selecting a blood group in the queue
        3. Difficulty, based on difficulty to match. Start with most difficult one. This method calculated with each
        individual request what the most difficult to match item is.
        :param inventory: array, inventory
        :param type_queue: the type of providing the item, default is 'difficulty'
        :return blood_idx: int, the index of the blood group
        """
        # First item -> latest item
        if type_queue == 'first':
            for blood_idx, i in enumerate(self.queue):
                if i > 0:
                    return blood_idx

        # Random item
        if type_queue == 'random':
            while True:
                blood_idx = random.randint(0, len(self.queue) - 1)
                if self.queue[blood_idx] > 0:
                    return blood_idx

        # "Most difficult first"
        if type_queue == 'difficulty':
            # Initialize the receivable dict
            receivable_dict = {}

            for idx, value in enumerate(self.queue):
                if value != 0:
                    # Initialize the receivable sum
                    receivable_sum = 0

                    # get possible actions
                    _, feasible_actions = self.compatible_match[idx]

                    # Calculate per compatible bloodgroup how many matches are still possible
                    for blood_binary in list(feasible_actions):
                        # blood_binary = binary_code.index(blood)
                        receivable_sum += inventory[blood_binary].sum()

                    receivable_dict[idx] = receivable_sum

            diff = (min(receivable_dict, key=receivable_dict.get))
            return diff
