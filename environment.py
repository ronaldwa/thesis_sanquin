import numpy as np
from gym import spaces
from gym.utils import seeding
import random
from csv import writer

import sanquin_inventory as sq_inv
from sanquin_inventory import find_compatible_blood_list


class Env:
    # Necessary FOR OPENAI GYM format:
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, supply_distribution, demand_distribution, max_age=5, amount=20, doi=4, eval_boolean=False,
                 max_day_in_episode=50,
                 obs_method=1, state_type='custom_category', file_name='place_holder', store_csv=False, verbose=1):
        """
        :param supply_distribution:  dict containing the supply distribution. Format: {blood_group : prevalence}
        :param demand_distribution:  a dict containing the demand distribution. Format: {blood_group : prevalence}
        :param max_age: int, the max age blood can reach
        :param amount: int, number of RBCs that are supplied / requested per day
        :param eval_boolean: boolean, indicating test evaluation must be run (only single runs)
        :param max_day_in_episode: number of days to be run per episode
        :param obs_method: int, there are two ways an observation can be communicated. Difference is the item requested:
        1. one hot encoded
        2. per antigen (1 for present, 0 for absent)
        :param state_type: string, indicating how the inventory is represented in the state. ('sum',
        'three categories', 'average_age', 'custom category')
        :param file_name: string name of how the file is called where the eval metrics are stored. (in /results folder)
        :param store_csv: boolean, True if CSV file with episode results need to be stored. Do not use when evaluating
        agent in-training
        :param verbose: int, 1 for printing the reuslts for each episode
        """

        # Initialize provided information
        self.supply_distribution_dict = supply_distribution
        self.demand_distribution_dict = demand_distribution
        self.blood_n = len(self.demand_distribution_dict.keys())  # number of blood groups in the data
        self.max_age = max_age  # Max age of the RBCs
        self.amount = amount  # Number of RBCs requested / supplied per day
        self.max_day = max_day_in_episode
        self.doi = doi  # Number of days to fill data in initialization
        self.obs_method = obs_method
        self.state_type = state_type
        self.n_antigens = len(list(self.supply_distribution_dict.keys())[0])
        self.file_name = file_name  # Export of eval doc in /results folder
        self.eval_boolean = eval_boolean  # evaluation run or not necessary or not (only single runs)
        self.store_csv = store_csv  # Stores the episode data in a CSV including outdating, do not use wh
        self.verbose = verbose

        # Initialize counters and booleans
        self.day_count = 0  # Tracks the day up to self.max_day
        self.repeat_problem = 0  # Keeps track of the number of repeated wrong the decisions the agent makes
        self.print_obs = True  # Initializes true so that when problem repeats, the observation is printed
        self.adjust_for_dummyvec = False  # For evaluation

        # initialize inventory
        self.inv = sq_inv.Inv(distribution=self.supply_distribution_dict, max_age=self.max_age,
                              eval_boolean=self.eval_boolean)
        self.inv.reset_inventory()  # Set all inventory values to zero
        self.inv.start_inventory(doi=self.doi, amount=self.amount)  # DIO = Days of Inventory

        # initialize queue
        self.Request = sq_inv.Request(distribution=self.demand_distribution_dict)
        self.current_item_requested = None  # Necessary to initialize state space

        # FOR GYM framework
        self.action_space = spaces.Discrete(len(demand_distribution.keys()))

        if obs_method == 1:  # inventory size + queue-size + n_antigens as columns
            y = self.inv.get_inventory_state(self.amount, state_type=self.state_type).shape[1]
            y += 2  # two columns for queue and inventory
            self.observation_space = spaces.Box(low=0, high=10, shape=(self.blood_n, y))  # Inventory

        elif obs_method == 2:  # inv-size + queue-size + n_antigens / n_bloodgroups
            x = self.inv.get_inventory_state(self.amount, state_type=self.state_type).size
            x += self.Request.queue_size
            x += self.n_antigens
            self.observation_space = spaces.Box(low=0, high=10, shape=(x,))  # Inventory

        # evaluation metrics
        self.inv.total_removed = 0
        self.inv.infeasible_match = 0
        self.inv.mismatch = 0
        self.inv.match = 0
        self.inv.notinstock = 0

        self.seed()

    # noinspection DuplicatedCode
    def reset(self):
        """
        Reset the environment in three steps:
        1. Reset the inventory. Every value in the inventory will be 0
        2. Start the inventory. An initial inventory will be created sampling an 'amount' supply
        for provided 'doi' days. (DOI = Days of inventory)
        3. A request is created for that day. Sampling 'amount' of values to answer to.
        :return: Observation (state)
        """

        if (self.adjust_for_dummyvec is True) and (
                self.eval_boolean is True):  # Adjustments to bypass dummyvec changes (stable baselines)
            self.adjust_for_dummyvec = False
            return self.get_obs()  # Open AI GYM - baselines

        # evaluation metrics
        self.inv.total_removed = 0
        self.inv.infeasible_match = 0
        self.inv.mismatch = 0
        self.inv.match = 0
        self.inv.notinstock = 0

        # Always start with a new day
        self.day_count = 0
        self.new_day()

        # Reset error counters
        self.repeat_problem = 0
        self.repeat_neg = 0

        return self.get_obs()  # Open AI GYM - baselines

    def new_day(self):
        """
        Initializes a new day in the following steps (order does not matter):
        1. Get a new batch of requests
        2. Get a new batch for the inventory
        """
        self.Request.new_request(self.amount)
        self.inv.new_blood_supply(self.amount)

        # Receive the first item in the queue
        self.current_item_requested = self.Request.item_requested(self.inv.inventory,
                                                                  list(self.demand_distribution_dict.keys()))
        self.day_count += 1

        # Eval
        if self.inv.eval_boolean:
            self.inv.inventory_eval_queue(self.Request.queue)

    def get_obs(self, method=0):
        """
        Returns the current state.
        :param method: int, there are two ways an observation can be communicated. Difference is the item requested:
        1. one hot encoded
        2. per antigen (1 for present, 0 for absent)
        :return: observation (np.array) vector that represents the state
        """
        observation = None  # initialize

        # Get Env default, other numbers are overruling for testing
        if method == 0:
            method = self.obs_method

        # Get the state of the inventory, this is a summary of the state
        inv_state = self.inv.get_inventory_state(amount=self.amount, state_type=self.state_type)

        if method == 1:
            # Create the requested bloodgroup as one-hot encoding
            choice_one_hot = np.zeros((self.blood_n,))
            if not self.Request.is_empty():
                choice_one_hot[
                    self.Request.item_requested(self.inv.inventory, list(self.demand_distribution_dict.keys()),
                                                type_queue='difficulty')] = 1

            queue = self.Request.queue
            queue = queue / self.amount  # Normalize the queue

            queue = np.reshape(queue, (len(self.Request.queue), 1))
            one_hot = np.reshape(choice_one_hot, (len(choice_one_hot), 1))

            observation = np.append(inv_state, queue, axis=1)
            observation = np.append(observation, one_hot, axis=1)

        if method == 2:

            # Get the binary representation of the requested antigen
            if not self.Request.is_empty():
                req = self.Request.item_requested(self.inv.inventory, list(self.demand_distribution_dict.keys()),
                                                  type_queue='difficulty')
                antigens_array = [float(x) for x in list(list(self.demand_distribution_dict.keys())[req])]
            else:
                antigens_array = [2] * self.n_antigens  # PLACE HOLDER

            queue = self.Request.queue

            queue = queue / self.amount  # Normalize the queue

            # Get 1-d array
            inv_state = inv_state.flatten()

            observation = np.append(inv_state, queue)
            observation = np.append(observation, antigens_array)

        return observation

    def render(self):
        """
        Rendering prints the eval metrics, and exports the data
        """
        if self.verbose == 1:
            print(
                f"{self.inv.match}|{self.inv.infeasible_match}|{self.inv.mismatch}|{self.inv.notinstock}|{self.inv.total_removed}")
            list_of_elem = [self.inv.match, self.inv.infeasible_match, self.inv.mismatch, self.inv.notinstock,
                            self.inv.total_removed]
            if self.store_csv:
                Env.append_list_as_row(self.file_name, list_of_elem)

    def render_blood_specific(self):
        """
        HELPER FUNCTION
        This function is necessary to bypass the dummyvec. It returns the evaluation metrics.
        :return: self.inv.eval, dict of evaluation metrcis
        """
        # print(self.inv.eval)
        return self.inv.eval

    def seed(self, seed=None):
        """
        Default BASELINES function
        Can set a random seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def find_repeat_error(self, add=True):
        """
        Keep track of repeating errors.
        :param add: Boolean, True if repetition of error, thus add to the counters. False if not repeated and all values
        can be reset
        """
        # In case of repetition of error
        if add is True:
            self.repeat_problem += 1

        # In case of no repetition action, set counter to 0
        if add is not True:
            self.repeat_problem = 0
            self.print_obs = True

    def get_correct_action(self, feasible_action):
        """
        Returns a random action that is feasible (in stock and compatible)
        :param feasible_action: list of actions that are feasible
        :return: int, random action that is feasible
        """
        possible_actions = []
        for blood_group in feasible_action:
            if self.inv.in_stock(blood_group):
                possible_actions.append(blood_group)
        return random.choice(possible_actions)

    def step(self, action):
        """
        Does the step in the RL algorithm. Checks whether the episode is done and calculates the reward for each step.
        In this order:
        - Does an feasible action exists?
        - Is the match compatible?
        - Is the match in stock?
        - Good match

        :param action: (int) index of the blood that will be matched to the requested blood
        :return: (tuple): state (np.array), reward (float), done(boolean), {}
        """
        # Initialize
        current_item_requested = self.Request.item_requested(self.inv.inventory,
                                                             list(self.demand_distribution_dict.keys()),
                                                             type_queue='difficulty')
        _, feasible_action = find_compatible_blood_list(self.demand_distribution_dict.keys(), current_item_requested)

        # Check infeasible action
        step_not_done, return_values = self.check_infeasible_action(current_item_requested, action, feasible_action)

        # Check if it is a wrong match
        if step_not_done:
            step_not_done, return_values = self.check_wrong_match(current_item_requested, action, feasible_action)

        # Check if the match is in stock
        if step_not_done:
            step_not_done, return_values = self.check_in_stock(current_item_requested, action, feasible_action)

        # Check if is a correct match
        if step_not_done:
            return_values = self.check_correct_match(current_item_requested, action, feasible_action)

        return self.get_obs(), return_values[0], return_values[1], return_values[2]  # obs, reward, done, info

    def check_infeasible_action(self, current_item_requested, action, feasible_action):
        """
        Checks whether the models action is infeasible. This means no blood group in stock that could be a match
        :param current_item_requested: int, the blood group currently in request
        :param action: int, the blood proposed
        :param feasible_action: list, of blood groups that are matching
        :return: boolean, list
        boolean: is True to indicate there is no action possible, False if an option is possible
        list: [reward, done, {'match_type': 'no_option'}]
        """
        # Initialize
        done = False  # done flag
        possible_action_exists = False

        # Check if a feasible blood group is in stock
        for blood_group in feasible_action:
            if self.inv.in_stock(blood_group):
                possible_action_exists = True
                break

        # If no option exists, give large penalty
        if possible_action_exists is False:
            self.inv.inventory_performance('infeasible')
            reward = -100

            self.Request.remove_from_request(current_item_requested)  # remove requested item from queue

            # Check whether this is the end of the day (queue is empty)
            if self.Request.is_empty():
                reward -= 10 * sum(self.inv.increase_age())
                # Check if this was the last day of the episode
                if self.day_count >= self.max_day:
                    done = True
                    self.day_count = 0
                    self.render()
                else:
                    self.new_day()

            # Evaluation
            if self.inv.eval_boolean:
                self.inv.eval['infeasible'][list(self.supply_distribution_dict.keys())[current_item_requested]] += 1

            return False, [reward, done, {'match_type': 'no_option'}]

        return True, []

    def check_wrong_match(self, current_item_requested, action, feasible_action):
        """
        Check whether the match is wrong, dus not matching.
        :param current_item_requested: int, the blood group currently in request
        :param action: int, the blood proposed
        :param feasible_action: list, of blood groups that are matching
        :return: boolean, list
        boolean: is True to indicate there is no action possible, False if an option is possible
        list: [reward, done, {'match_type': 'no_option'}]
        """
        # Initialize
        done = False

        # Repeat problem
        if self.repeat_problem > 100:
            correct_action = self.get_correct_action(feasible_action)
            reward = -100
            return_values = self.check_correct_match(current_item_requested, correct_action, feasible_action, reward)

            if self.inv.eval_boolean:
                self.inv.inventory_eval_match(req_blood=current_item_requested, sup_blood=action)

            return False, return_values

        # penalize if not matchable
        if action not in feasible_action:
            reward = -10

            self.inv.inventory_performance('mismatch')  # Keep track of values

            self.find_repeat_error(add=True)  # Add to repeat error, because no correct match
            return False, [reward, done, {'match_type': 'no_match'}]
        return True, []

    def check_in_stock(self, current_item_requested, action, feasible_action):
        """
        Checks whether the proposed action is in stock
        :param current_item_requested: int, the blood group currently in request
        :param action: int, the blood proposed
        :param feasible_action: list, of blood groups that are matching
        :return: boolean, list
        boolean: is True to indicate there is no action possible, False if an option is possible
        list: [reward, done, {'match_type': 'no_option'}]
        """

        done = False

        # If a repeat problem
        if self.repeat_problem > 100:
            correct_action = self.get_correct_action(feasible_action)
            reward = -100
            return_values = self.check_correct_match(current_item_requested, correct_action, feasible_action, reward)

            if self.inv.eval_boolean:
                self.inv.inventory_eval_match(req_blood=current_item_requested, sup_blood=action)

            return False, return_values

        if not self.inv.in_stock(action):
            reward = -10
            self.inv.inventory_performance('notinstock')  # keep track of evaluation values

            self.find_repeat_error(add=True)  # Add to repeat error, because no correct match
            return False, [reward, done, {'match_type': 'not_in_stock'}]
        return True, []

    def check_correct_match(self, current_item_requested, action, feasible_action, reward=0.1):
        """
        This is a correct match, because previous conditions where met. Match the invenotry to the request.
        :param current_item_requested: int, the blood group currently in request
        :param action: int, the blood proposed
        :param feasible_action: list, of blood groups that are matching
        :param reward: reward is an int, providing the reward that will be given
        :return: boolean, list
        boolean: is True to indicate there is no action possible, False if an option is possible
        list: [reward, done, {'match_type': 'no_option'}]
        """
        done = False
        # eval
        self.inv.inventory_performance('match')

        if self.inv.eval_boolean:
            self.inv.inventory_eval_match(req_blood=current_item_requested, sup_blood=action)

        self.inv.remove_from_inventory(action)  # remove from inventory

        self.Request.remove_from_request(current_item_requested)  # remove requested item from queue

        # Check end of the day
        if self.Request.is_empty():
            # Check last day of the episode
            if self.day_count >= self.max_day:
                reward -= 10 * sum(self.inv.increase_age())
                done = True
                self.day_count = 0
                self.render()

                self.adjust_for_dummyvec = True
            # Not end of the episode, thus new day
            else:
                reward -= 10 * sum(self.inv.increase_age())
                self.new_day()

        self.find_repeat_error(add=False)  # Reset the repeat error
        return [reward, done, {}]

    @staticmethod
    def append_list_as_row(file_name, list_of_elem):
        """
        exports information on training process. See render function for what information.
        :param file_name: name of the file where the results will be stored
        :param list_of_elem: the data to be written down in the csv
        :return:
        """
        # Open file in append mode
        with open('results/evaluation_metrics_data/' + file_name + '.csv', 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)

    def set_days(self, n_days):
        """
        HELPER FUNCTION
        This function is necessary to bypass the dummyvec. It changes the number of days.
        """
        self.day_count = 0  # Reset counter
        self.max_day = n_days

    def change_eval_boolean(self, eval_boolean):
        """
        HELPER FUNCTION
        This function is necessary to bypass the dummyvec. It changes the eval boolean.
        """
        self.inv.eval_boolean = eval_boolean
