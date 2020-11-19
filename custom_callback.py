"""
Custom callbacks
- StopTrainingOnDecayingRewardThreshold
Idea is that the reward threshold can decay over time, allowing for less good results over time
"""
from stable_baselines.common.callbacks import BaseCallback
import math


class StopTrainingOnDecayingRewardThreshold(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).
    It must be used with the `EvalCallback`.
    :param reward_threshold: (float)  Minimum expected reward per episode
        to stop training.
    :param verbose: (int)
    """

    def __init__(self, max_reward: float, episode_decay: float,
                 reward_decay: float, no_reward_episodes: int, verbose: int = 0):
        """
        :param max_reward: float, max reward that can be received
        :param episode_decay: float, after how many episodes the reward is decayed
        :param reward_decay: float, proportion with what the reward will be decayed. ie. 0.05
        :param no_reward_episodes: int, number of episodes before starting to decay
        :param verbose: int, if > 0 print information
        """
        super(StopTrainingOnDecayingRewardThreshold, self).__init__(verbose=verbose)
        self.max_reward = max_reward
        self.episode_decay = episode_decay
        self.reward_decay = reward_decay
        self.no_reward_episodes = no_reward_episodes
        self.reward_threshold = max_reward

    def change_reward_threshold(self) -> None:
        """
        Change the reward threshold to stop training at
        """
        if (self.num_timesteps - self.no_reward_episodes) > 0:
            reduce_value = math.floor((self.num_timesteps - self.no_reward_episodes) / self.episode_decay)
            self.reward_threshold = round(self.max_reward * (1 - (reduce_value * self.reward_decay)), 2)

            if self.reward_threshold > self.max_reward:
                self.reward_threshold = self.max_reward

            if self.verbose > 0:
                print(f'Current reward threshold  = {self.reward_threshold}')

    def _on_step(self) -> bool:
        """
        On the step check whether the reward is already smaller than the threshold.
        Also updates the threshold
        :return: continue_training: bool, True if continue training because reward < reward threshold
        """
        assert self.parent is not None, ("`StopTrainingOnRewardThreshold` callback must be used "
                                         "with an `EvalCallback`")

        self.change_reward_threshold()

        continue_training = bool(round(self.parent.best_mean_reward, 2) < round(self.reward_threshold, 2))
        if self.verbose > 0 and not continue_training:
            print("Stopping training because the mean reward {:.2f} "
                  " is above the threshold {}".format(self.parent.best_mean_reward, self.reward_threshold))
        return continue_training
