
from Agents.DDPG import DDPGNet

import argparse
import os

import numpy as np
import torch
import gym
import sys
sys.path.append("..")
from Common import utils, utils_for_q_learning

from Common.logging_utils import MetaLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_param_directory",
                        required=False,
                        default="./hyper_parameters",
                        type=str)

    parser.add_argument("--hyper_parameter_name",
                        required=True,
                        help="0, 10, 20, etc. Corresponds to .hyper file",
                        default="0")  # OpenAI gym environment name

    parser.add_argument("--experiment_name",
                        type=str,
                        help="Experiment Name",
                        required=True)

    parser.add_argument("--run_title",
                        type=str,
                        help="subdirectory for this run",
                        required=True)


    parser.add_argument("--seed", default=0,
                        type=int)  # Sets Gym, PyTorch and Numpy seeds


    args, unknown = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}

    full_experiment_name = os.path.join(args.experiment_name, args.run_title)

    utils.create_log_dir(full_experiment_name)
    hyperparams_dir = utils.create_log_dir(
        os.path.join(full_experiment_name, "hyperparams"))

    params = utils.get_hyper_parameters(args.hyper_parameter_name,
                                        args.hyper_param_directory)
    
    params['hyper_parameters_name'] = args.hyper_parameter_name

    for arg_name, arg_value in other_args:
        utils.update_param(params, arg_name, arg_value)

    params['hyperparams_dir'] = hyperparams_dir
    
    utils.save_hyper_parameters(params, args.seed)
    # Logging
    meta_logger = MetaLogger(full_experiment_name)
    logging_filename = f"seed_{args.seed}.pkl"

    meta_logger.add_field("episodic_rewards", logging_filename)
    meta_logger.add_field("evaluation_rewards", logging_filename)
    meta_logger.add_field("average_loss", logging_filename)
    meta_logger.add_field("average_q", logging_filename)
    meta_logger.add_field("average_q_star", logging_filename)
    meta_logger.add_field("all_times", logging_filename)
    meta_logger.add_field("average_update_time", logging_filename)

    # The rest is up to you!


    env = gym.make(params['env_name'])

    params['seed'] = args.seed
    params['env'] = env
    utils_for_q_learning.set_random_seed(params)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    s0 = env.reset()

    # baseline methods

    Q_object = DDPGNet(
        params,
        env,
        state_size=len(s0),
        action_size=env.action_space.shape[0],
        device=device,
    )
    Q_target = DDPGNet(
        params,
        env,
        state_size=len(s0),
        action_size=env.action_space.shape[0],
        device=device,
    )


    Q_target.eval()
    
    utils_for_q_learning.sync_networks(
        target=Q_target,
        online=Q_object,
        alpha=params['target_network_learning_rate'],
        copy=True)

    episodic_rewards_list = []
    evaluation_rewards_list = []

    if params['env_name'] == "HalfCheetah-v3" or params['env_name'] == 'Ant-v3':
        max_episode_steps = 2000
    else:
        max_episode_steps = 1600
    
    for episode in range(params['max_episode']):
        s, done, t = env.reset(), False, 0
        episodic_rewards = 0.
        print(f"Episode {episode}")   
        while not done:
            a = Q_object.e_greedy_policy(s, episode + 1, 'train')
            sp, r, done, info = env.step(np.array(a))
            done_p = False if t == max_episode_steps else done
            done = True if t == max_episode_steps else done
            episodic_rewards += r
            t += 1
            Q_object.buffer_object.append(s, a, r, done_p, sp)
            s = sp

        print(f"Episodic Reward ({episode}): {episodic_rewards}")

        meta_logger.append_datapoint("episodic_rewards", episodic_rewards, write=True)
        # episodic_rewards_list.append(episodic_rewards)

        total_loss = []

        qs = []
        q_stars = []

        for _ in range(params['updates_per_episode']):
            # Maybe doing the softmax thing.
            loss, update_params = Q_object.update(Q_target)
            total_loss.append(loss)
            qs.append(update_params["average_Q"])
            q_stars.append(update_params["average_Q_star"])

        average_loss = np.mean(total_loss)

        meta_logger.append_datapoint("average_loss", average_loss, write=True)

        average_Q = np.mean(qs)
        average_Q_star = np.mean(q_stars)

        meta_logger.append_datapoint("average_q", average_Q, write=True)
        meta_logger.append_datapoint("average_q_star", average_Q_star, write=True)

        # Evaluation now.
        evaluation_rewards = 0
        s, done, t = env.reset(), False, 0

        while not done:
            a = Q_object.e_greedy_policy(s, episode + 1, 'test')
            sp, r, done, _ = env.step(np.array(a))
            evaluation_rewards += r
            s, t = sp, t + 1

        print(f"Evaluation Reward ({episode}): {evaluation_rewards}")

        meta_logger.append_datapoint("evaluation_rewards", evaluation_rewards, write=True)