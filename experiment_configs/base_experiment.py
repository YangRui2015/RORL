from lifelong_rl.samplers.utils.model_rollout_functions import policy
import numpy as np
import torch
import time

import gtimer as gt
import os
import random

from experiment_utils.prepare_data import load_hdf5

from lifelong_rl.core.logging.logging import logger
from lifelong_rl.core.logging.logging_setup import setup_logger
from lifelong_rl.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from lifelong_rl.data_management.replay_buffers.mujoco_replay_buffer import MujocoReplayBuffer
from lifelong_rl.envs.env_processor import make_env
import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.envs.env_utils import get_dim
from lifelong_rl.samplers.data_collector.path_collector import MdpPathCollector, LatentPathCollector
from lifelong_rl.samplers.data_collector.step_collector import MdpStepCollector, RFCollector, \
    GoalConditionedReplayStepCollector
from lifelong_rl.samplers.utils.rollout_functions import rollout_with_attack, rollout



def simple_evaluation(config, variant, obs_dim, action_dim, num_paths=5):
    paths = []
    eval_env = config['evaluation_env']
    if variant['eval_attack']:
        from attackers.attacker import Evaluation_Attacker
        attacker = Evaluation_Attacker(config['evaluation_policy'],
                                    config['qfs'], variant['eval_attack_eps'],
                                    obs_dim, action_dim, obs_std=variant['normalization_info']['obs_std'],
                                    attack_mode=variant['eval_attack_mode'])
        for _ in range(num_paths):
            path = rollout_with_attack(eval_env, config['evaluation_policy'], attacker,
                                config['offline_kwargs']['max_path_length'], render=False)
            paths.append(path)
    else:   
        for _ in range(num_paths):
            path = rollout(eval_env, config['evaluation_policy'], 
                                config['offline_kwargs']['max_path_length'], render=True)
            paths.append(path)
    return paths


def experiment(
        # Variant
        variant,

        # Experiment config
        experiment_config,

        # Misc arguments
        exp_postfix='',
        use_gpu=True,
        log_to_tensorboard=False,
        base_log_dir='results',
        # data config
        data_args=None,
):
    print('base_experiment begin')
    """
    Reset timers
    (Useful if running multiple seeds from same command)
    """
    gt.reset()
    gt.start()
    """
    Setup logging
    """
    date_time = '_'.join(time.ctime().split(' '))
    seed = variant['seed']
    log_dir = os.path.join(
        '{}{}'.format(variant['algorithm'], exp_postfix),
        '{}_{}_{}'.format(variant['env_name'], date_time,  seed))

    print('setup logger')
    setup_logger(log_dir=log_dir,
                 variant=variant,
                 log_to_tensorboard=log_to_tensorboard,
                 base_log_dir=base_log_dir)
    print('logger set!')
    output_csv = logger.get_tabular_output()
    """
    Set GPU mode for pytorch (+ possible other things later)
    """
    ptu.set_gpu_mode(use_gpu)
    """
    Set experiment seeds
    """
    # fix random seed for experiment
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    """
    Environment setup
    """
    expl_env, env_infos = make_env(variant['env_name'],
                                    **variant.get('env_kwargs', {}))

    obs_dim = get_dim(expl_env.observation_space)
    action_dim = get_dim(expl_env.action_space)

    if env_infos['mujoco']:
        replay_buffer = MujocoReplayBuffer(variant['replay_buffer_size'], expl_env)
    else:
        replay_buffer = EnvReplayBuffer(variant['replay_buffer_size'], expl_env)
    

    eval_env, _ = make_env(variant['env_name'],
                        **variant.get('env_kwargs', {}))
    

    if env_infos['mujoco']:
        eval_env._max_episode_steps = 1000
        variant['offline_kwargs']['max_path_length'] = 1000
    else:
        max_epi_steps = eval_env._max_episode_steps
        variant['offline_kwargs']['max_path_length'] = max_epi_steps
    
    
    """
    Import offline data from d4rl
    """
    if not (variant['eval_no_training']):
        load_hdf5(expl_env, replay_buffer, data_args)
        
    # get obs mean and std
    if variant['norm_input']:
        from attackers.data_mean_std import get_obs_mean_std
        obs_mean, obs_std = get_obs_mean_std(variant['env_name'])
        variant['normalization_info'] = {'obs_mean': obs_mean,'obs_std': obs_std}
        if not (variant['eval_no_training']):
            assert np.all(np.abs(obs_mean -  replay_buffer._observations.mean(axis=0)) <= 1e-5) and np.all(np.abs(obs_std - replay_buffer._observations.std(axis=0)) <= 1e-5)
    else:
        variant['normalization_info'] = {'obs_mean': np.zeros(obs_dim),'obs_std': np.ones(obs_dim)}
    ###############################
    obs_dim = replay_buffer.obs_dim()

    """
    Experiment-specific configuration
    """
    config = experiment_config['get_config'](
        variant,
        expl_env=expl_env,
        eval_env=eval_env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        replay_buffer=replay_buffer,
    )

    if 'algorithm_kwargs' not in config:
        config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())
    if 'offline_kwargs' not in config:
        config['offline_kwargs'] = variant.get('offline_kwargs', dict())


    """
    Path collectors for sampling from environment
    """

    collector_type = variant.get('collector_type', 'step')

    if collector_type == 'gcr':
        eval_path_collector = GoalConditionedReplayStepCollector(
            eval_env,
            config['evaluation_policy'],
            replay_buffer,
            variant['resample_goal_every'],
        )
    else:
        eval_path_collector = MdpPathCollector(
            eval_env,
            config['evaluation_policy'],
        )

    """
    Finish timer
    """
    gt.stamp('initialization', unique=False)
    """
    Offline RL pretraining
    """
    logger.set_tabular_output(
        os.path.join(logger.log_dir, 'offline_progress.csv'))

    offline_algorithm = experiment_config['get_offline_algorithm'](
        config,
        eval_path_collector=eval_path_collector,
    )
    offline_algorithm.to(ptu.device)

    if variant['eval_no_training']:
        import d4rl
        num_paths = 10
        time_start = time.time()
        paths = simple_evaluation(config, variant, obs_dim, action_dim, num_paths=num_paths)
        returns = [sum(path["rewards"]) for path in paths]
        r_std, r_min, r_max = np.std(returns), np.min(returns), np.max(returns)
        norm_returns = [d4rl.get_normalized_score(variant['env_name'], x) for x in returns]
        r_std_norm, r_mean_norm = np.std(norm_returns), np.mean(norm_returns)
        from lifelong_rl.util import eval_util
        logger.record_dict(eval_util.get_generic_path_information(paths), prefix="evaluation/")
        logger.record_tabular('Returns Std', r_std)
        logger.record_tabular('Returns Min', r_min)
        logger.record_tabular('Returns Max', r_max)
        logger.record_tabular('Norm Returns Mean', r_mean_norm)
        logger.record_tabular('Norm Returns Std', r_std_norm)
        logger.record_tabular('Returns Min', r_min)
        logger.record_tabular('Returns Max', r_max)
        logger.record_tabular('Epoch Time (min)', (time.time() - time_start)/60)
        logger.record_tabular('Epoch', 0)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
    else:
        offline_algorithm.train()

    logger.set_tabular_output(output_csv)
