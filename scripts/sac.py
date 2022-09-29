from email.mime import image
from urllib.request import FancyURLopener
from experiment_utils.launch_experiment import launch_experiment
from experiment_configs.configs.q_learning.sac_config import get_config
from experiment_configs.algorithms.offline import get_offline_algorithm
import torch
import numpy as np
import lifelong_rl.torch.pytorch_util as ptu
import argparse
from lifelong_rl.core import logger
from scripts.get_config import get_rorl_config


def main(args):
    # Default parameters
    variant = dict(
        algorithm='',
        collector_type='step',
        env_name='hopper-random-v2',
        env_kwargs=dict(),
        replay_buffer_size=int(2e6),
        reward_mean=False,  # added for easy config checking
        reward_std=-1.0,  # added for easy config checking
        policy_kwargs=dict(
            layer_size=256,
            num_q_layers=3,
            num_p_layers=3,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=3e-4,
            qf_lr=3e-4,
            use_automatic_entropy_tuning=True,
            policy_eval_start=0,
            num_qs=10,
            target_update_period=1,
            max_q_backup=False,
            deterministic_backup=False,
            eta=-1.0,
        ),
        offline_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            max_path_length=1000, 
            batch_size=256,
            save_snapshot_freq=500,
        ),
    )

    experiment_kwargs = dict(
        exp_postfix='',
        use_gpu=True if torch.cuda.is_available() else False,
        log_to_tensorboard=True,
        base_log_dir=args.base_log_dir,
    )

    # Variant
    variant['env_name'] = args.env_name
    variant['seed'] = args.seed
    variant['load_path'] = args.load_path
    variant['load_Qs'] = args.load_Qs
    variant['eval_no_training'] = args.eval_no_training
    variant['eval_attack'] = args.eval_attack
    variant['eval_attack_eps'] = args.eval_attack_eps
    variant['eval_attack_mode'] = args.eval_attack_mode


    variant['offline_kwargs']['num_epochs'] = args.epoch

    # SAC-N
    variant['trainer_kwargs']['policy_lr'] = args.plr
    variant['trainer_kwargs']['qf_lr'] = args.qlr

    variant['trainer_kwargs']['num_qs'] = args.num_qs
    variant['trainer_kwargs']['max_q_backup'] = args.max_q_backup
    variant['trainer_kwargs']['deterministic_backup'] = args.deterministic_backup

    variant['reward_mean'] = args.reward_mean
    variant['reward_std'] = args.reward_std
    
    # EDAC
    variant['trainer_kwargs']['eta'] = args.eta

    # smooth
    if args.load_config_type != '':
        rorl_config = get_rorl_config(args.env_name, args.load_config_type)
        keys = ['num_samples', 'policy_smooth_eps', 'policy_smooth_reg', 'q_smooth_eps',
                'q_smooth_reg', 'q_smooth_tau', 'q_ood_eps', 'q_ood_reg', 'q_ood_uncertainty_reg',
                'q_ood_uncertainty_reg_min', 'q_ood_uncertainty_decay']
        for key in keys:
            variant['trainer_kwargs'][key] = rorl_config[key]
    else:
        variant['trainer_kwargs']['num_samples'] = args.num_samples
        variant['trainer_kwargs']['policy_smooth_eps'] = args.policy_smooth_eps
        variant['trainer_kwargs']['policy_smooth_reg'] = args.policy_smooth_reg
        variant['trainer_kwargs']['q_smooth_eps'] = args.q_smooth_eps
        variant['trainer_kwargs']['q_smooth_reg'] = args.q_smooth_reg
        variant['trainer_kwargs']['q_smooth_tau'] = args.q_smooth_tau
        variant['trainer_kwargs']['q_ood_eps'] = args.q_ood_eps
        variant['trainer_kwargs']['q_ood_reg'] = args.q_ood_reg
        variant['trainer_kwargs']['q_ood_uncertainty_reg'] = args.q_ood_uncertainty_reg
        variant['trainer_kwargs']['q_ood_uncertainty_reg_min'] = args.q_ood_uncertainty_reg_min
        variant['trainer_kwargs']['q_ood_uncertainty_decay'] = args.q_ood_uncertainty_decay

    # experiment name
    experiment_kwargs['exp_postfix'] = ''
    
    exp_postfix = args.exp_prefix + '_{}'.format(args.num_qs)
    
    # exp_postfix += '_plr{:.4f}_qlr{:.4f}'.format(args.plr, args.qlr)
    if variant['trainer_kwargs']['max_q_backup']:
        exp_postfix += '_maxq'
    if variant['trainer_kwargs']['deterministic_backup']:
        exp_postfix += '_detq'
    if args.eta > 0:
        exp_postfix += '_eta{:.2f}'.format(args.eta)
    if args.reward_mean:
        exp_postfix += '_mean'
    if args.reward_std > 0:
        exp_postfix += '_std'

    experiment_kwargs['exp_postfix'] = exp_postfix

    experiment_kwargs['data_args'] = {
        'reward_mean': args.reward_mean,
        'reward_std': args.reward_std,
        'shift_reward_minzero': args.shift_reward_minzero,
        'reward_norm': args.reward_norm,
    }

    ####### normalize input
    variant['norm_input'] = args.norm_input

    # Launch experiment
    launch_experiment(variant=variant, 
                      get_config=get_config,
                      get_offline_algorithm=get_offline_algorithm,
                      **experiment_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Variant
    parser.add_argument('-e',
                        '--env_name',
                        default='halfcheetah-random-v2',
                        type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--exp_prefix', default='RORL', type=str)
    # Misc arguments
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--log_to_tensorboard', action='store_true')
    parser.add_argument('--base_log_dir', default='results', type=str)
    parser.add_argument('--norm_input', action='store_true')
    parser.add_argument("--epoch", default=3000, type=int)
    parser.add_argument('--load_path', default='', type=str)
    parser.add_argument('--load_Qs', default='', type=str, help='Only load Qs')

    # Evaluation 
    parser.add_argument('--eval_no_training', action='store_true')
    parser.add_argument('--eval_attack', action='store_true')
    parser.add_argument('--eval_attack_eps', default=0.01, type=float)
    parser.add_argument('--eval_attack_mode', default='random', type=str)
    # SAC
    parser.add_argument("--plr",
                        default=3e-4,
                        type=float,
                        help='policy learning rate')
    parser.add_argument("--qlr",
                        default=3e-4,
                        type=float,
                        help='Q learning rate')
    parser.add_argument("--num_qs",
                        default=10,
                        type=int,
                        help='number of Q-functions to be used')
    parser.add_argument('--max_q_backup',
                        action='store_true',
                        help='use max q backup')
    parser.add_argument('--deterministic_backup',
                        action='store_true',
                        help='use deterministic backup')
    parser.add_argument('--eta',
                        default=-1.0,
                        type=float,
                        help='eta for diversifying Q-ensemble. < 0 for SAC-N.')
    
    # reward preprocessing
    parser.add_argument("--reward_mean",
                        action='store_true',
                        help='normalize rewards to 0 mean')
    parser.add_argument("--reward_std",
                        action='store_true',
                        help='normalize rewards to 1 std')
    parser.add_argument("--reward_norm",
                        action='store_true',
                        help='normalize rewards to 0 mean and 1 std')
    parser.add_argument("--shift_reward_minzero",
                        action='store_true',
                        help='normalize rewards to > 0')
    
    # smooth
    parser.add_argument('--num_samples', default=20, type=int)
    parser.add_argument('--policy_smooth_eps', default=0.0, type=float) 
    parser.add_argument('--policy_smooth_reg', default=1, type=float)
    parser.add_argument('--q_smooth_eps', default=0.0, type=float) 
    parser.add_argument('--q_smooth_reg', default=0.005, type=float) 
    parser.add_argument('--q_smooth_tau', default=0.2, type=float)
    parser.add_argument('--q_ood_eps', default=0.0, type=float)
    parser.add_argument('--q_ood_reg', default=0, type=float)
    parser.add_argument('--q_ood_uncertainty_reg', default=0, type=float)
    parser.add_argument('--q_ood_uncertainty_reg_min', default=0, type=float)
    parser.add_argument('--q_ood_uncertainty_decay', default=0, type=float)

    # load configs
    parser.add_argument('--load_config_type', default='', type=str)
    args = parser.parse_args()

    main(args)