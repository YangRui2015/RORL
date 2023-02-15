benchmark_config_dict = {
    'halfcheetah-default':{
        'policy_smooth_reg': 0.1,
        'q_smooth_reg': 0.0001,
        'q_smooth_tau': 0.2,
        'q_ood_eps': 0.00,
        'q_ood_reg': 0.0,
        'q_ood_uncertainty_reg': 0.0,
        'q_ood_uncertainty_reg_min': 0.0,
        'q_ood_uncertainty_decay': float(0)
    },

    'halfcheetah-random-v2':{
        'num_samples': 20,
        'policy_smooth_eps': 0.001,
        'q_smooth_eps': 0.001,
    },
    'halfcheetah-medium-v2':{
        'num_samples': 10,
        'policy_smooth_eps': 0.001,
        'q_smooth_eps': 0.001,
    },
    'halfcheetah-medium-expert-v2':{
        'num_samples': 10,
        'policy_smooth_eps': 0.001,
        'q_smooth_eps': 0.001,
    },
    'halfcheetah-medium-replay-v2':{
        'num_samples': 10,
        'policy_smooth_eps': 0.001,
        'q_smooth_eps': 0.001,
    },
    'halfcheetah-expert-v2':{
        'num_samples': 10,
        'policy_smooth_eps': 0.005,
        'q_smooth_eps': 0.005,
    },

    'hopper-default':{
        'num_samples': 20,
        'policy_smooth_eps': 0.005,
        'policy_smooth_reg': 0.1,
        'q_smooth_eps': 0.005,
        'q_smooth_reg': 0.0001,
        'q_smooth_tau': 0.2,
        'q_ood_eps': 0.01,
        'q_ood_reg': 0.5,
        'q_ood_uncertainty_decay': float(1e-6)
    },
    'hopper-random-v2':{
        'q_ood_uncertainty_reg': 1.0,
        'q_ood_uncertainty_reg_min': 0.5,
    },
    'hopper-medium-v2':{
        'q_ood_uncertainty_reg': 2.0,
        'q_ood_uncertainty_reg_min': 0.1,
    },
    'hopper-medium-expert-v2':{
        'q_ood_uncertainty_reg': 3.0,
        'q_ood_uncertainty_reg_min': 1.0,
    },
    'hopper-medium-replay-v2':{
        'q_ood_uncertainty_reg': 0.1,
        'q_ood_uncertainty_reg_min': 0.0,
    },
    'hopper-expert-v2':{
        'q_ood_uncertainty_reg': 4.0,
        'q_ood_uncertainty_reg_min': 1.0,
    },


    'walker2d-default':{
        'num_samples': 20,
        'policy_smooth_reg': 1.0,
        'q_smooth_reg': 0.0001,
        'q_smooth_tau': 0.2,
        'q_ood_eps': 0.01,
    },
    'walker2d-random-v2':{
        'policy_smooth_eps': 0.005,
        'q_smooth_eps': 0.005,
        'q_ood_reg': 0.5,
        'q_ood_uncertainty_reg': 5.0,
        'q_ood_uncertainty_reg_min': 0.5,
        'q_ood_uncertainty_decay': float(1e-5)
    },
    'walker2d-medium-v2':{
        'policy_smooth_eps': 0.01,
        'q_smooth_eps': 0.01,
        'q_ood_reg': 0.1,
        'q_ood_uncertainty_reg': 1.0,
        'q_ood_uncertainty_reg_min': 0.1,
        'q_ood_uncertainty_decay': float(5e-7)
    },
    'walker2d-medium-expert-v2':{
        'policy_smooth_eps': 0.01,
        'q_smooth_eps': 0.01,
        'q_ood_reg': 0.1,
        'q_ood_uncertainty_reg': 0.1,
        'q_ood_uncertainty_reg_min': 0.1,
        'q_ood_uncertainty_decay': float(0)
    },
    'walker2d-medium-replay-v2':{
        'policy_smooth_eps': 0.01,
        'q_smooth_eps': 0.01,
        'q_ood_reg': 0.1,
        'q_ood_uncertainty_reg': 0.1,
        'q_ood_uncertainty_reg_min': 0.1,
        'q_ood_uncertainty_decay': float(0)
    },
    'walker2d-expert-v2':{
        'policy_smooth_eps': 0.005,
        'q_smooth_eps': 0.005,
        'q_ood_reg': 0.5,
        'q_ood_uncertainty_reg': 1.0,
        'q_ood_uncertainty_reg_min': 0.5,
        'q_ood_uncertainty_decay': float(1e-6)
    }
}

adversarial_config_dict = {
    'halfcheetah-medium-v2':{
        'num_samples': 20,
        'policy_smooth_eps': 0.05,
        'policy_smooth_reg': 1.0,
        'q_smooth_eps': 0.03,
        'q_smooth_reg': 0.0001,
        'q_smooth_tau': 0.2,
        'q_ood_eps': 0.00,
        'q_ood_reg': 0.0,
        'q_ood_uncertainty_reg': 0.0,
        'q_ood_uncertainty_reg_min': 0.0,
        'q_ood_uncertainty_decay': float(0)
    },
    'walker2d-medium-v2':{
        'num_samples': 20,
        'policy_smooth_eps': 0.07,
        'policy_smooth_reg': 0.5,
        'q_smooth_eps': 0.03,
        'q_smooth_reg': 0.0001,
        'q_smooth_tau': 0.2,
        'q_ood_eps': 0.03,
        'q_ood_reg': 0.5,
        'q_ood_uncertainty_reg': 1.0,
        'q_ood_uncertainty_reg_min': 0.1,
        'q_ood_uncertainty_decay': float(1e-6)
    },
    'hopper-medium-v2':{
        'num_samples': 20,
        'policy_smooth_eps': 0.005,
        'policy_smooth_reg': 0.1,
        'q_smooth_eps': 0.005,
        'q_smooth_reg': 0.0001,
        'q_smooth_tau': 0.2,
        'q_ood_eps': 0.02,
        'q_ood_reg': 0.5,
        'q_ood_uncertainty_reg': 2.0,
        'q_ood_uncertainty_reg_min': 0.1,
        'q_ood_uncertainty_decay': float(1e-6)
    }

}

def get_rorl_config(env_name, exp_type='benchmark'):
    configs = adversarial_config_dict if exp_type == 'attack' else benchmark_config_dict
    assert env_name in configs.keys()
    if exp_type != 'attack':
        confs = configs[env_name]
        default_confs = configs[env_name.split('-')[0]+'-default']
        default_confs.update(confs)
    else:
        default_confs = configs[env_name]
    return default_confs
