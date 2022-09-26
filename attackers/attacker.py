import numpy as np
import torch
import lifelong_rl.torch.pytorch_util as ptu
from torch.distributions import kl_divergence


def get_policy_kl(policy, observation, noised_obs): 
    _, policy_mean, policy_log_std, _, *_ = policy.stochastic_policy(observation)
    _, noised_policy_mean, noised_policy_log_std, _, *_ = policy.stochastic_policy(noised_obs)
    action_dist = torch.distributions.Normal(policy_mean, policy_log_std.exp())
    noised_action_dist = torch.distributions.Normal(noised_policy_mean, noised_policy_log_std.exp())

    kl_loss = kl_divergence(action_dist, noised_action_dist).sum(axis=-1) + kl_divergence(noised_action_dist, action_dist).sum(axis=-1)
    return kl_loss

def optimize_para(para, observation, loss_fun, update_times, step_size, eps, std):
    for i in range(update_times):
        para = torch.nn.Parameter(para.clone(), requires_grad=True)
        optimizer = torch.optim.Adam([para], lr=step_size * eps) 
        loss = loss_fun(observation, para)
        # optimize noised obs
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        # para = torch.clamp(para, -eps, eps).detach()
        para = torch.maximum(torch.minimum(para, eps * std), -eps * std).detach()
    return para 


class Evaluation_Attacker:
    def __init__(self, policy, q_fun, eps, obs_dim, action_dim, obs_std=None, attack_mode='random',  num_samples=50):
        self.policy = policy
        self.q_fun = q_fun
        self.eps = eps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.attack_mode = attack_mode
        self.num_samples = num_samples
        self.obs_std = ptu.from_numpy(obs_std) if obs_std is not None else torch.ones(1, self.obs_dim, device=ptu.device)

    def sample_random(self, size):
        return 2 * self.eps * self.obs_std * (torch.rand(size, self.obs_dim, device=ptu.device) - 0.5) 

    def noise_action_diff(self, observation, M):
        observation = observation.reshape(M, self.obs_dim)
        update_times, step_size = 10, 0.1 # for first order and mixed order
        size = self.num_samples # for zero order 
        def _loss_action(observation, para):
            noised_obs = observation + para
            return - get_policy_kl(self.policy, observation, noised_obs)

        if self.attack_mode == 'action_diff':
            delta_s = self.sample_random(size).reshape(1, size, self.obs_dim).repeat(M, 1, 1).reshape(-1, self.obs_dim)
            tmp_obs = observation.reshape(-1, 1, self.obs_dim).repeat(1, size, 1).reshape(-1, self.obs_dim)
            with torch.no_grad():
                kl_loss = _loss_action(tmp_obs, delta_s)
                max_id = torch.argmin(kl_loss.reshape(M, size), axis=1)
            noise_obs_final = ptu.get_numpy(delta_s.reshape(M, size, self.obs_dim)[np.arange(M), max_id])
        elif self.attack_mode == 'action_diff_first_order':
            para = self.sample_random(1).repeat(M, 1).reshape(-1, self.obs_dim)
            para = optimize_para(para, observation, _loss_action, update_times, step_size, self.eps, self.obs_std)
            noise_obs_final = ptu.get_numpy(para)
        elif self.attack_mode == 'action_diff_mixed_order':
            size = 20
            tmp_obs = observation.reshape(-1, 1, self.obs_dim).repeat(1, size, 1).reshape(-1, self.obs_dim)
            para = self.sample_random(size).reshape(1, size, self.obs_dim).repeat(M, 1, 1).reshape(-1, self.obs_dim)
            para = optimize_para(para, tmp_obs, _loss_action, update_times, step_size, self.eps, self.obs_std)
            with torch.no_grad():
                kl_loss = _loss_action(tmp_obs, para)
                max_id = torch.argmin(kl_loss.reshape(M, size), axis=1)
            noise_obs_final = ptu.get_numpy(para).reshape(M, size, self.obs_dim)[np.arange(M),max_id]

        return ptu.get_numpy(observation) + noise_obs_final

    def noise_min_Q(self, observation, M):
        observation = observation.reshape(M, self.obs_dim)
        update_times, step_size = 10, 0.1 # for first order and mixed order
        size = self.num_samples  # for zero order
        weight_std = 10

        def _loss_Q(observation, para):
            noised_obs = observation + para
            pred_actions, _, _, _, *_ = self.policy.stochastic_policy(noised_obs,  deterministic=True)
            return self.q_fun(observation, pred_actions) 

        def _loss_Q_std(observation, para):
            Q_loss = _loss_Q(observation, para)
            Q_std = Q_loss.std(axis=0).reshape(1, -1, 1)
            return  - weight_std * Q_std # or combine std with Q_loss

        loss_fun = _loss_Q_std if 'std' in self.attack_mode else _loss_Q
        if self.attack_mode == 'min_Q' or self.attack_mode == 'min_Q_std':
            delta_s = self.sample_random(size+1)
            delta_s[-1,:] = torch.zeros((1, self.obs_dim), device=ptu.device)
            delta_s = delta_s.reshape(1, size+1, self.obs_dim).repeat(M, 1, 1).reshape(-1, self.obs_dim)
            tmp_obs = observation.reshape(-1, 1, self.obs_dim).repeat(1, size+1, 1).reshape(-1, self.obs_dim)
            noised_qs_pred = loss_fun(tmp_obs, delta_s).mean(axis=0).reshape(-1, size+1)
            min_id = torch.argmin(noised_qs_pred, axis=1)
            # print(ptu.get_numpy(noised_qs_pred.std(axis=1)))
            noise_obs_final = ptu.get_numpy(delta_s).reshape(M, size+1, self.obs_dim)[np.arange(M), min_id]
        elif 'first_order' in self.attack_mode:
            para = self.sample_random(1).repeat(M, 1).reshape(-1, self.obs_dim)
            para = optimize_para(para, observation, loss_fun, update_times, step_size, self.eps, self.obs_std)
            noise_obs_final = ptu.get_numpy(para)
        elif 'mixed_order' in self.attack_mode:
            size = 20
            tmp_obs = observation.reshape(-1, 1, self.obs_dim).repeat(1, size, 1).reshape(-1,self.obs_dim)
            para = self.sample_random(size).reshape(1, size, self.obs_dim).repeat(M, 1, 1).reshape(-1, self.obs_dim)
            para = optimize_para(para, tmp_obs, loss_fun, update_times, step_size, self.eps, self.obs_std)
            with torch.no_grad():
                Q_loss = loss_fun(tmp_obs, para)
                # print(Q_loss)
                min_id = torch.argmin(Q_loss.mean(axis=0).reshape(-1, size), axis=1)
            noise_obs_final = ptu.get_numpy(para).reshape(M, size, self.obs_dim)[np.arange(M), min_id]
        else:
            raise NotImplementedError
        return ptu.get_numpy(observation) + noise_obs_final
    
    def attack_obs(self, observation):
        M = observation.shape[0] if len(observation.shape) == 2 else 1
        observation = ptu.from_numpy(observation)
        if self.attack_mode == 'random':
            delta_s = self.sample_random(M)
            noised_observation = observation.reshape(M, self.obs_dim) + delta_s
            noised_observation = ptu.get_numpy(noised_observation)
        elif 'action_diff' in self.attack_mode:
            noised_observation = self.noise_action_diff(observation, M)

        elif 'min_Q' in self.attack_mode:
            noised_observation = self.noise_min_Q(observation, M)
        else:
            raise NotImplementedError
        
        return noised_observation

        