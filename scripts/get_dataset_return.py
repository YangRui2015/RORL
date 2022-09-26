from experiment_utils.prepare_data import load_hdf5
from lifelong_rl.core.logging.logging_setup import setup_logger
from lifelong_rl.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from lifelong_rl.data_management.replay_buffers.mujoco_replay_buffer import MujocoReplayBuffer
from lifelong_rl.envs.env_processor import make_env
import numpy as np



envs = ['halfcheetah', 'walker2d', 'hopper']
types = ['expert', 'medium', 'medium-replay', 'medium-expert', 'random']
env_name_lis = [x + '-' + y + '-v2' for x in envs for y in types]
buffer_capacity = int(1e6)
res = []
#Environment setup
for env_name in env_name_lis:
    expl_env, env_infos = make_env(env_name)
    if env_infos['mujoco']:
        replay_buffer = MujocoReplayBuffer(buffer_capacity, expl_env)
    else:
        replay_buffer = EnvReplayBuffer(buffer_capacity, expl_env)
    eval_env, _ = make_env(env_name)

    load_hdf5(expl_env, replay_buffer, None)
    terminals = replay_buffer._terminals
    positive_indexs = np.where(terminals > 0)[0]
    pos_list = list(positive_indexs)
    N = terminals.shape[0]
    start = 0
    end = start + 1000
    episode = 0
    episode_reward = []
    while start < N:
        if not len(pos_list) or end < pos_list[0]:
            episode_reward.append(replay_buffer._rewards[start:end].sum())
            episode += 1
            start = end 
        else:
            episode_reward.append(replay_buffer._rewards[start:pos_list[0]].sum())
            episode += 1
            start = pos_list[0]
            pos_list.pop(0)
        end = start + 1000
    avg_epi_reward = replay_buffer._rewards.sum()/episode
    print('{} episode reward: {}, episode reward2: {}, episode reward std: {}'.format(
                                                        env_name, avg_epi_reward, np.mean(episode_reward), np.std(episode_reward)
                                                        ))
    res.append([env_name, avg_epi_reward, episode_reward])

print()
print('Summary:')
for x in res:
    env_name, avg_epi_reward, episode_reward = x
    print('{} episode reward: {}, episode reward2: {}, episode reward std: {}'.format(
                    env_name, avg_epi_reward, np.mean(episode_reward), np.std(episode_reward)
    ))
import pdb;pdb.set_trace()