# RORL: Robust Offline Reinforcement Learning via Conservative Smoothing

RORL trades off robustness and conservatism for offline RL via conservative smoothing and OOD underestimation.

The code is based on EDAC's code and rlkit.
## Requirements For RORL

To install the required dependencies:

1. Install the MuJoCo 2.0 engine, which can be downloaded from [here](https://mujoco.org/download).

2. Install Python packages listed in `requirements.txt` using `pip`. 

3. Manually download and install `d4rl` package from [here](https://github.com/rail-berkeley/d4rl). You should remove lines including `dm_control` in `setup.py`.

Here is an example of how to install all the dependencies on Ubuntu:
  
```bash
conda create -n edac python=3.7
conda activate edac
pip install --no-cache-dir -r requirements.txt

cd .
git clone https://github.com/rail-berkeley/d4rl.git

cd d4rl
# Remove lines including 'dm_control' in setup.py
pip install -e .
```

## Reproducing the results

### Training 
#### RORL Experiments for MuJoCo Gym:

```bash
python -m scripts.sac --env_name [ENVIRONMENT] --num_qs 10 --norm_input --load_config_type 'benchmark'
```
To reproduce results of adersarial experiments, you can simply replace 'benchmark' with 'attack'.

#### SAC-10 results for MuJoCo Gym:

```bash
python -m scripts.sac --env_name [ENVIRONMENT] --num_qs 10 --norm_input
```

#### EDAC results for MuJoCo Gym:

```bash
python -m scripts.sac --env_name [ENVIRONMENT] --num_qs 10 --eta 1 --norm_input
```

### Evaluation
#### To evaluate trained agents in clean environments, run
```bash
python -m scripts.sac --env_name [ENVIRONMENT] --num_qs 10 --norm_input --eval_no_training
```

#### To evaluate trained agents in adversarial environments, run
```bash
python -m scripts.sac --env_name [ENVIRONMENT] --num_qs 10 --norm_input --eval_no_training --load_path [model path] --eval_attack  --eval_attack_mode [mode]    --eval_attack_eps [epsilon] 
```
'mode': 'random, action_diff, min_Q, action_diff_mixed_order, min_Q_mixed_order'.

'model path':  e.g., ~/offline_itr_3000.pt.