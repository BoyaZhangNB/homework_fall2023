# HW1: Imitation Learning (Behavior Cloning + DAgger)

## Scope
This homework implements supervised imitation learning for continuous-control MuJoCo tasks.

Implemented methods:
- Behavior Cloning (BC)
- Dataset Aggregation (DAgger)

Key code paths:
- `cs285/policies/MLP_policy.py`
- `cs285/infrastructure/utils.py`
- `cs285/scripts/run_hw1.py`

## RL Methods and Equations Used

### 1) Behavior Cloning (supervised policy regression)
The policy is an MLP that predicts continuous actions from observations.

Training objective implemented in `MLPPolicySL.update`:

\[
\mathcal{L}_{BC}(\theta)
= \frac{1}{N}\sum_{i=1}^N \left\|\pi_\theta(s_i)-a_i^{expert}\right\|_2^2
\]

This is standard MSE regression onto expert actions.

### 2) DAgger data aggregation loop
In later iterations (`itr > 0`), trajectories are sampled from the current learner policy, then relabeled with expert actions, and appended to replay data.

Conceptually:

\[
\mathcal{D} \leftarrow \mathcal{D} \cup \{(s, \pi_E(s))\;|\; s\sim d_{\pi_\theta}\}
\]

The policy is repeatedly retrained on aggregated data using the same BC loss.

## Training Pipeline
- Iteration 0:
	- load expert dataset from `cs285/expert_data/*.pkl`
	- train policy via BC objective
- Iterations 1..T (when `--do_dagger`):
	- roll out current learner
	- relabel observed states with expert actions
	- append to replay buffer
	- sample minibatches and train by supervised MSE

## Run

### Behavior Cloning
```bash
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v4 \
	--exp_name bc_ant \
	--n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

### DAgger
```bash
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v4 \
	--exp_name dagger_ant \
	--n_iter 10 \
	--do_dagger \
	--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

## Notes
- `sample_trajectory` in `cs285/infrastructure/utils.py` uses the old Gym step API.
- Logging includes train/eval returns and optional rollout videos.


