# HW3: DQN and Soft Actor-Critic

## Scope
This homework implements off-policy value-based and actor-critic methods.

Implemented methods:
- Deep Q-Network (DQN)
- Optional Double DQN targets
- Soft Actor-Critic (SAC) with configurable actor gradients
- Multi-critic target backup strategies (`mean`, `min`, `doubleq`, `redq` scaffold)
- Optional entropy bonus in SAC

Key code paths:
- `cs285/agents/dqn_agent.py`
- `cs285/agents/soft_actor_critic.py`
- `cs285/scripts/run_hw3_dqn.py`
- `cs285/scripts/run_hw3_sac.py`
- `cs285/env_configs/dqn_basic_config.py`
- `cs285/env_configs/sac_config.py`

## RL Methods and Equations Used

### 1) DQN critic target
In `DQNAgent.update_critic`, Bellman targets are:

\[
y_t = r_t + \gamma (1-d_t)\, \max_{a'} Q_{\bar\theta}(s_{t+1},a')
\]

Critic loss:

\[
\mathcal{L}_{Q}(\theta)=\frac{1}{N}\sum_t\left(Q_\theta(s_t,a_t)-y_t\right)^2
\]

If `use_double_q=True`, action selection uses online critic and evaluation uses target critic:

\[
a^* = \arg\max_{a'}Q_\theta(s_{t+1},a'),\quad
y_t = r_t + \gamma(1-d_t)Q_{\bar\theta}(s_{t+1},a^*)
\]

Target network is periodically synchronized.

### 2) SAC critic update
In `SoftActorCritic.update_critic`, sampled next actions come from the actor and targets are built from target critics:

\[
y_t = r_t + \gamma(1-d_t)\,\tilde Q(s_{t+1},a'_{t+1})
\]

where \(\tilde Q\) is produced by backup strategy (`mean/min/doubleq/...`).

When entropy backup is enabled:

\[
y_t = r_t + \gamma(1-d_t)\left(\tilde Q(s_{t+1},a'_{t+1}) + \alpha\,\mathcal{H}[\pi(\cdot\mid s_{t+1})]\right)
\]

and with sampled-action entropy estimator:

\[
\mathcal{H}[\pi(\cdot\mid s)] \approx -\log \pi(a\mid s)
\]

### 3) SAC actor updates
Two actor gradient modes exist.

Reparameterization mode:

\[
\mathcal{L}_{\pi} = -\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_\theta}[Q(s,a)]
\]

REINFORCE-style mode (Monte Carlo samples from policy):

\[
\mathcal{L}_{\pi}
= -\mathbb{E}\left[\log\pi_\theta(a\mid s)\,Q(s,a)\right]
\]

If entropy bonus is enabled for actor update:

\[
\mathcal{L}_{\pi} \leftarrow \mathcal{L}_{\pi} - \alpha\,\mathcal{H}[\pi(\cdot\mid s)]
\]

## Notes
- DQN script uses replay-buffer sampling with epsilon-greedy behavior.
- SAC script supports random warm-up steps before gradient updates.
- Configs expose most algorithmic switches directly.
