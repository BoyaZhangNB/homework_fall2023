# HW4: Model-Based RL, MPC, and MBPO

### Scope
- This homework implements model-based reinforcement learning with learned dynamics, model-predictive control (MPC), and optional MBPO-style coupling with SAC.

### Implemented methods
- Ensemble dynamics model learning on state deltas
- MPC with random shooting
- MPC with Cross-Entropy Method (CEM)
- Optional MBPO rollout generation using learned model + SAC actor

### Key code paths
- cs285/agents/model_based_agent.py
- cs285/scripts/run_hw4.py
- cs285/env_configs/mpc_config.py
- cs285/env_configs/sac_config.py
- cs285/agents/soft_actor_critic.py

RL methods and equations used

1) Dynamics model training
- Each ensemble model predicts normalized next-state delta from normalized (state, action).

$\[
\Delta s_t = s_{t+1} - s_t
\]$

$\[\mathcal{L}_{dyn}(\phi_i) = \frac{1}{N}\sum_t\left\|f_{\phi_i}(\mathrm{norm}(s_t,a_t)) - \mathrm{norm}(\Delta s_t)\right\|_2^2 \]$

- Inference unnormalizes and reconstructs next state:

$\[
\hat s_{t+1} = s_t + \widehat{\Delta s_t}
\]$

2) MPC objective
- For each candidate action sequence $\(a_{t:t+H-1}\)$, roll out the learned model and score cumulative reward.

$\[
J(a_{t:t+H-1}) = \sum_{k=0}^{H-1} r(\hat s_{t+k+1}, a_{t+k})
\]$

- Random-shooting MPC returns first action from best sampled sequence.

3) CEM planner
- Iteratively select elites and refit a Gaussian over action sequences.

Elite updates in code:

$\[
\mu \leftarrow \alpha\,\mu_{elite} + (1-\alpha)\,\mu,
\quad
\sigma \leftarrow \alpha\,\sigma_{elite} + (1-\alpha)\,\sigma
\]$

- New action sequences are sampled from $\(\mathcal N(\mu,\sigma^2)\)$, clipped to action bounds.

4) MBPO data generation
- In `collect_mbpo_rollout`, SAC chooses actions while the learned dynamics predicts next states, then synthetic transitions are inserted into SAC replay.

Notes
- `run_hw4.py` first fits dynamics models, then (optionally) trains SAC.
- Real-environment transitions are always collected each iteration; model rollouts are additional synthetic data for MBPO mode.
