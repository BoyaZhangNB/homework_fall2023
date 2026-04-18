# HW2: Policy Gradient and Advantage Estimation

## Scope
This homework implements on-policy policy-gradient training for continuous and discrete control.

Implemented methods:
- Vanilla policy gradient
- Reward-to-go returns
- Neural baseline (value critic)
- Generalized Advantage Estimation (GAE)
- Advantage normalization

Key code paths:
- `cs285/agents/pg_agent.py`
- `cs285/networks/policies.py`
- `cs285/networks/critics.py`
- `cs285/scripts/run_hw2.py`

## RL Methods and Equations Used

### 1) Monte Carlo Q-value targets
Two return estimators are supported in `PGAgent._calculate_q_vals`:

Trajectory return for each timestep:

\[
Q_t = \sum_{t'=0}^{T} \gamma^{t'} r_{t'}
\]

Reward-to-go:

\[
Q_t = \sum_{t'=t}^{T} \gamma^{(t'-t)} r_{t'}
\]

### 2) Advantage estimation
If baseline is disabled:

\[
A_t = Q_t
\]

If baseline is enabled without GAE:

\[
A_t = Q_t - V_\phi(s_t)
\]

With GAE (`gae_lambda` provided), temporal-difference residuals are recursively accumulated:

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

\[
A_t = \delta_t + \gamma\lambda A_{t+1}
\]

Advantages can then be normalized:

\[
\hat A_t = \frac{A_t-\mu_A}{\sigma_A+\epsilon}
\]

### 3) Policy objective
`MLPPolicyPG.update` implements the policy-gradient loss:

\[
\mathcal{L}_{\pi}(\theta)
= -\frac{1}{N}\sum_{t=1}^{N} \log \pi_\theta(a_t\mid s_t)\,\hat A_t
\]

For continuous actions, log-probabilities are summed across action dimensions.

### 4) Baseline critic objective
`ValueCritic.update` trains a value regressor with MSE:

\[
\mathcal{L}_{V}(\phi)
= \frac{1}{N}\sum_{t=1}^{N}\left(V_\phi(s_t)-Q_t\right)^2
\]

## Notes
- This implementation is explicitly on-policy and uses trajectory batches per iteration.
- The critic is optional and can be updated for multiple gradient steps per policy update.
