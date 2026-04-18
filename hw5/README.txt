HW5: Exploration and Offline RL (RND, CQL, AWAC, IQL)

Scope
- This homework is a template for exploration-data collection and offline RL on gridworld tasks.
- Several algorithms are scaffolded with TODO markers; only framework wiring is fully present in many files.

Algorithms intended in this homework
- Random agent exploration
- Random Network Distillation (RND)
- Conservative Q-Learning (CQL)
- Advantage-Weighted Actor Critic (AWAC)
- Implicit Q-Learning (IQL)

Key code paths
- cs285/agents/random_agent.py
- cs285/agents/rnd_agent.py
- cs285/agents/dqn_agent.py
- cs285/agents/cql_agent.py
- cs285/agents/awac_agent.py
- cs285/agents/iql_agent.py
- cs285/scripts/run_hw5_explore.py
- cs285/scripts/run_hw5_offline.py
- cs285/scripts/run_hw5_finetune.py

What is currently implemented vs scaffolded

Implemented infrastructure
- Full explore/offline/finetune training script structure and logging flow.
- Env config plumbing for `rnd`, `cql`, `awac`, `iql`.
- Dataset save/load workflow using replay buffers.

Still scaffolded (TODO)
- Core algorithm updates in many agent files remain incomplete (e.g., `...` placeholders and `NotImplementedError` in DQN/CQL/AWAC/IQL/RND/random agent).

Equations the template is built around

1) RND bonus
- Predictor trained to match fixed random target on next state features:

\[
\mathcal{E}_\phi(s') = \left\|\hat f_\phi(s') - f^*_\theta(s')\right\|_2
\]

\[
\mathcal{L}_{RND}(\phi)=\mathbb{E}_{(s,a,s')\sim\mathcal D}[\mathcal E_\phi(s')]
\]

- Intrinsic reward bonus is proportional to prediction error.

2) CQL regularization
- Q-learning loss plus conservative penalty:

\[
\mathcal{L}_{CQL}
= \mathcal{L}_{TD}
+ \alpha\,\mathbb{E}_{(s,a)\sim\mathcal D}
\left[\log\sum_{a'}\exp(Q(s,a')) - Q(s,a)\right]
\]

3) AWAC actor update

\[
\mathcal{L}_{\pi}^{AWAC}
= -\mathbb{E}_{(s,a)\sim\mathcal D}
\left[\log\pi_\theta(a\mid s)\exp\left(\frac{A(s,a)}{\lambda}\right)\right]
\]

4) IQL critic/value updates
- Expectile value loss:

\[
\mathcal{L}_V(\phi)=\mathbb{E}_{(s,a)\sim\mathcal D}
\left[L_2^\tau\big(Q_\theta(s,a)-V_\phi(s)\big)\right]
\]

- Q regression target:

\[
\mathcal{L}_Q(\theta)=\mathbb{E}_{(s,a,s')\sim\mathcal D}
\left(Q_\theta(s,a)-\left[r+\gamma V_\phi(s')\right]\right)^2
\]

Notes
- Because several agent implementations are incomplete in this repository snapshot, this homework currently serves as a partially implemented template rather than a fully runnable algorithm suite.
