# 4. Reinforcement Learning Background — Q-Learning and PPO

Theoretical background for the two learning agents that will be compared against the
dynamic-programming (DP) ground truth on the market-making MDP. The notation matches
the DP model: state is inventory $q$ with time index $t$, action is the quote pair
$a = (\delta_a, \delta_b)$, reward is the mark-to-market PnL $R(q, a)$, and the
horizon is finite ($T$ steps) with terminal penalty $-\alpha q^2$.

---

## 4.1. The MDP and the objects every method estimates

A finite-horizon Markov decision process is the tuple
$(\mathcal{S}, \mathcal{A}, P, R, T)$:

- $\mathcal{S}$ — states. Here a state is $(q, t)$: inventory and time-to-go.
- $\mathcal{A}$ — actions $a = (\delta_a, \delta_b)$.
- $P(s' \mid s, a)$ — transition kernel (the three-branch inventory transition).
- $R(s, a)$ — expected one-step reward.
- $T$ — horizon; the terminal value is $V_T(q) = -\alpha q^2$.

A **policy** $\pi(a \mid s)$ maps states to action distributions. The **return** from
time $t$ is the sum of future rewards,

$$G_t = \sum_{k=t}^{T-1} R_k + V_T(q_T),$$

(no discount needed for a finite horizon; a factor $\gamma \in (0,1]$ may be inserted).

Two value functions summarize a policy:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[G_t \mid s_t = s\right], \qquad
Q^\pi(s, a) = \mathbb{E}_\pi\!\left[G_t \mid s_t = s,\, a_t = a\right].$$

The **optimal** functions satisfy the Bellman optimality equations

$$Q^*(s, a) = R(s, a) + \sum_{s'} P(s' \mid s, a)\, \max_{a'} Q^*(s', a'),$$
$$V^*(s) = \max_a Q^*(s, a), \qquad \pi^*(s) = \arg\max_a Q^*(s, a).$$

**DP solves these exactly** by backward induction — it knows $P$ and $R$ in closed
form and sweeps $V_T \to V_{T-1} \to \dots \to V_0$. Q-learning and PPO estimate the
**same** $Q^*$ / $\pi^*$ **without** assuming knowledge of $P$ and $R$: they learn
from sampled transitions. This is the entire difference, and it is why DP is the
ground truth — it is the exact answer the learners are trying to approximate.

---

## 4.2. Q-Learning

### 4.2.1 Idea

Q-learning is a **model-free, off-policy, value-based** method. It estimates $Q^*$
directly from sampled transitions $(s, a, r, s')$, never forming an explicit model of
$P$ or $R$. It is the sampling counterpart of value iteration.

### 4.2.2 The update — why it converges to the optimal values

**The whole idea in one sentence.** $Q(s, a)$ is meant to mean "the total reward I
will get if I take action $a$ in state $s$, then play optimally afterward." Q-learning
makes that sentence *true* by repeatedly forcing each $Q$ value to agree with the $Q$
values that come right after it.

**The TD target is what $Q(s,a)$ should equal.** Take action $a$, collect reward $r$,
land in $s'$. From $s'$ you play optimally, so the best you can get from there is
$\max_{a'} Q(s', a')$. Therefore the correct value is

$$Q(s, a) = \underbrace{r}_{\text{reward now}} + \underbrace{\max_{a'} Q(s', a')}_{\text{best from next state}}.$$

The right-hand side is the **TD target**: it is just $Q(s,a)$ rewritten in terms of
what happens *one step later*. If every entry in the table satisfies this equation
simultaneously, the table is correct — it is $Q^*$. Being optimal *is* being
self-consistent one step at a time.

**The update shrinks the disagreement.** Usually $Q(s,a)$ does **not** equal the
target yet; the gap between them is the **TD error**. The update nudges $Q$ a little
toward the target:

$$\boxed{\;Q(s, a) \leftarrow Q(s, a) + \eta \Big[\underbrace{r + \max_{a'} Q(s', a')}_{\text{TD target}} - Q(s, a)\Big].\;}$$

**Why shrinking these gaps everywhere lands on $Q^*$.** Two things work together:

*1. Truth flows backward from the rewards.* The target contains $r$ — a **real**
reward from the environment, not a guess. So every update injects a bit of
ground-truth information and mixes it into $Q(s,a)$. The $\max_{a'} Q(s',a')$ part is
still an estimate, but it too absorbed real reward the last time it was updated. So
real information seeps backward one step at a time: rewards inform the states just
before them, which inform the states before *those*, and so on. Picture the rewards as
light sources and the early states as dark; each update lets a state copy a bit of
what its neighbor learned plus the real reward in between, and the light spreads
outward from the rewards until the whole table is lit correctly.

*2. The flow can only stop at $Q^*$.* When does the table stop changing? Only when
every TD error is zero — when $Q(s,a) = r + \max_{a'} Q(s',a')$ everywhere. But that is
exactly the self-consistency condition that *defines* $Q^*$. There is nowhere else to
come to rest, and the self-consistent table is unique, so the updates cannot get stuck
on a wrong one.

Together: real reward keeps flooding in and spreading backward (the table can't stay
wrong), and the only configuration where everything stops moving is $Q^*$ (so where it
stops is the right answer).

**Why a small step $\eta$ rather than setting $Q$ equal to the target.** The
environment is random — the same $(s,a)$ can yield different $r$ and $s'$ on different
visits, so a single target is a noisy sample of the true average. Slamming $Q$ to each
noisy target would make it jitter forever. The learning rate $\eta$ makes $Q$ a
**running average** of all targets seen, averaging the noise away so it settles on the
true expected return instead of bouncing around.

**Each piece, summarized.**

- The bracket is the **TD error** — the gap between the target and the current
  estimate. It is zero in expectation exactly when $Q = Q^*$, which is why driving it
  to zero finds $Q^*$.
- $\eta$ is the step that **averages out sampling noise** in $r$ and $s'$.
- $\max_{a'}$ assumes optimal play from the next state — and bootstraps from the
  greedy action regardless of what exploration actually did next, which is what makes
  Q-learning **off-policy**.
- The target uses the current estimate $Q(s', \cdot)$ rather than a full
  Monte-Carlo return — **bootstrapping**: biased early (the estimate is wrong) but far
  lower variance than waiting for the episode to end, and the bias vanishes as
  $Q \to Q^*$.

**Relation to DP.** The model-based version of this idea computes the target's
expectation exactly (using known $P$ and $R$) instead of sampling it — that is value
iteration, and its finite-horizon sweep is exactly DP's backward induction. Q-learning
is the same fixed-point logic with the expectation **sampled** rather than computed.

**Finite-horizon form.** The table is indexed by $(q, t)$ and the target steps the
time index forward:

$$Q(q, t, a) \leftarrow Q(q, t, a) + \eta\Big[r + \max_{a'} Q(q', t{+}1, a') - Q(q, t, a)\Big],$$

with the bootstrap at the last step replaced by the known terminal value,
$\max_{a'} Q(q', T, \cdot) \to V_T(q') = -\alpha q'^2$. This mirrors DP's backward
induction, except the next-state value is sampled rather than computed from $P$.

### 4.2.3 Exploration

Because the update bootstraps from $\max_{a'} Q$, the agent must still **visit** all
relevant $(s, a)$ to learn their values. The standard scheme is
**$\varepsilon$-greedy**:

$$a = \begin{cases}\arg\max_{a'} Q(s, a') & \text{w.p. } 1 - \varepsilon,\\
\text{uniform random action} & \text{w.p. } \varepsilon,\end{cases}$$

typically with $\varepsilon$ annealed from high to low over training.

### 4.2.4 Convergence and relation to DP

For a **finite, tabular** MDP with state $(q, t)$, tabular Q-learning converges to
$Q^*$ with probability 1 under standard conditions (every $(s,a)$ visited infinitely
often; learning-rate schedule satisfying $\sum_t \eta_t = \infty$,
$\sum_t \eta_t^2 < \infty$).

**Consequence for the comparison:** because the table over $(q, t)$ is expressive
enough to represent $\pi^*$ exactly, a correct tabular Q-learner *must* reproduce the
DP policy in the limit. Any remaining gap is attributable to **finite samples and
exploration**, not to representational limits. This makes Q-learning the cleanest
sanity check on the whole pipeline: if it does not approach DP, something is wrong in
the environment or the training loop, not in the theory.

> **Caveat — state must include time.** A stationary Q-table over $q$ alone *cannot*
> represent the optimal policy, because $\pi^*$ genuinely depends on $t$ (terminal
> penalty). Such an agent converges to a time-averaged compromise and will look
> worse than DP for reasons unrelated to the algorithm. The state must be $(q, t)$.

### 4.2.5 Strengths and limits

- **Strengths:** simple, sample-reuse via off-policy updates, exact in the tabular
  limit, no policy-gradient variance.
- **Limits:** the table grows with $|\mathcal{S}| \times |\mathcal{A}|$ (fine here:
  $(2K_Q{+}1) \times T \times (K{+}1)^2$); the $\max$ operator introduces
  **maximization bias** (overestimation), which Double Q-learning mitigates;
  needs explicit exploration; does not scale to continuous/high-dimensional states
  without function approximation (DQN).

---

## 4.3. Policy Gradients and PPO

### 4.3.1 Idea

PPO is a **model-free, on-policy, policy-gradient** method. Instead of learning
values and acting greedily, it parameterizes the policy directly,
$\pi_\theta(a \mid s)$ (a neural network), and adjusts $\theta$ to increase expected
return. A separate **value network** $V_\phi(s)$ is learned to reduce gradient
variance (this is the "actor–critic" structure: actor $\pi_\theta$, critic $V_\phi$).

### 4.3.2 Deriving the policy gradient

We want $\nabla_\theta J(\theta)$ where $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$
and $\tau = (s_0, a_0, s_1, a_1, \dots)$ is a trajectory with return $R(\tau)$.

**Five facts.**

*(1) Probability of a trajectory.* Under $\pi_\theta$,

$$P(\tau \mid \theta) = \rho_0(s_0) \prod_{t=0}^{T} P(s_{t+1} \mid s_t, a_t)\, \pi_\theta(a_t \mid s_t),$$

with $\rho_0$ the initial-state distribution.

*(2) The log-derivative trick.* Since $\nabla_x \log x = 1/x$, the chain rule gives

$$\nabla_\theta P(\tau \mid \theta) = P(\tau \mid \theta)\, \nabla_\theta \log P(\tau \mid \theta).$$

*(3) Log-probability of a trajectory.* Taking logs of (1) turns the product into a sum:

$$\log P(\tau \mid \theta) = \log \rho_0(s_0) + \sum_{t=0}^{T}\Big(\log P(s_{t+1} \mid s_t, a_t) + \log \pi_\theta(a_t \mid s_t)\Big).$$

*(4) Gradients of environment terms vanish.* The environment ($\rho_0$, $P$, $R$) does
not depend on $\theta$, so $\nabla_\theta \log \rho_0 = 0$ and
$\nabla_\theta \log P(s_{t+1}\mid s_t,a_t) = 0$.

*(5) Grad-log-prob of a trajectory.* Combining (3) and (4), every environment term
drops and only the policy terms survive:

$$\nabla_\theta \log P(\tau \mid \theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t).$$

**Putting it together.**

$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \nabla_\theta \int_\tau P(\tau \mid \theta)\, R(\tau)
&&\text{(expand expectation)}\\
&= \int_\tau \nabla_\theta P(\tau \mid \theta)\, R(\tau)
&&\text{(gradient under integral)}\\
&= \int_\tau P(\tau \mid \theta)\, \nabla_\theta \log P(\tau \mid \theta)\, R(\tau)
&&\text{(log-derivative trick, fact 2)}\\
&= \mathbb{E}_{\tau \sim \pi_\theta}\!\big[\nabla_\theta \log P(\tau \mid \theta)\, R(\tau)\big]
&&\text{(back to expectation)}\\
&= \mathbb{E}_{\tau \sim \pi_\theta}\!\Big[\textstyle\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, R(\tau)\Big]
&&\text{(fact 5).}
\end{aligned}
$$

This is the **simplest policy gradient**. It is an expectation, so it can be estimated
from a batch of trajectories $\mathcal{D} = \{\tau_i\}$ by a sample mean:

$$\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, R(\tau).$$

Intuition: each action's log-probability is pushed up in proportion to the total
return of the trajectory it belonged to.

### 3.3 The Expected Grad-Log-Prob (EGLP) lemma

A small but pivotal result, used to both *drop* and *add* terms below.

**Lemma.** For any parameterized distribution $P_\theta$ over a variable $x$,

$$\mathbb{E}_{x \sim P_\theta}\!\big[\nabla_\theta \log P_\theta(x)\big] = 0.$$

**Proof.** Every distribution normalizes: $\int_x P_\theta(x) = 1$. Differentiate both
sides, then apply the log-derivative trick:

$$0 = \nabla_\theta \!\int_x P_\theta(x)
= \int_x \nabla_\theta P_\theta(x)
= \int_x P_\theta(x)\, \nabla_\theta \log P_\theta(x)
= \mathbb{E}_{x \sim P_\theta}\!\big[\nabla_\theta \log P_\theta(x)\big]. \qquad \blacksquare$$

The gradient of the log-probability has **zero mean** under its own distribution.

### 4.3.4 Reward-to-go — dropping useless terms

In the simplest gradient, each action $a_t$ is weighted by $R(\tau)$, the sum of
**all** rewards in the episode — including those collected *before* $t$. But an action
cannot affect past rewards, so those terms only add noise. Formally, the contribution
of any past reward couples to $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ with zero
mean (by the EGLP lemma applied at $s_t$) but nonzero variance. Removing them is
unbiased and lowers variance. The result is the **reward-to-go** form:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\Big[
\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
\underbrace{\sum_{t'=t}^{T} R(s_{t'}, a_{t'}, s_{t'+1})}_{\hat{R}_t}\Big],$$

where $\hat{R}_t$ is the return accumulated **from** $t$ onward. Each action is now
reinforced only by its own consequences.

### 4.3.5 Baselines — adding useful terms

A direct corollary of the EGLP lemma: for any function $b(s_t)$ that depends **only on
state** (not on the action),

$$\mathbb{E}_{a_t \sim \pi_\theta}\!\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, b(s_t)\big] = 0.$$

(Inside the expectation over $a_t$, $b(s_t)$ is constant and factors out, leaving the
EGLP-lemma expectation, which is zero.) So we may **subtract** any such $b(s_t)$ from
the weight without changing the gradient in expectation:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\Big[
\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
\big(\hat{R}_t - b(s_t)\big)\Big].$$

Such a $b$ is a **baseline**. It leaves the gradient **unbiased** (mean unchanged) but
can sharply **reduce variance**. The most common choice is the on-policy value
function $b(s_t) = V^\pi(s_t)$ — the average return from $s_t$. Conceptually: an action
is reinforced by how much better it did than expected, so "getting what you expected"
produces a neutral signal.

### 4.3.6 The advantage form

With $b(s_t) = V^\pi(s_t)$ and recognizing that $\hat{R}_t$ is a sample of
$Q^\pi(s_t, a_t)$, the weight $\hat{R}_t - V^\pi(s_t)$ estimates the **advantage**

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s),$$

how much better $a$ is than the policy's average at $s$. This gives the form used in
practice:

$$\boxed{\;\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\Big[
\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A^\pi(s_t, a_t)\Big].\;}$$

More generally the gradient has the shape
$\mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \Phi_t]$, and
$\Phi_t \in \{R(\tau),\; \hat{R}_t,\; \hat{R}_t - b(s_t),\; Q^\pi(s_t,a_t),\; A^\pi(s_t,a_t)\}$
are all valid (same mean, different variance). The advantage choice has the lowest
variance and is what PPO uses.

### 4.3.7 Advantage estimation (GAE)

$A^\pi$ is not known and must be estimated from rollouts. The one-step TD residual,
using the learned critic $V_\phi$, is

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t),$$

itself a one-step advantage estimate. **Generalized Advantage Estimation** combines
residuals across horizons with a decay $\lambda \in [0, 1]$:

$$\hat{A}_t = \sum_{l=0}^{T-1-t} (\gamma\lambda)^l\, \delta_{t+l},$$

trading bias ($\lambda \to 0$: low variance, leans on the critic) against variance
($\lambda \to 1$: low bias, leans on raw returns). For the finite horizon the sum
terminates at $T$ and the bootstrap at the last step uses the terminal value.

### 4.3.8 The clipped surrogate objective — what makes it "PPO"

Vanilla policy gradients take one gradient step per batch of fresh on-policy data —
sample-inefficient and unstable if the step is too large. PPO reuses each batch for
several epochs while preventing the policy from moving too far. Define the
probability ratio between the new and old (data-collecting) policy:

$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}.$$

PPO maximizes the **clipped surrogate**

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[
\min\!\Big(\rho_t \hat{A}_t,\;
\operatorname{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\Big)\right].$$

The clip removes the incentive to push $\rho_t$ beyond $[1-\epsilon, 1+\epsilon]$
(typically $\epsilon \approx 0.2$): once the ratio leaves the trust region the
objective flattens, so a single batch cannot move the policy arbitrarily far. This is
the core trick — it gives most of the stability of a trust-region method (TRPO) with
first-order optimization.

### 4.3.9 The full objective

PPO optimizes a sum of three terms:

$$L(\theta, \phi) = \underbrace{L^{\text{CLIP}}(\theta)}_{\text{policy}}
\;-\; c_1\, \underbrace{\mathbb{E}_t\big[(V_\phi(s_t) - \hat{G}_t)^2\big]}_{\text{value (critic) loss}}
\;+\; c_2\, \underbrace{\mathbb{E}_t\big[\mathcal{H}(\pi_\theta(\cdot \mid s_t))\big]}_{\text{entropy bonus}},$$

where $\hat{G}_t = \hat{A}_t + V_\phi(s_t)$ is the value target, the critic loss fits
$V_\phi$ to returns, and the entropy bonus $\mathcal{H}$ encourages exploration by
discouraging premature collapse to a deterministic policy. $c_1, c_2$ weight the
terms.

### 4.3.10 The training loop

1. Run $\pi_{\theta_{\text{old}}}$ in the environment to collect a batch of rollouts.
2. Compute returns $\hat{G}_t$ and advantages $\hat{A}_t$ (GAE).
3. For several epochs, take minibatch gradient steps on $L(\theta, \phi)$.
4. Set $\theta_{\text{old}} \leftarrow \theta$, discard the old data, repeat.

PPO is **on-policy**: each batch is collected fresh under the current policy and
discarded after its few epochs of reuse (unlike Q-learning, which can replay old
data).

### 4.3.11 Convergence and relation to DP

Policy gradients converge to a **local** optimum of $J(\theta)$ under standard
stochastic-approximation conditions; with a sufficiently expressive network and
adequate exploration the local optimum is typically near-global for problems this
small. PPO does **not** come with the exact-convergence guarantee that tabular
Q-learning has — function approximation and the clip introduce bias.

**Consequence for the comparison:** expect PPO to *approximately* match DP. It should
recover the qualitative structure of $\pi^*$ — wider quotes on the side that would
worsen inventory, tighter on the side that unwinds it, and increasing urgency to
flatten inventory as $t \to T$ — but be fuzzier near the inventory boundaries and the
terminal time. The size and location of the residual gap is itself an informative
result.

> **Caveat — state must include time.** As with Q-learning, the policy and value
> networks must receive $t$ (or steps-remaining), normalized alongside $q$, or they
> cannot represent the time-dependent optimal policy.

---

## 4.4. Side-by-side summary

| Aspect | DP (ground truth) | Q-Learning | PPO |
| --- | --- | --- | --- |
| Class | model-based, exact | model-free, value-based | model-free, policy-gradient |
| On/off-policy | n/a | off-policy | on-policy |
| Knows $P, R$? | yes (closed form) | no (samples) | no (samples) |
| Learns | $V^*, \pi^*$ exactly | $Q^*$ table | $\pi_\theta$, $V_\phi$ nets |
| Exploration | n/a | $\varepsilon$-greedy | entropy + stochastic policy |
| Representation | exact over $(q,t)$ | table over $(q,t,a)$ | function approximation |
| Guarantee | exact optimum | exact in tabular limit | local optimum |
| Expected vs DP | — | matches in the limit | approximate match |
| Main failure mode | none (it is truth) | slow / under-explored | local optima, FA bias |

---

## 4.5. Why all three must share one environment

For the comparison to be valid, DP, Q-learning, and PPO must solve the **identical**
MDP — same $P$, $R$, terminal condition, and horizon, all derived from the same
parameters and equations. The recommended structure is a single environment exposing
two interfaces on the *same* underlying dynamics:

- **Closed-form interface** — $R(q, a)$ and the transition row $P(\cdot \mid q, a)$,
  consumed by DP.
- **Sampling interface** — `reset()` / `step(action)`, consumed by Q-learning and
  PPO.

If the two interfaces ever diverge, the "ground truth" is no longer ground truth for
what the learners actually experienced. This shared-environment design is the
foundation of the comparison and is addressed in the implementation phase.

---

## 4.6. What we will measure (preview)

The comparison will use, at minimum:

- **Policy agreement** — fraction of $(q, t)$ cells where the learner's greedy action
  matches $\pi^*$, weighted by the optimal state-visitation distribution.
- **Value gap** — $V^*(s) - V^{\pi_{\text{learner}}}(s)$, the reward left on the table
  by following the learned policy in the exact environment.
- **Realized PnL distribution** — mean *and* variance of terminal wealth over many
  rollouts (variance is where the inventory penalty shows up).
- **Policy heatmaps** — $\pi^*_t(q)$ vs learner $\pi_t(q)$ over the $(q, t)$ grid,
  with their difference, to localize where and when the learners diverge.

Details and code are deferred to the implementation phase.
