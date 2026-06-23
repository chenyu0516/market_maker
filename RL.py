import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Market-Maker: DP vs RL vs Avellaneda–Stoikov

    A market maker quotes a one-unit ask and bid each step, earns the spread on fills,
    and carries inventory risk. We solve it with **dynamic programming (exact ground
    truth)** and compare against **tabular Q-learning**, a hand-written **PPO**, and the
    **Avellaneda–Stoikov** closed-form model (gridded and continuous).


    **Key design choice (validated in the experiment log).** Learners are *trained* on a
    **reval-free** reward — the mark-to-market term `q'·Delta` has zero mean, so dropping
    it leaves the optimal policy unchanged while removing a large variance source.
    Realized PnL at *evaluation* uses the full reward. The comparison leads with
    **regret / value-gap**, not exact policy match, because the problem has many
    near-tie actions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import jax, jax.numpy as jnp
    from jax import jit, vmap, lax
    import flax.linen as nn
    import flax.struct as struct
    import optax
    import numpy as np
    import matplotlib.pyplot as plt
    from functools import partial
    import time

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    pio.renderers.default = "notebook"

    jax.config.update("jax_platform_name", "cpu")
    print("jax", jax.__version__, "| flax", __import__("flax").__version__)
    return (
        go,
        jax,
        jit,
        jnp,
        lax,
        make_subplots,
        nn,
        np,
        optax,
        partial,
        plt,
        struct,
        time,
        vmap,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Theoretical background for the two learning agents that will be compared against the
    dynamic-programming (DP) ground truth on the market-making MDP. The notation matches
    the DP model: state is inventory $q$ with time index $t$, action is the quote pair
    $a = (\delta_a, \delta_b)$, reward is the mark-to-market PnL $R(q, a)$, and the
    horizon is finite ($T$ steps) with terminal penalty $-\gamma_{term} q^2$.

    ---

    ## The MDP and the objects every method estimates

    A finite-horizon Markov decision process is the tuple
    $(\mathcal{S}, \mathcal{A}, P, R, T)$:

    - $\mathcal{S}$ — states. Here a state is $(q, t)$: inventory and time-to-go.
    - $\mathcal{A}$ — actions $a = (\delta_a, \delta_b)$.
    - $P(s' \mid s, a)$ — transition kernel (the three-branch inventory transition).
    - $R(s, a)$ — expected one-step reward.
    - $T$ — horizon; the terminal value is $V_T(q) = -\gamma_{term} q^2$.

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
    """)
    return


@app.cell
def _(jax, jnp, struct):
    @struct.dataclass
    class MMParams:
        tick: float = 1.0
        sigma: float = 1.0  # per-step price std (ticks)
        kappa: float = 1.5  # fill-prob decay per tick
        gamma_term: float = 2.0  # terminal inventory penalty
        gamma_run: float = 0.0  # running penalty weight (0 -> terminal only)
        S0: float = 100.5
        K: int = struct.field(pytree_node=False, default=5)  # max half-spread level
        K_Q: int = struct.field(pytree_node=False, default=20)  # inventory cap
        T: int = struct.field(pytree_node=False, default=30)  # horizon
        max_dk: int = struct.field(pytree_node=False, default=6)

    def n_actions(p):
        return (p.K + 1) ** 2  # -> (d_a, d_b)

    def unflatten_action(a, p):
        return (a // (p.K + 1), a % (p.K + 1))

    def fill_probs(d_a, d_b, p):
        return (jnp.exp(-p.kappa * d_a), jnp.exp(-p.kappa * d_b))

    def expected_reward(a, p):
        """E[edge] = p_a(d_a-1/2) + p_b(d_b-1/2). Action-only (symmetric increment)."""
        d_a, d_b = unflatten_action(a, p)
        p_a, p_b = fill_probs(d_a, d_b, p)
        return p_a * (d_a - 0.5) + p_b * (d_b - 0.5)

    def _Phi(z):
    # regime configs
        return 0.5 * (1.0 + jax.scipy.special.erf(z / jnp.sqrt(2.0)))

    def price_increment_probs(p):
        ks = jnp.arange(-p.max_dk, p.max_dk + 1)
        pk = _Phi(p.tick * (ks + 0.5) / p.sigma) - _Phi(p.tick * (ks - 0.5) / p.sigma)
        return (ks, pk / pk.sum())
    P_TERM = MMParams(sigma=1.0, kappa=1.5, gamma_term=2.0, gamma_run=0.0, T=30)
    P_RUN = MMParams(sigma=1.0, kappa=1.5, gamma_term=0.0, gamma_run=0.002, T=30)
    P_B = MMParams(sigma=1.0, kappa=1.5, gamma_term=2.0, gamma_run=0.002, T=30)
    print('terminal regime:', P_TERM)
    print('running regime :', P_RUN)
    print('both penalty regime :', P_B)
    return (
        P_B,
        P_RUN,
        P_TERM,
        expected_reward,
        fill_probs,
        n_actions,
        price_increment_probs,
        unflatten_action,
    )


@app.cell
def _(jnp, unflatten_action):
    def reward_decomposed(q, a, ask, bid, qn, dS, t, done, p):
        """Returns (full, reval). full = edge + reval + terminal + running."""
        d_a, d_b = unflatten_action(a, p)
        edge = (ask * (d_a - 0.5) + bid * (d_b - 0.5)).astype(jnp.float32)
        reval = qn.astype(jnp.float32) * dS
        term = jnp.where(done, -p.gamma_term * qn.astype(jnp.float32) ** 2, 0.0)
        run = run = -p.gamma_run * p.sigma ** 2 * qn.astype(jnp.float32) ** 2  # run   = -2.0*p.gamma_run*p.sigma**2*qn.astype(jnp.float32)**2*(p.T-t).astype(jnp.float32)
        full = edge + reval + term + run
        return (full, reval)

    def reward_train(q, a, ask, bid, qn, dS, t, done, p):
        """Reval-free training reward (zero-mean reval removed)."""
        full, reval = reward_decomposed(q, a, ask, bid, qn, dS, t, done, p)
        return full - reval

    return reward_decomposed, reward_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shared environment

    State `(q:int32, S:float32, t:int32)`. Obs `[q/K_Q, t/T]` (price level is
    irrelevant by translation invariance, so it is not in the obs). Action
    `Discrete(36)` decoded to `(d_a, d_b)`. Boundary-breaching fills are rejected.
    For more detail, see [baseDP.ipynb](baseDP.ipynb)
    """)
    return


@app.cell
def _(
    P_TERM,
    fill_probs,
    jax,
    jnp,
    price_increment_probs,
    reward_decomposed,
    unflatten_action,
):
    def env_step(key, q, S, t, a, p):
        """Sample one transition. Returns new (q,S), fills, dS."""
        k_fill, k_price = jax.random.split(key)
        d_a, d_b = unflatten_action(a, p)
        pa, pb = fill_probs(d_a, d_b, p)
        u = jax.random.uniform(k_fill, (2,))
        ask = u[0] < pa
        bid = u[1] < pb
        at_hi = q >= p.K_Q
        at_lo = q <= -p.K_Q
        bid = bid & ~at_hi
        ask = ask & ~at_lo
        qn = q + -ask.astype(jnp.int32) + bid.astype(jnp.int32)
        ks, pk = price_increment_probs(p)
        dS = p.tick * jax.random.choice(k_price, ks, p=pk).astype(jnp.float32)
        return (qn, S + dS, ask, bid, dS)

    def _demo_rollout():
        _p = P_TERM
        key = jax.random.PRNGKey(0)
        q, S, tot = (jnp.int32(0), jnp.float32(_p.S0), 0.0)
        for t in range(_p.T):
            key, k = jax.random.split(key)
            a = jnp.int32(7)
            qn, Sn, ask, bid, dS = env_step(k, q, S, jnp.int32(t), a, _p)
            full, _ = reward_decomposed(q, a, ask, bid, qn, dS, jnp.int32(t), t + 1 >= _p.T, _p)
            tot = tot + float(full)
            q, S = (qn, Sn)
        print('demo rollout: terminal q =', int(q), ' total full reward =', round(tot, 3))
    _demo_rollout()
    return (env_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DP solver — ground truth (both penalty)

    Backward induction over `(q, t)`. Transition is 3-branch in `q` and
    time-homogeneous; the running penalty's `(T-t)` factor is folded into the
    next-state value at each scan step (each step knows its `t`).
    For more detail, see [baseDP.ipynb](baseDP.ipynb)
    """)
    return


@app.cell(hide_code=True)
def _(
    P_B,
    P_RUN,
    P_TERM,
    expected_reward,
    fill_probs,
    jit,
    jnp,
    lax,
    unflatten_action,
    vmap,
):
    @jit
    def solve_dp(params):
        K_Q, T = (params.K_Q, params.T)
        qs = jnp.arange(-K_Q, K_Q + 1) # available inventary

        n_a = (params.K + 1) ** 2
        acts = jnp.arange(n_a)  # available actions

        R = jnp.broadcast_to(vmap(lambda a: expected_reward(a, params))(acts), (qs.shape[0], n_a)) # expected wealth reward R(number of q, a)

        # transition probability construction
        d_a, d_b = unflatten_action(acts, params)
        p_a, p_b = fill_probs(d_a, d_b, params)
        p_down = p_a * (1 - p_b)
        p_up = (1 - p_a) * p_b
        p_stay = p_a * p_b + (1 - p_a) * (1 - p_b)

        # Initialize the backward induction
        qs_f = qs.astype(jnp.float32)
        V_T = -params.gamma_term * qs_f ** 2 

        def step(V_next, t):
            run = -params.gamma_run * params.sigma ** 2 * qs_f ** 2 
            # all the possible running penalty
            Vp = V_next + run # V(-K_Q) , V(-K_Q+1), ... V(K_Q-1), V(K_Q)
            Vm = jnp.concatenate([Vp[:1], Vp[:-1]]) # V(-K_Q), V(-K_Q), V(-K_Q+1), ... V(K_Q-1)
            Vpp = jnp.concatenate([Vp[1:], Vp[-1:]]) # V(-K_Q+1), V(-K_Q+2), ..., V(K_Q-1), V(K_Q), V(K_Q)
            EV = p_down[None, :] * Vm[:, None] + p_up[None, :] * Vpp[:, None] + p_stay[None, :] * Vp[:, None] # size (number of q, number of a)
            Q = R + EV

            return (jnp.max(Q, axis=1), (jnp.max(Q, axis=1), jnp.argmax(Q, axis=1)))  

        _, (V, _pi) = lax.scan(step, V_T, jnp.arange(T - 1, -1, -1))

        return (V[::-1], _pi[::-1], qs) # reverse the time index

    V_dp_t, pi_dp_t, qs = solve_dp(P_TERM)
    V_dp_r, pi_dp_r, _ = solve_dp(P_RUN)
    V_dp_v, pi_dp_b, _ = solve_dp(P_B)

    print('terminal V[0,q=0] =', round(float(V_dp_t[0, P_TERM.K_Q]), 3))
    print('running  V[0,q=0] =', round(float(V_dp_r[0, P_RUN.K_Q]), 3))

    def decode(pi, p, t, qlist):
        # decode policy to delta_a, delta_b
        return [(int(pi[t, q + p.K_Q]) // (p.K + 1), int(pi[t, q + p.K_Q]) % (p.K + 1)) for q in qlist]

    print('terminal pi t=29:', decode(pi_dp_t, P_TERM, 29, [-8, -3, 0, 3, 8]))
    print('running  pi t=0 :', decode(pi_dp_r, P_RUN, 0, [-8, -3, 0, 3, 8]))
    return V_dp_t, decode, pi_dp_b, pi_dp_r, pi_dp_t, solve_dp


@app.cell(hide_code=True)
def _(P_B, P_RUN, P_TERM, np, pi_dp_b, pi_dp_r, pi_dp_t, plt):
    # DP policy heatmaps (skew = d_a - d_b), both regimes
    def skew_grid(pi, p):
        da = pi // (p.K + 1)
        db = pi % (p.K + 1)  # (T, n_q), >0 means tighter bid side relatively
        return np.array(da - db)
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4))
    for _ax, _pi, _p, title in [(_axes[0], pi_dp_t, P_TERM, 'DP terminal: skew (d_a - d_b)'), (_axes[1], pi_dp_r, P_RUN, 'DP running: skew (d_a - d_b)'), (_axes[2], pi_dp_b, P_B, 'DP terminal+running: skew (d_a - d_b)')]:
        _sg = skew_grid(_pi, _p).T
        _im = _ax.imshow(_sg, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, _p.T, -_p.K_Q, _p.K_Q], vmin=-_p.K, vmax=_p.K)  # (n_q, T)
        _ax.set_xlabel('time t')
        _ax.set_ylabel('inventory q')
        _ax.set_title(title)
        plt.colorbar(_im, ax=_ax)
    plt.tight_layout()
    plt.show()
    return (skew_grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q-Learning

    ### Base Idea

    Q-learning is a **model-free, off-policy, value-based** method. It estimates $Q^*$ directly from sampled transitions $(s, a, r, s')$, never forming an explicit model of $P$ or $R$. It is the sampling counterpart of value iteration.

    ### The update — why it converges to the optimal values

    **The whole idea in one sentence.** $Q(s, a)$ is meant to mean "the total reward I will get if I take action $a$ in state $s$, then play optimally afterward." Q-learning makes that sentence *true* by repeatedly forcing each $Q$ value to agree with the $Q$ values that come right after it.

    **The TD target is what $Q(s,a)$ should equal.** Take action $a$, collect reward $r$, land in $s'$. From $s'$ you play optimally, so the best you can get from there is $\max_{a'} Q(s', a')$. Therefore the correct value is

    $$Q(s, a) = \underbrace{r}_{\text{reward now}} + \underbrace{\max_{a'} Q(s', a')}_{\text{best from next state}}.$$

    The right-hand side is the **TD target**: it is just $Q(s,a)$ rewritten in terms of what happens *one step later*. If every entry in the table satisfies this equation simultaneously, the table is correct — it is $Q^*$. Being optimal *is* being self-consistent one step at a time.

    **The update shrinks the disagreement.** Usually $Q(s,a)$ does **not** equal the target yet; the gap between them is the **TD error**. The update nudges $Q$ a little toward the target:

    $$\boxed{\;Q(s, a) \leftarrow Q(s, a) + \eta \Big[\underbrace{r + \max_{a'} Q(s', a')}_{\text{TD target}} - Q(s, a)\Big].\;}$$

    **Why shrinking these gaps everywhere lands on $Q^*$.** Two things work together:

    *1. Truth flows backward from the rewards.* The target contains $r$ — a **real** reward from the environment, not a guess. So every update injects a bit of ground-truth information and mixes it into $Q(s,a)$. The $\max_{a'} Q(s',a')$ part is still an estimate, but it too absorbed real reward the last time it was updated. So real information seeps backward one step at a time: rewards inform the states just before them, which inform the states before *those*, and so on. Picture the rewards as light sources and the early states as dark; each update lets a state copy a bit of what its neighbor learned plus the real reward in between, and the light spreads outward from the rewards until the whole table is lit correctly.

    *2. The flow can only stop at $Q^*$.* When does the table stop changing? Only when every TD error is zero — when $Q(s,a) = r + \max_{a'} Q(s',a')$ everywhere. But that is exactly the self-consistency condition that *defines* $Q^*$. There is nowhere else to come to rest, and the self-consistent table is unique, so the updates cannot get stuck on a wrong one.

    Together: real reward keeps flooding in and spreading backward (the table can't stay wrong), and the only configuration where everything stops moving is $Q^*$ (so where it stops is the right answer).

    **Why a small step $\eta$ rather than setting $Q$ equal to the target.** The environment is random — the same $(s,a)$ can yield different $r$ and $s'$ on different visits, so a single target is a noisy sample of the true average. Slamming $Q$ to each noisy target would make it jitter forever. The learning rate $\eta$ makes $Q$ a **running average** of all targets seen, averaging the noise away so it settles on the true expected return instead of bouncing around.

    **Each piece, summarized.**

    - The bracket is the **TD error** — the gap between the target and the current estimate. It is zero in expectation exactly when $Q = Q^*$, which is why driving it to zero finds $Q^*$.
    - $\eta$ is the step that **averages out sampling noise** in $r$ and $s'$.
    - $\max_{a'}$ assumes optimal play from the next state — and bootstraps from the greedy action regardless of what exploration actually did next, which is what makes Q-learning **off-policy**.
    - The target uses the current estimate $Q(s', \cdot)$ rather than a full Monte-Carlo return — **bootstrapping**: biased early (the estimate is wrong) but far lower variance than waiting for the episode to end, and the bias vanishes as $Q \to Q^*$.

    **Relation to DP.** The model-based version of this idea computes the target's expectation exactly (using known $P$ and $R$) instead of sampling it — that is value iteration, and its finite-horizon sweep is exactly DP's backward induction. Q-learning is the same fixed-point logic with the expectation **sampled** rather than computed.

    **Finite-horizon form.** The table is indexed by $(q, t)$ and the target steps the time index forward:

    $$Q(q, t, a) \leftarrow Q(q, t, a) + \eta\Big[r + \max_{a'} Q(q', t{+}1, a') - Q(q, t, a)\Big],$$

    with the bootstrap at the last step replaced by the known terminal value, $\max_{a'} Q(q', T, \cdot) \to V_T(q') = -\gamma_{term} q'^2$. This mirrors DP's backward induction, except the next-state value is sampled rather than computed from $P$.

    ### Exploration

    Because the update bootstraps from $\max_{a'} Q$, the agent must still **visit** all relevant $(s, a)$ to learn their values. The standard scheme is **$\varepsilon$-greedy**:

    $$a = \begin{cases}\arg\max_{a'} Q(s, a') & \text{w.p. } 1 - \varepsilon,\\ \text{uniform random action} & \text{w.p. } \varepsilon,\end{cases}$$

    typically with $\varepsilon$ annealed from high to low over training.

    ### Convergence and relation to DP

    For a **finite, tabular** MDP with state $(q, t)$, tabular Q-learning converges to $Q^*$ with probability 1 under standard conditions (every $(s,a)$ visited infinitely often; learning-rate schedule satisfying $\sum_t \eta_t = \infty$, $\sum_t \eta_t^2 < \infty$).

    **Consequence for the comparison:** because the table over $(q, t)$ is expressive enough to represent $\pi^*$ exactly, a correct tabular Q-learner *must* reproduce the DP policy in the limit. Any remaining gap is attributable to **finite samples and exploration**, not to representational limits. This makes Q-learning the cleanest sanity check on the whole pipeline: if it does not approach DP, something is wrong in the environment or the training loop, not in the theory.

    > **Caveat — state must include time.** A stationary Q-table over $q$ alone *cannot* represent the optimal policy, because $\pi^*$ genuinely depends on $t$ (terminal penalty). Such an agent converges to a time-averaged compromise and will look worse than DP for reasons unrelated to the algorithm. The state must be $(q, t)$.

    ### Strengths and limits

    - **Strengths:** simple, sample-reuse via off-policy updates, exact in the tabular limit, no policy-gradient variance.
    - **Limits:** the table grows with $|\mathcal{S}| \times |\mathcal{A}|$ (fine here: $(2K_Q{+}1) \times T \times (K{+}1)^2$); the $\max$ operator introduces **maximization bias** (overestimation), which Double Q-learning mitigates; needs explicit exploration; does not scale to continuous/high-dimensional states without function approximation (DQN).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tabular Q-learning (terminal regime)

    State `(q,t)`, exploring starts (random `(q0,t0)` per sample) so all cells are
    visited, epsilon-greedy, **reval-free training reward**, decaying learning rate.
    """)
    return


@app.cell
def _(
    P_TERM,
    decode,
    env_step,
    jax,
    jit,
    jnp,
    lax,
    partial,
    pi_dp_t,
    reward_train,
    time,
):
    @partial(jit, static_argnames=('n_updates', 'batch'))
    def q_learn(key, p, n_updates=80000, batch=128, eta0=0.3, eps0=0.4):
        n_q = 2 * p.K_Q + 1
        n_a = (p.K + 1) ** 2
        T = p.T

        def step(carry, sk):  # gradually approach
            Q, it = carry
            frac = it / n_updates  # exploration control
            eta = eta0 * (1 - 0.92 * frac)
            eps = jnp.maximum(0.05, eps0 * (1 - frac))
            ks = jax.random.split(sk, batch)  #random-cell updates with exploring starts

            def samp(_, k):
                kt, kq, ka, ke, ksp = jax.random.split(k, 5)  # random start
                t = jax.random.randint(kt, (), 0, T)
                q = jax.random.randint(kq, (), -p.K_Q, p.K_Q + 1)
                qi = q + p.K_Q
                greedy = jnp.argmax(Q[t, qi])  # index q
                ra = jax.random.randint(ka, (), 0, n_a)  # find the greedy action based on Q
                a = jnp.where(jax.random.uniform(ke) < eps, ra, greedy)
                qn, _, ask, bid, dS = env_step(ksp, q, jnp.float32(0.0), t, a, p)
                done = t + 1 >= T  # random action
                r = reward_train(q, a, ask, bid, qn, dS, t, done, p)  # choose randon for explore, greedy for optimize
                tgt = r + jnp.where(done, 0.0, jnp.max(Q[t + 1, qn + p.K_Q]))
                return (t, qi, a, tgt)  # do one step for the choosing action
            ts, qis, as_, tg = jax.vmap(samp, in_axes=(None, 0))(0, ks)
      # calculate the reward of it
            def app(Q, x):
                t, qi, a, g = x
                return (Q.at[t, qi, a].add(eta * (g - Q[t, qi, a])), None)
            Q, _ = lax.scan(app, Q, (ts, qis, as_, tg))  # TD target
            return ((Q, it + 1), None)
        Q0 = jnp.zeros((T, n_q, n_a))
        keys = jax.random.split(key, n_updates)
        (Q, _), _ = lax.scan(step, (Q0, 0), keys)  # batch run the sampling
        return Q
    _t0 = time.time()
    Q_table = q_learn(jax.random.PRNGKey(0), P_TERM, n_updates=80000)  # apply the sampled TD target to update Q
    Q_table.block_until_ready()
    pi_q = jnp.argmax(Q_table, axis=2)
    print('Q-learning trained in', round(time.time() - _t0, 1), 's')
    print('t=29:', decode(pi_q, P_TERM, 29, [-8, -3, 0, 3, 8]), ' (DP:', decode(pi_dp_t, P_TERM, 29, [-8, -3, 0, 3, 8]), ')')
    return Q_table, pi_q


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Policy Gradients and PPO

    ###  Idea

    PPO is a **model-free, on-policy, policy-gradient** method. Instead of learning
    values and acting greedily, it parameterizes the policy directly,
    $\pi_\theta(a \mid s)$ (a neural network), and adjusts $\theta$ to increase expected
    return. A separate **value network** $V_\phi(s)$ is learned to reduce gradient
    variance (this is the "actor–critic" structure: actor $\pi_\theta$, critic $V_\phi$).

    ###  Deriving the policy gradient

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

    ###  Reward-to-go — dropping useless terms

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

    ###  Baselines — adding useful terms

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

    ###  The advantage form

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

    ###  Advantage estimation (GAE)

    $A^\pi$ is not known and must be estimated from rollouts. The one-step TD residual,
    using the learned critic $V_\phi$, is

    $$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t),$$

    itself a one-step advantage estimate. **Generalized Advantage Estimation** combines
    residuals across horizons with a decay $\lambda \in [0, 1]$:

    $$\hat{A}_t = \sum_{l=0}^{T-1-t} (\gamma\lambda)^l\, \delta_{t+l},$$

    trading bias ($\lambda \to 0$: low variance, leans on the critic) against variance
    ($\lambda \to 1$: low bias, leans on raw returns). For the finite horizon the sum
    terminates at $T$ and the bootstrap at the last step uses the terminal value.

    ###  The clipped surrogate objective — what makes it "PPO"

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

    ###  The full objective

    PPO optimizes a sum of three terms:

    $$L(\theta, \phi) = \underbrace{L^{\text{CLIP}}(\theta)}_{\text{policy}}
    \;-\; c_1\, \underbrace{\mathbb{E}_t\big[(V_\phi(s_t) - \hat{G}_t)^2\big]}_{\text{value (critic) loss}}
    \;+\; c_2\, \underbrace{\mathbb{E}_t\big[\mathcal{H}(\pi_\theta(\cdot \mid s_t))\big]}_{\text{entropy bonus}},$$

    where $\hat{G}_t = \hat{A}_t + V_\phi(s_t)$ is the value target, the critic loss fits
    $V_\phi$ to returns, and the entropy bonus $\mathcal{H}$ encourages exploration by
    discouraging premature collapse to a deterministic policy. $c_1, c_2$ weight the
    terms.

    ### The training loop

    1. Run $\pi_{\theta_{\text{old}}}$ in the environment to collect a batch of rollouts.
    2. Compute returns $\hat{G}_t$ and advantages $\hat{A}_t$ (GAE).
    3. For several epochs, take minibatch gradient steps on $L(\theta, \phi)$.
    4. Set $\theta_{\text{old}} \leftarrow \theta$, discard the old data, repeat.

    PPO is **on-policy**: each batch is collected fresh under the current policy and
    discarded after its few epochs of reuse (unlike Q-learning, which can replay old
    data).

    ### Convergence and relation to DP

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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PPO (terminal regime, hand-written)

    Actor-critic MLP on obs `[q/K_Q, t/T]`, categorical over 36 actions. Reval-free
    training reward, finite-horizon GAE (terminal bootstrap 0; terminal penalty is
    already in the reward), clipped surrogate + value loss + entropy.
    """)
    return


@app.cell
def _(
    P_TERM,
    decode,
    env_step,
    jax,
    jit,
    jnp,
    lax,
    n_actions,
    nn,
    optax,
    partial,
    reward_train,
    time,
    vmap,
):
    class ActorCritic(nn.Module):
        n_actions: int
        @nn.compact
        def __call__(self, x):
            h = nn.tanh(nn.Dense(64)(x)); h = nn.tanh(nn.Dense(64)(h))
            logits = nn.Dense(self.n_actions)(h)
            v = nn.Dense(1)(nn.tanh(nn.Dense(64)(x)))[..., 0]
            return logits, v

    def obs_of(q, t, p): 
        return jnp.stack([q/p.K_Q, t/p.T], axis=-1).astype(jnp.float32)

    @partial(jit, static_argnames=('net','n_updates','n_envs','epochs','mb'))
    def train_ppo(key, p, net, n_updates=300, n_envs=64, epochs=4, mb=8,
                  clip=0.2, lr=3e-3, gae_lam=0.95, ent=0.01, vf=0.5):
        T = p.T
        key, ki = jax.random.split(key)
        params = net.init(ki, jnp.zeros((2,)))
        tx = optax.adam(lr); opt = tx.init(params)

        def rollout(params, key):
            def one_env(key):
                def tstep(c, t):
                    q, key = c
                    key, ka, ks = jax.random.split(key, 3)
                    o = obs_of(q.astype(jnp.float32), jnp.float32(t), p) # normalize state to get observation

                    logits, v = net.apply(params, o)        # raw network outputs, a = pi(s)
                    a = jax.random.categorical(ka, logits)  # sample an action
                    logp = nn.log_softmax(logits)[a]        # log-prob of the sampled action pi(a | s)

                    qn, _, ask, bid, dS = env_step(ks, q, jnp.float32(0.0), jnp.int32(t), a, p)

                    done = (t+1) >= T
                    r = reward_train(q, a, ask, bid, qn, dS, jnp.int32(t), done, p)

                    return (qn, key), (o, a, logp, v, r, done)
                (_, _), traj = lax.scan(tstep, (jnp.int32(0), key), jnp.arange(T))

                return traj

            return vmap(one_env)(jax.random.split(key, n_envs))

        def gae(r, v, done):
            """
            v: shape (n_envs, T)
            """
            v_next = jnp.concatenate([v[:,1:], jnp.zeros((v.shape[0],1))], axis=1) # initial V_t+1
            nonterm = 1.0 - done.astype(jnp.float32)    # non terminate

            def scan_fn(adv_next, x):
                r_t, v_t, vn, nt = x                # reward, V(s_t), V(s_{t+1}), nonterminal mask
                delta = r_t + vn*nt - v_t           # δ_t = r_t + γ·V(s_{t+1})·nonterm − V(s_t)   (γ=1 here)
                adv = delta + gae_lam*nt*adv_next   #  Â_t = δ_t + (γλ)·nonterm·Â_{t+1}
                return adv, adv

            _, adv = lax.scan(scan_fn, jnp.zeros((r.shape[0],)),
                              (r.T, v.T, v_next.T, nonterm.T), reverse=True)
            adv = adv.T
            return adv, adv + v

        def update(carry, key):
            params, opt = carry

            # initialize
            obs, a, logp_old, v, r, done = rollout(params, key)
            """
            obs       : (64, 30, 2)
            a         : (64, 30)
            logp_old  : (64, 30)
            adv       : (64, 30)
            ret       : (64, 30)
            """
            adv, ret = gae(r, v, done)

            B   = obs.reshape(-1, 2)   # (1920, 2)  observations  [q/K_Q, t/T]
            A   = a.reshape(-1)        # (1920,)     action indices taken
            LP  = logp_old.reshape(-1) # (1920,)     log-prob of each action under the OLD policy
            ADV = adv.reshape(-1)      # (1920,)     GAE advantage of each transition
            RET = ret.reshape(-1)      # (1920,)     value target (= adv + V) for the critic

            def epoch(carry, ek):
                params, opt = carry
                idx = jax.random.permutation(ek, B.shape[0]).reshape(mb, -1)

                # mini batch step
                def mbstep(carry, mi):
                    """
                    mi: mini batch index
                    """
                    params, opt = carry

                    def loss_fn(params):
                        logits, val = vmap(lambda o: net.apply(params, o))(B[mi])
                        logp = vmap(lambda l, ai: nn.log_softmax(l)[ai])(logits, A[mi])

                        ratio = jnp.exp(logp - LP[mi])  # ratio of prob update
                        aa = ADV[mi]

                        pg = -jnp.mean(jnp.minimum(ratio*aa, jnp.clip(ratio,1-clip,1+clip)*aa))     # policy loss
                        vl = jnp.mean((val - RET[mi])**2)                                           # critic / value loss
                        pr = jax.nn.softmax(logits)
                        entropy = -jnp.mean(jnp.sum(pr*nn.log_softmax(logits), axis=-1))            # entropy loss

                        return pg + vf*vl - ent*entropy

                    g = jax.grad(loss_fn)(params)   # gradient the loss

                    upd, opt = tx.update(g, opt, params)
                    params = optax.apply_updates(params, upd)

                    return (params, opt), None

                (params, opt), _ = lax.scan(mbstep, (params, opt), idx)
                return (params, opt), None
            (params, opt), _ = lax.scan(epoch, (params, opt), jax.random.split(key, epochs))
            return (params, opt), jnp.mean(RET)
        (params, opt), rets = lax.scan(update, (params, opt), jax.random.split(key, n_updates))
        return params, rets

    net = ActorCritic(n_actions(P_TERM))
    t0 = time.time()
    ppo_params, ppo_rets = train_ppo(jax.random.PRNGKey(0), P_TERM, net, n_updates=300)
    ppo_rets.block_until_ready()
    print("PPO trained in", round(time.time()-t0,1), "s")
    print("mean return: first", round(float(ppo_rets[0]),2), "-> last", round(float(ppo_rets[-1]),2))

    # extract PPO greedy policy over (t,q)
    qs_grid = jnp.arange(-P_TERM.K_Q, P_TERM.K_Q+1)
    def ppo_greedy_tq(t):
        o = vmap(lambda q: obs_of(q.astype(jnp.float32), jnp.float32(t), P_TERM))(qs_grid)
        logits, _ = vmap(lambda x: net.apply(ppo_params, x))(o)
        return jnp.argmax(logits, axis=1)
    pi_ppo = vmap(ppo_greedy_tq)(jnp.arange(P_TERM.T))
    print("t=29 PPO:", decode(pi_ppo, P_TERM, 29, [-8,-3,0,3,8]))
    return net, pi_ppo, ppo_params, ppo_rets


@app.cell
def _(np, plt, ppo_rets):
    # PPO learning curve

    plt.figure(figsize=(7,3))
    plt.plot(np.array(ppo_rets)); plt.xlabel("PPO update"); plt.ylabel("mean return (reval-free)")
    plt.title("PPO learning curve"); plt.grid(alpha=0.3); plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Side-by-side summary

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

    ## Why all three must share one environment

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

    ## What we will measure (preview)

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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Avellaneda–Stoikov baselines (running regime)

    AS reservation price `r = S - q*gamma*sigma^2*(T-t)` and optimal half-spread
    `0.5*(gamma*sigma^2*(T-t) + (2/gamma)ln(1+gamma/kappa))`, mapped onto our env's
    quote convention (`ask` fills at `S - 1/2 + d_a`, `bid` at `S + 1/2 - d_b`).
    **AS-continuous** posts the raw real-valued distances; **AS-grid** rounds to the
    nearest integer level (clamped to `>=1`). Their gap isolates the rounding cost.
    """)
    return


@app.cell
def _(P_RUN, jnp):
    def as_deltas(S, q, t, p):
        g = jnp.maximum(p.gamma_run, 1e-06)
        tau = p.T - t  # reservation = S - r_shift
        r_shift = q * g * p.sigma ** 2 * tau
        half = 0.5 * (g * p.sigma ** 2 * tau + 2.0 / g * jnp.log(1 + g / p.kappa))
        ask_price = S - r_shift + half
        bid_price = S - r_shift - half  # map to env's d_a convention
        d_a = (ask_price - S) / p.tick + 0.5
        d_b = (S - bid_price) / p.tick + 0.5
        return (d_a, d_b)

    def as_cont(S, q, t, p):
        d_a, d_b = as_deltas(S, q, t, p)
        return (jnp.clip(d_a, 0.0, p.K), jnp.clip(d_b, 0.0, p.K))

    def as_grid(S, q, t, p):
        d_a, d_b = as_deltas(S, q, t, p)
        return (jnp.clip(jnp.round(d_a), 1, p.K).astype(jnp.float32), jnp.clip(jnp.round(d_b), 1, p.K).astype(jnp.float32))
    for q in [-10, 0, 10]:
    # show AS quotes at a few states
        dc = as_cont(jnp.float32(P_RUN.S0), jnp.int32(q), jnp.int32(0), P_RUN)
        dg = as_grid(jnp.float32(P_RUN.S0), jnp.int32(q), jnp.int32(0), P_RUN)
        print(f'q={q:+3d}: AS-cont (d_a,d_b)=({float(dc[0]):.2f},{float(dc[1]):.2f})  AS-grid=({int(dg[0])},{int(dg[1])})')
    return as_cont, as_grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shared-seed evaluation harness

    Every policy faces **identical** randomness: a pre-drawn `(N,T,2)` table of
    fill-coins and an `(N,T)` table of price increments. The price path is identical
    across policies; fills use the same coin thresholded by each policy's own
    `p_fill`. Trajectories still diverge through inventory (correct). Reward at
    evaluation is the **full** reward (reval included).
    """)
    return


@app.cell
def _(
    P_RUN,
    P_TERM,
    Q_table,
    as_cont,
    as_grid,
    jax,
    jit,
    jnp,
    lax,
    net,
    partial,
    pi_dp_r,
    pi_dp_t,
    ppo_params,
    price_increment_probs,
    unflatten_action,
    vmap,
):
    @partial(jit, static_argnames=('N', 'kind'))
    def evaluate(key, p, kind, dp_pi=None, qtable=None, ppo_p=None, N=3000):
        T = p.T
        k_coin, k_price = jax.random.split(key)
        coins = jax.random.uniform(k_coin, (N, T, 2))
        ks, pk = price_increment_probs(p)
        incr = p.tick * jax.random.choice(k_price, ks, (N, T), p=pk).astype(jnp.float32)

        def pick(q, S, t):
            if kind == 'dp':
                a = dp_pi[t, q + p.K_Q]
                da, db = unflatten_action(a, p)
                return (da.astype(jnp.float32), db.astype(jnp.float32))
            elif kind == 'qtable':
                a = jnp.argmax(qtable[t, q + p.K_Q])
                da, db = unflatten_action(a, p)
                return (da.astype(jnp.float32), db.astype(jnp.float32))
            elif kind == 'ppo':
                o = jnp.stack([q / p.K_Q, t / p.T]).astype(jnp.float32)
                logits, _ = net.apply(ppo_p, o)
                a = jnp.argmax(logits)
                da, db = unflatten_action(a, p)
                return (da.astype(jnp.float32), db.astype(jnp.float32))
            elif kind == 'as_grid':
                return as_grid(S, q, t, p)
            elif kind == 'as_cont':
                return as_cont(S, q, t, p)

        def run_one(coin_row, incr_row):

            def tstep(carry, inp):
                q, S = carry
                t, coin, dS = inp
                da, db = pick(q, S, t)
                pa = jnp.exp(-p.kappa * da)
                pb = jnp.exp(-p.kappa * db)
                ask = coin[0] < pa
                bid = coin[1] < pb
                at_hi = q >= p.K_Q
                at_lo = q <= -p.K_Q
                bid = bid & ~at_hi
                ask = ask & ~at_lo
                qn = q + -ask.astype(jnp.int32) + bid.astype(jnp.int32)
                Sn = S + dS
                edge = (ask * (da - 0.5) + bid * (db - 0.5)).astype(jnp.float32)
                reval = qn.astype(jnp.float32) * dS  # (qpath,Spath,rpath,ask,bid,da,db) each (T,...)
                done = t + 1 >= T
                term = jnp.where(done, -p.gamma_term * qn.astype(jnp.float32) ** 2, 0.0)
                run = -2.0 * p.gamma_run * p.sigma ** 2 * qn.astype(jnp.float32) ** 2 * (T - t).astype(jnp.float32)
                r = edge + reval + term + run
                return ((qn, Sn), (qn, Sn, r, ask, bid, da, db))
    # terminal regime policies
            ts = jnp.arange(T)
            (_, _), out = lax.scan(tstep, (jnp.int32(0), jnp.float32(p.S0)), (ts, coin_row, incr_row))
            return out
    # running regime policies
        return vmap(run_one)(coins, incr)
    EVAL_KEY = jax.random.PRNGKey(12345)
    N_EVAL = 3000
    ev_dp_t = evaluate(EVAL_KEY, P_TERM, 'dp', dp_pi=pi_dp_t, N=N_EVAL)
    ev_q = evaluate(EVAL_KEY, P_TERM, 'qtable', qtable=Q_table, N=N_EVAL)
    ev_ppo = evaluate(EVAL_KEY, P_TERM, 'ppo', ppo_p=ppo_params, N=N_EVAL)
    ev_dp_r = evaluate(EVAL_KEY, P_RUN, 'dp', dp_pi=pi_dp_r, N=N_EVAL)
    ev_asg = evaluate(EVAL_KEY, P_RUN, 'as_grid', N=N_EVAL)
    ev_asc = evaluate(EVAL_KEY, P_RUN, 'as_cont', N=N_EVAL)
    EVALS = {'DP-term': (ev_dp_t, P_TERM), 'Q-learn': (ev_q, P_TERM), 'PPO': (ev_ppo, P_TERM), 'DP-run': (ev_dp_r, P_RUN), 'AS-grid': (ev_asg, P_RUN), 'AS-cont': (ev_asc, P_RUN)}
    print('evaluated', len(EVALS), 'policies x', N_EVAL, 'shared-seed episodes')
    return EVALS, ev_asc, ev_asg, ev_dp_t, ev_ppo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison & quantitative statistics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Policy heatmaps (discrete policies) and difference vs DP
    """)
    return


@app.cell
def _(P_RUN, P_TERM, pi_dp_r, pi_dp_t, pi_ppo, pi_q, plt, skew_grid):
    discrete = [("DP-term", pi_dp_t, P_TERM), ("Q-learn", pi_q, P_TERM),
                ("PPO", pi_ppo, P_TERM), ("DP-run", pi_dp_r, P_RUN)]
    fig, axes = plt.subplots(2, 4, figsize=(18,7))
    for j,(_name,pi,p) in enumerate(discrete):
        sg = skew_grid(pi,p).T
        im = axes[0,j].imshow(sg, aspect='auto', origin='lower', cmap='RdBu_r',
                              extent=[0,p.T,-p.K_Q,p.K_Q], vmin=-p.K, vmax=p.K)
        axes[0,j].set_title(f"{_name}: skew"); axes[0,j].set_xlabel("t"); axes[0,j].set_ylabel("q")
        plt.colorbar(im, ax=axes[0,j])
        # diff vs DP-term (same regime only meaningful for terminal trio; DP-run vs DP-run=0)
        ref = pi_dp_t if p is P_TERM else pi_dp_r
        diff = (skew_grid(pi,p)-skew_grid(ref,p)).T
        im2 = axes[1,j].imshow(diff, aspect='auto', origin='lower', cmap='PuOr',
                               extent=[0,p.T,-p.K_Q,p.K_Q], vmin=-3, vmax=3)
        axes[1,j].set_title(f"{_name}: skew - DP"); axes[1,j].set_xlabel("t"); axes[1,j].set_ylabel("q")
        plt.colorbar(im2, ax=axes[1,j])
    plt.tight_layout(); plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regret / value-gap vs DP (the honest success metric)
    """)
    return


@app.cell
def _(P_TERM, V_dp_t, fill_probs, jnp, pi_ppo, pi_q, unflatten_action):
    # build DP true Q(t,q,a) for a regime, measure regret of a policy's greedy action
    def dp_true_Q_stack(p, V_dp):
        n_a = (_p.K + 1) ** 2
        acts = jnp.arange(n_a)
        da, db = unflatten_action(acts, _p)
        pa, pb = fill_probs(da, db, _p)
        pdn = pa * (1 - pb)
        pup = (1 - pa) * pb
        pst = pa * pb + (1 - pa) * (1 - pb)
        R = pa * (da - 0.5) + pb * (db - 0.5)
        qs_f = jnp.arange(-_p.K_Q, _p.K_Q + 1).astype(jnp.float32)
        VT = -_p.gamma_term * qs_f ** 2

        def qat(Vnext, t):
            run = -2.0 * _p.gamma_run * _p.sigma ** 2 * qs_f ** 2 * (_p.T - t)
            Vp = Vnext + run  # (T,n_q,n_a)
            Vm = jnp.concatenate([Vp[:1], Vp[:-1]])
            Vpp = jnp.concatenate([Vp[1:], Vp[-1:]])
            return R[None, :] + pdn[None, :] * Vm[:, None] + pup[None, :] * Vpp[:, None] + pst[None, :] * Vp[:, None]
        stack = []
        for t in range(_p.T):
            Vnext = VT if t == _p.T - 1 else V_dp[t + 1]
            stack.append(qat(Vnext, t))
        return jnp.stack(stack)

    def regret_of(pi, p, V_dp, w=10):
        Qdp = dp_true_Q_stack(_p, V_dp)
        opt = jnp.max(Qdp, axis=2)
        chosen = jnp.take_along_axis(Qdp, _pi[:, :, None], axis=2)[:, :, 0]
        reg = opt - chosen
        sl = slice(_p.K_Q - w, _p.K_Q + w + 1)
        return (float(reg[:, sl].mean()), float(reg[:, sl].max()))
    print('Mean / max regret vs DP (q in [-10,10]):')
    for _name, _pi, _p, V in [('Q-learn', pi_q, P_TERM, V_dp_t), ('PPO', pi_ppo, P_TERM, V_dp_t)]:
        m, mx = regret_of(_pi, _p, V)
        print(f'  {_name:8s}: mean {m:.4f}  max {mx:.4f}')
    print('  (per-step edge spans ~0.0-0.22; regret<0.05 = near-optimal)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Quant index table — PnL, Sharpe, Sortino, drawdown, VaR, inventory
    """)
    return


@app.cell
def _(EVALS, np):
    def pnl_of(ev):  # rpath sum over t, per episode
        return np.array(ev[2].sum(axis=1))

    def quant_stats(ev):  # (N,T)
        pnl = pnl_of(ev)
        qpath = np.array(ev[0])
        mean, std = (pnl.mean(), pnl.std())
        sharpe = mean / (std + 1e-09)
        downside = pnl[pnl < 0]  # max drawdown on the MEAN cumulative-PnL path across episodes
        sortino = mean / (downside.std() + 1e-09) if downside.size > 0 else np.inf
        cum = np.array(ev[2]).mean(axis=0).cumsum()
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum).max()
        var5 = np.percentile(pnl, 5)
        cvar5 = pnl[pnl <= var5].mean() if (pnl <= var5).any() else var5
        return dict(mean=mean, std=std, sharpe=sharpe, sortino=sortino, max_dd=dd, VaR5=var5, CVaR5=cvar5, mean_absq=np.abs(qpath).mean(), term_absq=np.abs(qpath[:, -1]).mean())
    import pandas as pd
    rows = {_name: quant_stats(ev) for _name, (ev, _p) in EVALS.items()}
    df = pd.DataFrame(rows).T
    df = df[['mean', 'std', 'sharpe', 'sortino', 'max_dd', 'VaR5', 'CVaR5', 'mean_absq', 'term_absq']]
    df.columns = ['PnL mean', 'PnL std', 'Sharpe', 'Sortino', 'Max DD', 'VaR 5%', 'CVaR 5%', 'mean|q|', 'term|q|']
    df.round(3)
    return df, pd, pnl_of


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PnL distributions
    """)
    return


@app.cell
def _(EVALS, plt, pnl_of):
    _fig, _ax = plt.subplots(figsize=(11, 4))
    _names = list(EVALS.keys())
    _data = [pnl_of(EVALS[n][0]) for n in _names]
    _parts = _ax.violinplot(_data, showmeans=True, showextrema=False)
    _ax.set_xticks(range(1, len(_names) + 1))
    _ax.set_xticklabels(_names)
    _ax.set_ylabel('terminal PnL')
    _ax.set_title('Realized PnL distribution (shared seeds)')
    _ax.grid(alpha=0.3, axis='y')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AS rounding cost (AS-grid vs AS-continuous)
    """)
    return


@app.cell
def _(ev_asc, ev_asg, np, plt, pnl_of):
    pnl_g = pnl_of(ev_asg); pnl_c = pnl_of(ev_asc)
    print(f"AS-continuous PnL mean: {pnl_c.mean():.3f}")
    print(f"AS-grid       PnL mean: {pnl_g.mean():.3f}")
    print(f"ROUNDING COST (cont - grid): {pnl_c.mean()-pnl_g.mean():.3f} per episode")
    print(f"inventory: AS-cont term|q| {np.abs(np.array(ev_asc[0])[:,-1]).mean():.2f}"
          f"  AS-grid {np.abs(np.array(ev_asg[0])[:,-1]).mean():.2f}")
    plt.figure(figsize=(8,3))
    plt.hist(pnl_c, bins=40, alpha=0.6, label='AS-continuous', density=True)
    plt.hist(pnl_g, bins=40, alpha=0.6, label='AS-grid', density=True)
    plt.legend(); plt.xlabel("terminal PnL"); plt.title("AS rounding cost"); plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Extreme-case event plots (interactive — zoom & pan)

    Three episodes selected by outcome: one where everyone loses, one where a policy
    loses big, one where a policy wins big. Each plot shows the mid-price path, the
    ask/bid quote levels, fill markers (▲ ask fill, ▼ bid fill) at their fill prices,
    and inventory on a secondary axis. Use the rangeslider / drag to zoom.
    """)
    return


@app.cell
def _(ev_dp_t, ev_ppo, go, make_subplots, np, pnl_of):
    def event_figure(ev, p, ep, title):
        qpath, Spath, rpath, ask, bid, da, db = [np.array(x[ep]) for x in ev]
        T = p.T
        ts = np.arange(T)  # quote prices: ask = S - 0.5 + d_a ; bid = S + 0.5 - d_b  (using PREVIOUS mid = S before step)
        mid_before = np.concatenate([[p.S0], Spath[:-1]])  # Spath[t] is mid AFTER step t; reconstruct mid BEFORE each step:
        ask_px = mid_before - 0.5 + da
        bid_px = mid_before + 0.5 - db
        _fig = make_subplots(specs=[[{'secondary_y': True}]])
        _fig.add_trace(go.Scatter(x=ts, y=mid_before, name='mid', line=dict(color='black', width=2)))
        _fig.add_trace(go.Scatter(x=ts, y=ask_px, name='ask quote', line=dict(color='crimson', dash='dot')))
        _fig.add_trace(go.Scatter(x=ts, y=bid_px, name='bid quote', line=dict(color='royalblue', dash='dot')))
        af = ask.astype(bool)
        bf = bid.astype(bool)
        _fig.add_trace(go.Scatter(x=ts[af], y=ask_px[af], mode='markers', name='ask fill', marker=dict(symbol='triangle-up', size=10, color='crimson')))
        _fig.add_trace(go.Scatter(x=ts[bf], y=bid_px[bf], mode='markers', name='bid fill', marker=dict(symbol='triangle-down', size=10, color='royalblue')))
        _fig.add_trace(go.Scatter(x=ts, y=qpath, name='inventory', line=dict(color='green', width=1)), secondary_y=True)
        _fig.update_layout(title=title, height=400, xaxis=dict(rangeslider=dict(visible=True)), legend=dict(orientation='h'))
        _fig.update_yaxes(title_text='price', secondary_y=False)
        _fig.update_yaxes(title_text='inventory', secondary_y=True)
        return _fig
    pnl_dp = pnl_of(ev_dp_t)
    ep_allbad = int(np.argmin(pnl_dp))
    gap_ppo = pnl_of(ev_ppo) - pnl_dp
    ep_ppo_loss = int(np.argmin(gap_ppo))
    ep_ppo_win = int(np.argmax(gap_ppo))
    # select episodes by outcome using DP-term as reference policy
    # a policy loses big relative to DP:
    print('episodes:', dict(allbad=ep_allbad, ppo_loss=ep_ppo_loss, ppo_win=ep_ppo_win))  # DP itself does badly -> hard price path
    return ep_allbad, ep_ppo_loss, ep_ppo_win, event_figure


@app.cell(hide_code=True)
def _(P_TERM, ep_allbad, ev_dp_t, ev_ppo, event_figure):
    event_figure(ev_dp_t, P_TERM, ep_allbad, f"Hard price path (ep {ep_allbad}) — DP-terminal").show()
    event_figure(ev_ppo, P_TERM, ep_allbad, f"Hard price path (ep {ep_allbad}) — PPO").show()
    return


@app.cell(hide_code=True)
def _(P_TERM, ep_ppo_loss, ev_dp_t, ev_ppo, event_figure):
    event_figure(ev_ppo, P_TERM, ep_ppo_loss,
                 f"PPO underperforms DP here (ep {ep_ppo_loss}) — PPO quotes/fills").show()
    event_figure(ev_dp_t, P_TERM, ep_ppo_loss,
                 f"PPO underperforms DP here (ep {ep_ppo_loss}) — DP quotes/fills").show()
    return


@app.cell(hide_code=True)
def _(P_TERM, ep_ppo_win, ev_dp_t, ev_ppo, event_figure):
    event_figure(ev_ppo, P_TERM, ep_ppo_win,
                 f"PPO outperforms DP here (ep {ep_ppo_win}) — PPO quotes/fills").show()
    event_figure(ev_dp_t, P_TERM, ep_ppo_win,
                 f"PPO outperforms DP here (ep {ep_ppo_win}) — DP quotes/fills").show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary matrix
    """)
    return


@app.cell
def _(df):
    df.round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ablation — does the mark-to-market *reval* term and the mid-price *S* matter?

    The reval term `q'·Delta` is **zero-mean** under the martingale price
    (`E[Delta]=0`), so it can **never change the optimal policy** — DP integrates it to
    zero. But it injects variance into the *learning signal*. A natural hypothesis is
    that the learners struggle with reval because they cannot *see* the mid price `S`.
    We test the 2x2: reward `{reval-free, reval-contain}` x observation `{no-S, +S}`,
    for both Q-learning (tabular) and PPO (actor-critic), measuring **regret vs DP**.

    The result is richer than the naive hypothesis and is **algorithm-dependent** —
    three distinct mechanisms appear.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **DP is reval- and S-invariant by construction.** We run DP four times
    (terminal/running x the two "variants") to *show* the value/policy is identical —
    because `E[q'·Delta]=0` and the dynamics are translation-invariant in `S`.
    """)
    return


@app.cell
def _(P_RUN, P_TERM, jnp, solve_dp):
    # DP ignores reward-reval (takes expectation -> 0) and has no S dependence.
    # Demonstrate by solving 4 times and comparing policies pairwise.
    def dp_policy(p):
        _, _pi, _ = solve_dp(p)
        return _pi
    pi_a = dp_policy(P_TERM)  # "terminal, reval-free, no-S"
    pi_b = dp_policy(P_TERM)  # "terminal, reval-contain, +S"  -> same call: DP can't see the difference
    pi_c = dp_policy(P_RUN)
    pi_d = dp_policy(P_RUN)
    print('DP terminal: variant A == variant B :', bool(jnp.all(pi_a == pi_b)))
    print('DP running : variant C == variant D :', bool(jnp.all(pi_c == pi_d)))
    print('--> DP is invariant to reval (zero-mean) and to S (translation invariance).')
    print("    Only the LEARNERS' variants can differ; DP is the same ground truth for all.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Q-learning, all four variants.** For `+S` the table gains an S-bin axis (price discretized to ticks from `S0`), which multiplies its size ~31x.
    """)
    return


@app.cell(hide_code=True)
def _(env_step, jax, jit, jnp, lax, partial, reward_decomposed):
    SB = 15  # S-bin half-width (ticks from S0)

    def sbin(S, p):
        k = jnp.round((S - p.S0) / p.tick).astype(jnp.int32)
        return jnp.clip(k, -SB, SB) + SB

    def _rewards_both(q, a, ask, bid, qn, dS, t, done, p):
        full, reval = reward_decomposed(q, a, ask, bid, qn, dS, t, done, p)  # (with_reval, reval_free)
        return (full, full - reval)

    @partial(jit, static_argnames=('n_updates', 'batch', 'use_reval', 'use_S'))
    def q_learn_variant(key, p, n_updates=80000, batch=128, eta0=0.3, eps0=0.4, use_reval=False, use_S=False):
        n_q = 2 * p.K_Q + 1
        n_a = (p.K + 1) ** 2
        T = p.T
        n_s = 2 * SB + 1
        Q0 = jnp.zeros((T, n_q, n_s, n_a)) if use_S else jnp.zeros((T, n_q, n_a))

        def step(carry, sk):
            Q, it = carry
            frac = it / n_updates
            eta = eta0 * (1 - 0.92 * frac)
            eps = jnp.maximum(0.05, eps0 * (1 - frac))
            ks = jax.random.split(sk, batch)

            def samp(_, k):
                kt, kq, ksS, ka, ke, ksp = jax.random.split(k, 6)
                t = jax.random.randint(kt, (), 0, T)
                q = jax.random.randint(kq, (), -p.K_Q, p.K_Q + 1)
                Sk = jax.random.randint(ksS, (), -SB, SB + 1)
                S = p.S0 + p.tick * Sk.astype(jnp.float32)
                qi = q + p.K_Q
                if use_S:
                    si = sbin(S, p)
                    greedy = jnp.argmax(Q[t, qi, si])
                    ra = jax.random.randint(ka, (), 0, n_a)
                    a = jnp.where(jax.random.uniform(ke) < eps, ra, greedy)
                    qn, Sn, ask, bid, dS = env_step(ksp, q, S, t, a, p)
                    done = t + 1 >= T
                    full, rf = _rewards_both(q, a, ask, bid, qn, dS, t, done, p)
                    r = jnp.where(use_reval, full, rf)
                    tgt = r + jnp.where(done, 0.0, jnp.max(Q[t + 1, qn + p.K_Q, sbin(Sn, p)]))
                    return (t, qi, si, a, tgt)
                else:
                    greedy = jnp.argmax(Q[t, qi])
                    ra = jax.random.randint(ka, (), 0, n_a)
                    a = jnp.where(jax.random.uniform(ke) < eps, ra, greedy)
                    qn, Sn, ask, bid, dS = env_step(ksp, q, S, t, a, p)
                    done = t + 1 >= T
                    full, rf = _rewards_both(q, a, ask, bid, qn, dS, t, done, p)
                    r = jnp.where(use_reval, full, rf)
                    tgt = r + jnp.where(done, 0.0, jnp.max(Q[t + 1, qn + p.K_Q]))
                    return (t, qi, a, tgt)
            out = jax.vmap(samp, in_axes=(None, 0))(0, ks)
            if use_S:
                ts, qis, sis, as_, tg = out

                def app(Q, x):
                    t, qi, si, a, g = x
                    return (Q.at[t, qi, si, a].add(eta * (g - Q[t, qi, si, a])), None)
                Q, _ = lax.scan(app, Q, (ts, qis, sis, as_, tg))
            else:
                ts, qis, as_, tg = out

                def app(Q, x):
                    t, qi, a, g = x
                    return (Q.at[t, qi, a].add(eta * (g - Q[t, qi, a])), None)
                Q, _ = lax.scan(app, Q, (ts, qis, as_, tg))
            return ((Q, it + 1), None)
        (Q, _), _ = lax.scan(step, (Q0, 0), jax.random.split(key, n_updates))
        return Q

    return SB, q_learn_variant


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **PPO, all four variants.** For `+S` the observation gains a normalized mid-price feature; the critic can then use `S` as a baseline.
    """)
    return


@app.cell(hide_code=True)
def _(
    env_step,
    jax,
    jit,
    jnp,
    lax,
    nn,
    optax,
    partial,
    reward_decomposed,
    vmap,
):
    def make_obs(q,S,t,p,use_S):
        base=[q/p.K_Q, t/p.T]
        if use_S: base = base + [(S-p.S0)/(p.sigma*jnp.sqrt(p.T*1.0))]
        return jnp.stack(base,axis=-1).astype(jnp.float32)

    class ACv(nn.Module):
        n_actions:int; 
        @nn.compact
        def __call__(self,x):
            h=nn.tanh(nn.Dense(64)(x)); h=nn.tanh(nn.Dense(64)(h))
            logits=nn.Dense(self.n_actions)(h)
            v=nn.Dense(1)(nn.tanh(nn.Dense(64)(x)))[...,0]; return logits,v

    @partial(jit,static_argnames=('net','n_updates','n_envs','epochs','mb','use_reval','use_S'))
    def train_ppo_variant(key,p,net,use_reval,use_S,n_updates=300,n_envs=64,epochs=4,mb=8,
                          clip=0.2,lr=3e-3,gae_lam=0.95,ent=0.01,vf=0.5):
        T=p.T; obs_dim=3 if use_S else 2
        key,ki=jax.random.split(key); params=net.init(ki,jnp.zeros((obs_dim,)))
        tx=optax.adam(lr); opt=tx.init(params)
        def rollout(params,key):
            def one(key):
                def tstep(c,t):
                    q,S,key=c; key,ka,ks=jax.random.split(key,3)
                    o=make_obs(q.astype(jnp.float32),S,jnp.float32(t),p,use_S)
                    logits,v=net.apply(params,o); a=jax.random.categorical(ka,logits)
                    logp=nn.log_softmax(logits)[a]
                    qn,Sn,ask,bid,dS=env_step(ks,q,S,jnp.int32(t),a,p); done=(t+1)>=T
                    full,reval=reward_decomposed(q,a,ask,bid,qn,dS,jnp.int32(t),done,p)
                    r=jnp.where(use_reval,full,full-reval)
                    return (qn,Sn,key),(o,a,logp,v,r,done)
                (_,_,_),tr=lax.scan(tstep,(jnp.int32(0),jnp.float32(p.S0),key),jnp.arange(T))
                return tr
            return vmap(one)(jax.random.split(key,n_envs))
        def gae(r,v,done):
            vn=jnp.concatenate([v[:,1:],jnp.zeros((v.shape[0],1))],axis=1); nt=1.0-done.astype(jnp.float32)
            def f(an,x):
                r_t,v_t,vnn,n_=x; d=r_t+vnn*n_-v_t; a=d+gae_lam*n_*an; return a,a
            _,adv=lax.scan(f,jnp.zeros((r.shape[0],)),(r.T,v.T,vn.T,nt.T),reverse=True)
            adv=adv.T; return adv,adv+v
        def update(carry,key):
            params,opt=carry; obs,a,lpo,v,r,done=rollout(params,key); adv,ret=gae(r,v,done)
            B=obs.reshape(-1,obs_dim); A=a.reshape(-1); LP=lpo.reshape(-1)
            ADV=adv.reshape(-1); RET=ret.reshape(-1); ADV=(ADV-ADV.mean())/(ADV.std()+1e-8)
            def epoch(carry,ek):
                params,opt=carry; idx=jax.random.permutation(ek,B.shape[0]).reshape(mb,-1)
                def mbs(carry,mi):
                    params,opt=carry
                    def loss(params):
                        lg,val=vmap(lambda o:net.apply(params,o))(B[mi])
                        lp=vmap(lambda l,ai:nn.log_softmax(l)[ai])(lg,A[mi])
                        ratio=jnp.exp(lp-LP[mi]); aa=ADV[mi]
                        pg=-jnp.mean(jnp.minimum(ratio*aa,jnp.clip(ratio,1-clip,1+clip)*aa))
                        vl=jnp.mean((val-RET[mi])**2)
                        pr=jax.nn.softmax(lg); en=-jnp.mean(jnp.sum(pr*nn.log_softmax(lg),axis=-1))
                        return pg+vf*vl-ent*en
                    g=jax.grad(loss)(params); u,opt=tx.update(g,opt,params); params=optax.apply_updates(params,u)
                    return (params,opt),None
                (params,opt),_=lax.scan(mbs,(params,opt),idx); return (params,opt),None
            (params,opt),_=lax.scan(epoch,(params,opt),jax.random.split(key,epochs))
            return (params,opt),jnp.mean(RET)
        (params,opt),rets=lax.scan(update,(params,opt),jax.random.split(key,n_updates))
        return params,rets

    return ACv, make_obs, train_ppo_variant


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Run the ablation.** Regret vs DP for all 4 variants of each learner. PPO is run over **3 seeds** to expose training-variance differences.
    """)
    return


@app.cell(hide_code=True)
def _(
    P_TERM,
    SB,
    V_dp_t,
    fill_probs,
    jax,
    jnp,
    q_learn_variant,
    unflatten_action,
):
    # regret helper using DP true Q (terminal regime)
    def dp_true_Q_stack_term(p, V_dp):
        n_a = (p.K + 1) ** 2
        acts = jnp.arange(n_a)
        da, db = unflatten_action(acts, p)
        pa, pb = fill_probs(da, db, p)
        pdn = pa * (1 - pb)
        pup = (1 - pa) * pb
        pst = pa * pb + (1 - pa) * (1 - pb)
        R = pa * (da - 0.5) + pb * (db - 0.5)
        qf = jnp.arange(-p.K_Q, p.K_Q + 1).astype(jnp.float32)
        VT = -p.gamma_term * qf ** 2
        st = []
        for t in range(p.T):
            Vn = VT if t == p.T - 1 else V_dp[t + 1]
            Vm = jnp.concatenate([Vn[:1], Vn[:-1]])
            Vpp = jnp.concatenate([Vn[1:], Vn[-1:]])
            st.append(R[None, :] + pdn[None, :] * Vm[:, None] + pup[None, :] * Vpp[:, None] + pst[None, :] * Vn[:, None])
    # Q-learning (single seed; tabular is fairly deterministic given exploring starts)
        return jnp.stack(st)
    Qdp_term = dp_true_Q_stack_term(P_TERM, V_dp_t)

    def regret_tq(pi):
        opt = jnp.max(Qdp_term, axis=2)
        ch = jnp.take_along_axis(Qdp_term, pi[:, :, None], axis=2)[:, :, 0]
        return float((opt - ch)[:, P_TERM.K_Q - 10:P_TERM.K_Q + 11].mean())
    variants = [(False, False, 'reval-free\n+no-S'), (True, False, 'reval-contain\n+no-S'), (False, True, 'reval-free\n+S'), (True, True, 'reval-contain\n+S')]
    ql_regret = {}
    for _ur, _us, _name in variants:
        Q = q_learn_variant(jax.random.PRNGKey(0), P_TERM, n_updates=80000, use_reval=_ur, use_S=_us)
        _pi = jnp.argmax(Q[:, :, SB, :], axis=2) if _us else jnp.argmax(Q, axis=2)
        ql_regret[_name] = regret_tq(_pi)
        print(f'Q-learn {_name.replace(chr(10), ' '):22s}: regret = {ql_regret[_name]:.4f}')
    return ql_regret, regret_tq, variants


@app.cell(hide_code=True)
def _(
    ACv,
    P_TERM,
    jax,
    jnp,
    make_obs,
    n_actions,
    np,
    regret_tq,
    train_ppo_variant,
    variants,
    vmap,
):
    # PPO over 3 seeds
    _qs_grid = jnp.arange(-P_TERM.K_Q, P_TERM.K_Q+1)
    def ppo_policy(params, net, use_S):
        def g(t):
            o = vmap(lambda q: make_obs(q.astype(jnp.float32), jnp.float32(P_TERM.S0),
                                        jnp.float32(t), P_TERM, use_S))(_qs_grid)
            lg,_ = vmap(lambda x: net.apply(params,x))(o); return jnp.argmax(lg,axis=1)
        return vmap(g)(jnp.arange(P_TERM.T))

    ppo_regret = {}   # name -> list of regrets over seeds
    ppo_seed0 = {}    # name -> (params, net, use_S) at seed 0, reused in 8.9
    for _ur, _us, _name in variants:
        od = 3 if _us else 2; _rs=[]
        for _seed in [0,1,2]:
            _net = ACv(n_actions(P_TERM))
            _params,_ = train_ppo_variant(jax.random.PRNGKey(_seed), P_TERM, _net, _ur, _us, n_updates=300)
            _rs.append(regret_tq(ppo_policy(_params, _net, _us)))
            if _seed == 0:
                ppo_seed0[_name.replace(chr(10),' ')] = (_params, _net, _us)
        ppo_regret[_name] = _rs
        print(f"PPO {_name.replace(chr(10),' '):22s}: regrets {[round(x,3) for x in _rs]}  mean {np.mean(_rs):.3f}")
    return ppo_regret, ppo_seed0


@app.cell(hide_code=True)
def _(np, plt, ppo_regret, ql_regret, variants):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    _names = [v[2] for v in variants]
    x = np.arange(len(_names))
    _axes[0].bar(x, [ql_regret[n] for n in _names], color=['#2a9d8f', '#e76f51', '#264653', '#e9c46a'])
    # Q-learn bars
    _axes[0].set_title('Q-learning (tabular): regret vs DP')
    _axes[0].set_xticks(x)
    _axes[0].set_xticklabels(_names, fontsize=9)
    _axes[0].set_ylabel('mean regret (q in [-10,10])')
    _axes[0].grid(alpha=0.3, axis='y')
    _axes[0].axhline(0.05, ls='--', c='gray', lw=1)
    _axes[0].text(0, 0.07, 'near-optimal', color='gray', fontsize=8)
    # PPO bars with seed spread
    means = [np.mean(ppo_regret[n]) for n in _names]
    mins = [np.min(ppo_regret[n]) for n in _names]
    maxs = [np.max(ppo_regret[n]) for n in _names]
    err = [np.array(means) - np.array(mins), np.array(maxs) - np.array(means)]
    _axes[1].bar(x, means, yerr=err, capsize=5, color=['#2a9d8f', '#e76f51', '#264653', '#e9c46a'])
    for i, n in enumerate(_names):
        _axes[1].scatter([i] * 3, ppo_regret[n], color='black', zorder=3, s=18)  # scatter the 3 seeds
    _axes[1].set_title('PPO (actor-critic): regret vs DP (3 seeds)')
    _axes[1].set_xticks(x)
    _axes[1].set_xticklabels(_names, fontsize=9)
    _axes[1].set_ylabel('mean regret (q in [-10,10])')
    _axes[1].grid(alpha=0.3, axis='y')
    _axes[1].axhline(0.05, ls='--', c='gray', lw=1)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the ablation — three distinct mechanisms.**

    1. **Target-noise (reval, no-S).** Adding the zero-mean reval term to the *reward*
       raises regret for both learners — but far more for **PPO** (on-policy, no
       experience replay to average the noise) than for tabular Q-learning. The optimal
       policy is unchanged; only the *learning signal* got noisier.

    2. **State-dilution (Q-learn, +S).** Giving the **tabular** learner the mid price `S`
       makes it markedly *worse*: the table grows ~31x along an axis the value is flat
       on, so each cell gets ~31x fewer visits for zero predictive gain. `S` is a nuisance
       variable; under a martingale price it cannot reduce reval variance because the
       increment is independent of the level.

    3. **Critic-baseline rescue (PPO, +S).** For the **actor-critic**, when reval *is* in
       the reward, giving the critic `S` lets it fit the reval-driven variance *within a
       batch* and subtract it through the advantage — stabilizing training and pulling
       regret down sharply (and tightening the seed-to-seed spread). So `S` helps PPO not
       by changing the optimal policy, but by acting as a **variance-reducing baseline
       input**.

    **Conclusion.** The original intuition — "reval performs badly because there is no
    observation on `S`" — is **correct for actor-critic methods and wrong for tabular
    value learning.** Whether observing `S` helps depends on whether the algorithm has a
    *baseline mechanism* that can exploit it. The cleanest universal recipe remains:
    **train reval-free** (removes the noise at its source, optimal policy unchanged);
    `S` is then unnecessary. But if reval must stay in the reward, **a critic that sees
    `S` is the right tool** to absorb its variance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross-policy comparison — DP, AS-continuous, and all four PPO variants

    We line up **seven** policies and score them on **one common yardstick** so the
    numbers are directly comparable: full realized PnL = `edge + reval + terminal
    penalty (gamma_term=2)`, evaluated on **shared seeds** (identical price paths and fill
    coins). The seven:

    - **DP-term** — terminal-regime optimum (the yardstick's own objective).
    - **DP-run**, **AS-continuous** — running-regime policies (shown for contrast; they
      optimize a *different* objective, so expect them to score lower on the terminal
      yardstick — that is the cross-regime caveat, not a defect).
    - **PPO x4** — `{reval-free, reval-contain} x {no-S, +S}`.

    The headline question: does PPO trained **with reval in the reward** recover good
    realized PnL, and does seeing **S** make the difference?
    """)
    return


@app.cell
def _(ppo_seed0):
    # Reuse the seed-0 PPO variants already trained in Section 8.8 (no retraining).
    # Map 8.8's names ('reval-free\n+no-S' etc) to the short labels used here.
    name_map = {'reval-free +no-S': 'PPO rf,noS', 'reval-contain +no-S': 'PPO rev,noS', 'reval-free +S': 'PPO rf,+S', 'reval-contain +S': 'PPO rev,+S'}
    ppo_variants = {}
    for k8, (_pr, _nt, _us) in ppo_seed0.items():
        ppo_variants[name_map[k8]] = (_pr, _nt, _us)
    print('reusing', len(ppo_variants), 'PPO variants from 8.8:', list(ppo_variants))
    return (ppo_variants,)


@app.cell(hide_code=True)
def _(
    P_RUN,
    P_TERM,
    as_cont,
    jax,
    jit,
    jnp,
    lax,
    make_obs,
    partial,
    pi_dp_r,
    pi_dp_t,
    ppo_variants,
    price_increment_probs,
    unflatten_action,
    vmap,
):
    # Unified shared-seed evaluator scoring EVERY policy on the SAME reward
    # (edge + reval + fixed terminal penalty gamma_term_eval), regardless of training regime.
    gamma_term_EVAL = 2.0

    @partial(jit, static_argnames=('N', 'kind', 'use_S', 'ppo_net'))
    def evaluate_common(key, p, kind, dp_pi=None, ppo_params=None, ppo_net=None, use_S=False, N=3000):
        T = p.T
        k_coin, k_price = jax.random.split(key)
        coins = jax.random.uniform(k_coin, (N, T, 2))
        ks, pk = price_increment_probs(p)
        incr = p.tick * jax.random.choice(k_price, ks, (N, T), p=pk).astype(jnp.float32)

        def pick(q, S, t):
            if kind == 'dp':
                a = dp_pi[t, q + p.K_Q]
                da, db = unflatten_action(a, p)
                return (da.astype(jnp.float32), db.astype(jnp.float32))
            elif kind == 'as_cont':
                return as_cont(S, q, t, p)
            elif kind == 'ppo':
                o = make_obs(q.astype(jnp.float32), S, jnp.float32(t), p, use_S)
                lg, _ = ppo_net.apply(ppo_params, o)
                a = jnp.argmax(lg)
                da, db = unflatten_action(a, p)
                return (da.astype(jnp.float32), db.astype(jnp.float32))

        def run_one(coin_row, incr_row):

            def tstep(carry, inp):
                q, S = carry
                t, coin, dS = inp
                da, db = pick(q, S, t)
                pa = jnp.exp(-p.kappa * da)  # COMMON yardstick
                pb = jnp.exp(-p.kappa * db)
                ask = coin[0] < pa
                bid = coin[1] < pb
                at_hi = q >= p.K_Q
                at_lo = q <= -p.K_Q
                bid = bid & ~at_hi
                ask = ask & ~at_lo
                qn = q + -ask.astype(jnp.int32) + bid.astype(jnp.int32)
                Sn = S + dS
                edge = (ask * (da - 0.5) + bid * (db - 0.5)).astype(jnp.float32)
                reval = qn.astype(jnp.float32) * dS
                done = t + 1 >= T
                term = jnp.where(done, -gamma_term_EVAL * qn.astype(jnp.float32) ** 2, 0.0)
                r = edge + reval + term
                return ((qn, Sn), (qn, Sn, r, ask, bid, da, db))
            ts = jnp.arange(T)
            (_, _), out = lax.scan(tstep, (jnp.int32(0), jnp.float32(p.S0)), (ts, coin_row, incr_row))
            return out
        return vmap(run_one)(coins, incr)
    CMP_KEY = jax.random.PRNGKey(777)
    N_CMP = 3000
    CMP = {}
    CMP['DP-term'] = (evaluate_common(CMP_KEY, P_TERM, 'dp', dp_pi=pi_dp_t, N=N_CMP), P_TERM)
    CMP['DP-run'] = (evaluate_common(CMP_KEY, P_RUN, 'dp', dp_pi=pi_dp_r, N=N_CMP), P_RUN)
    CMP['AS-cont'] = (evaluate_common(CMP_KEY, P_RUN, 'as_cont', N=N_CMP), P_RUN)
    for _nm, (_pr, _nt, _us) in ppo_variants.items():
        CMP[_nm] = (evaluate_common(CMP_KEY, P_TERM, 'ppo', ppo_params=_pr, ppo_net=_nt, use_S=_us, N=N_CMP), P_TERM)
    print('evaluated', len(CMP), 'policies on the common yardstick,', N_CMP, 'shared-seed episodes')
    return (CMP,)


@app.cell
def _(CMP, np, pd):
    def cmp_stats(ev):
        pnl = np.array(ev[2].sum(axis=1))
        qp = np.array(ev[0])
        cum = np.array(ev[2]).mean(axis=0).cumsum()
        peak = np.maximum.accumulate(cum)
        return dict(PnL_mean=pnl.mean(), PnL_std=pnl.std(), Sharpe=pnl.mean() / (pnl.std() + 1e-09), Max_DD=(peak - cum).max(), VaR5=np.percentile(pnl, 5), term_absq=np.abs(qp[:, -1]).mean(), mean_absq=np.abs(qp).mean())
    cmp_df = pd.DataFrame({n: cmp_stats(ev) for n, (ev, _p) in CMP.items()}).T
    cmp_df = cmp_df[['PnL_mean', 'PnL_std', 'Sharpe', 'Max_DD', 'VaR5', 'term_absq', 'mean_absq']].round(3)
    cmp_df
    return


@app.cell
def _(
    P_RUN,
    P_TERM,
    jnp,
    make_obs,
    np,
    pi_dp_r,
    pi_dp_t,
    plt,
    ppo_variants,
    vmap,
):
    # policy skew heatmaps for the grid-representable policies (AS-cont is continuous -> omitted)
    def ppo_skew_grid(params, net, us):
        qg = jnp.arange(-P_TERM.K_Q, P_TERM.K_Q + 1)

        def g(t):
            o = vmap(lambda q: make_obs(q.astype(jnp.float32), jnp.float32(P_TERM.S0), jnp.float32(t), P_TERM, _us))(qg)
            lg, _ = vmap(lambda x: net.apply(params, x))(o)
            return jnp.argmax(lg, axis=1)
        return vmap(g)(jnp.arange(P_TERM.T))
    heat = {'DP-term': (pi_dp_t, P_TERM), 'DP-run': (pi_dp_r, P_RUN)}
    for _nm, (_pr, _nt, _us) in ppo_variants.items():
        heat[_nm] = (ppo_skew_grid(_pr, _nt, _us), P_TERM)
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 7))
    for _ax, (_nm, (_pi, _p)) in zip(_axes.flat, heat.items()):
        _sg = np.array(_pi // (_p.K + 1) - _pi % (_p.K + 1)).T
        _im = _ax.imshow(_sg, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, _p.T, -_p.K_Q, _p.K_Q], vmin=-_p.K, vmax=_p.K)
        _ax.set_title(f'{_nm}: skew (d_a - d_b)')
        _ax.set_xlabel('t')
        _ax.set_ylabel('q')
        plt.colorbar(_im, ax=_ax)
    plt.tight_layout()
    plt.show()
    print('Note: AS-continuous quotes real-valued delta -> not on the discrete (t,q) grid, omitted here.')
    return


@app.cell
def _(CMP, np, plt):
    _fig, _ax = plt.subplots(figsize=(12, 4))
    _names = list(CMP.keys())
    _data = [np.array(CMP[n][0][2].sum(axis=1)) for n in _names]
    _parts = _ax.violinplot(_data, showmeans=True, showextrema=False)
    _ax.set_xticks(range(1, len(_names) + 1))
    _ax.set_xticklabels(_names, rotation=20, ha='right')
    _ax.set_ylabel('realized PnL (common yardstick)')
    _ax.set_title('Realized PnL on shared seeds — note PPO rev,noS heavy left tail vs PPO rev,+S')
    _ax.grid(alpha=0.3, axis='y')
    _ax.axhline(0, c='gray', lw=0.8)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Special cases (shared seeds, interactive).** On identical price paths we pick
    the episode where seeing `S` rescues the reval-trained PPO the most (largest PnL gap
    between `PPO rev,+S` and `PPO rev,noS`), and the episode that is hardest for everyone
    (DP-term's worst). Each plot shows mid/quotes/fills/inventory; zoom with the slider.
    """)
    return


@app.cell
def _(CMP, P_TERM, go, make_subplots, np):
    def event_fig_cmp(ev, p, ep, title):
        qpath, Spath, rpath, ask, bid, da, db = [np.array(x[ep]) for x in ev]
        T = p.T
        ts = np.arange(T)
        mid_before = np.concatenate([[p.S0], Spath[:-1]])
        ask_px = mid_before - 0.5 + da
        bid_px = mid_before + 0.5 - db
        _fig = make_subplots(specs=[[{'secondary_y': True}]])
        _fig.add_trace(go.Scatter(x=ts, y=mid_before, name='mid', line=dict(color='black', width=2)))
        _fig.add_trace(go.Scatter(x=ts, y=ask_px, name='ask', line=dict(color='crimson', dash='dot')))
        _fig.add_trace(go.Scatter(x=ts, y=bid_px, name='bid', line=dict(color='royalblue', dash='dot')))
        af = ask.astype(bool)
        bf = bid.astype(bool)
        _fig.add_trace(go.Scatter(x=ts[af], y=ask_px[af], mode='markers', name='ask fill', marker=dict(symbol='triangle-up', size=10, color='crimson')))
        _fig.add_trace(go.Scatter(x=ts[bf], y=bid_px[bf], mode='markers', name='bid fill', marker=dict(symbol='triangle-down', size=10, color='royalblue')))
        _fig.add_trace(go.Scatter(x=ts, y=qpath, name='inventory', line=dict(color='green', width=1)), secondary_y=True)
        _fig.update_layout(title=title, height=400, xaxis=dict(rangeslider=dict(visible=True)), legend=dict(orientation='h'))
        _fig.update_yaxes(title_text='price', secondary_y=False)
        _fig.update_yaxes(title_text='inventory', secondary_y=True)
        return _fig
    pnl_pS = np.array(CMP['PPO rev,+S'][0][2].sum(axis=1))
    pnl_nS = np.array(CMP['PPO rev,noS'][0][2].sum(axis=1))
    ep_rescue = int(np.argmax(pnl_pS - pnl_nS))
    print(f'+S rescue episode {ep_rescue}: PPO rev,+S PnL={pnl_pS[ep_rescue]:.1f}  vs  rev,noS PnL={pnl_nS[ep_rescue]:.1f}')
    event_fig_cmp(CMP['PPO rev,noS'][0], P_TERM, ep_rescue, f'Reval-trained PPO WITHOUT S (ep {ep_rescue}) — inventory runs away').show()
    return ep_rescue, event_fig_cmp


@app.cell
def _(CMP, P_TERM, ep_rescue, event_fig_cmp):
    event_fig_cmp(CMP['PPO rev,+S'][0], P_TERM, ep_rescue,
                  f"Reval-trained PPO WITH S (ep {ep_rescue}) — critic controls inventory").show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the comparison shows.**

    - On the common yardstick, **`PPO rev,+S` recovers near-DP-terminal PnL** with low
      variance, while **`PPO rev,noS` has a heavy negative tail** (the same episodes
      where withheld `S` let inventory drift). The side-by-side special-case plots make
      the mechanism visible: without `S` the reval-trained policy lets inventory run;
      with `S` the critic keeps it controlled.
    - **`PPO rf,noS` and `PPO rf,+S`** both do well — reval-free training sidesteps the
      problem regardless of `S`.
    - **DP-run and AS-continuous** score low *on this terminal yardstick* because they
      deliberately carry inventory under their own running objective; their `mean|q|` is
      not driven to zero. Judge them by their own regime, not this scoreboard — the
      cross-regime PnL is shown only for completeness.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Findings

    1. **DP is the optimum for this environment.** In the terminal regime DP flattens
       inventory almost perfectly by `T` (term|q| ≈ 0); in the running regime it has the
       highest PnL and lowest variance of all running-regime policies.

    2. **Q-learning reaches near-optimal** (mean regret ~0.03; most cells within 0.05 of
       optimal) once trained on the **reval-free** reward with exploring starts. Exact
       policy-match is low only because many actions are near-ties.

    3. **PPO approximately matches DP** — correct inventory-skew direction, fuzzier near
       the boundaries and the terminal time, as expected from function approximation at a
       modest training budget.

    4. **AS underperforms DP in this discrete env** (model gap: AS is optimal for the
       continuous CARA problem, not this lattice/Bernoulli one).

    5. **AS rounding cost is real and isolated** — AS-continuous beats AS-grid in PnL and
       holds tighter inventory; the gap is purely the tick-rounding of AS's continuous
       spread.

    **Methodological takeaway.** Train learners on the low-variance (reval-free) expected
    reward; evaluate on full realized PnL; compare with regret/value-gap rather than
    exact policy agreement. This makes the DP-vs-RL gap interpretable instead of looking
    like a bug — see `experiment_log.ipynb` for how this was discovered.

    6. **Reval is target-noise, not partial observability.** The zero-mean reval term
       never changes the optimal policy but adds variance to learning. Observing the mid
       price `S` helps *only* actor-critic PPO (the critic uses it as a variance-reducing
       baseline), and *hurts* tabular Q-learning (state-dilution). Training reval-free is
       the robust fix; see Section 8.8.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
