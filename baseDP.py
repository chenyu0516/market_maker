import marimo

__generated_with = "0.23.10"
app = marimo.App()


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



    | Regime | Inventory control | Policies |
    |---|---|---|
    | Terminal | $-\gamma_{term}\,q_T^2$ at the end | DP, Q-learning, PPO |
    | Running | $-\gamma_{run} \, \sigma^2 \, q_t^2$ each step | DP, Qlearn, AS-grid, AS-continuous |
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
    return jax, jit, jnp, lax, np, partial, plt, struct, vmap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Setup
    ### Env mechanism
    #### **Mid price movement**
    * The initial price is $S_0 = 100.5$.
    * The mid price lives on **half-integers**, while the maker's quotes are placed on **integers**.
    * The mid size movement is

    $$S_{t+1} = S_t + \tau \cdot \operatorname{round}\!\left(\frac{\sigma Z_t}{\tau}\right), \qquad Z_t \overset{\text{iid}}{\sim} \mathcal{N}(0,1),$$

    > Where,
    $\tau = 1.$ is the tick size
    $\sigma$ is the per-step **standard deviation** of the price (not the variance)
    The increment $\Delta_{S} = S_{t+1} - S_t = k\tau$ is a lattice variable with probability distribution denoted by **standard normal CDF**, $\Phi$

    > $$p_k := \Pr[\Delta_{S} = k\tau]=\Phi\!\left(\frac{\tau(k + \tfrac{1}{2})}{\sigma}\right)-\Phi\!\left(\frac{\tau(k - \tfrac{1}{2})}{\sigma}\right),$$

    > $$\Phi(z) = \tfrac{1}{2}\left[\,1+\operatorname{erf}\!\left(\frac{z}{\sqrt{2}}\right)\right].$$

    * The half-integer lattice of mid price ensure the gap between the mid and the nearest integer quote is always $\tfrac{1}{2} \tau$.
    * It has zero-drift property

    $$\mu_{\Delta_{S}} := \mathbb{E}[\Delta_{S}] = \tau \sum_k k\, p_k = 0.$$

    #### **Order exectution**
    Each side fills independently with an intensity that decays in its half-spread:

    $$P_{\text{fill}}^a = A\, e^{-\kappa \delta_a}\, dt, \qquad
    P_{\text{fill}}^b = A\, e^{-\kappa \delta_b}\, dt.$$

    **Calibration of $A$.** The most aggressive quote ($\delta = 0$), sitting half a tick
    inside the mid, fills with certainty: $P_{\text{fill}}(0) = 1$. Imposing this gives

    $$A\, dt = 1 \quad\Longrightarrow\quad A = \frac{1}{dt},$$

    so the $dt$ cancels and the fill probability is simply

    $$\boxed{\,P_{\text{fill}}(\delta) = e^{-\kappa \delta}\,.}$$

    This is automatically in $[0,1]$ for $\delta \ge 0$, $\kappa > 0$. Write
    $p_a := e^{-\kappa \delta_a}$, $p_b := e^{-\kappa \delta_b}$ for brevity.
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
        # action of (2,) => Discrete((p.K+1)^2)
        return (a // (p.K + 1), a % (p.K + 1))

    def fill_probs(d_a, d_b, p):
        return (jnp.exp(-p.kappa * d_a), jnp.exp(-p.kappa * d_b))

    def expected_reward(a, p):
        """E[edge] = p_a(d_a-1/2) + p_b(d_b-1/2). Action-only (symmetric increment)."""
        d_a, d_b = unflatten_action(a, p)
        p_a, p_b = fill_probs(d_a, d_b, p)
        return p_a * (d_a - 0.5) + p_b * (d_b - 0.5)

    def Phi(z):
    # regime configs
        return 0.5 * (1.0 + jax.scipy.special.erf(z / jnp.sqrt(2.0)))

    def price_increment_probs(p):
        ks = jnp.arange(-p.max_dk, p.max_dk + 1)
        pk = Phi(p.tick * (ks + 0.5) / p.sigma) - Phi(p.tick * (ks - 0.5) / p.sigma)
        return (ks, pk / pk.sum())

    P_TERM = MMParams(sigma=1.0, kappa=1.5, gamma_term=2.0, gamma_run=0.0, T=30) # Only terminal penalty
    P_RUN = MMParams(sigma=1.0, kappa=1.5, gamma_term=0.0, gamma_run=0.002, T=30) # Only running penalty
    P_B = MMParams(sigma=1.0, kappa=1.5, gamma_term=2.0, gamma_run=0.002, T=30) # Both
    print('terminal penalty regime:', P_TERM)
    print('running penalty regime :', P_RUN)
    print('both included regime : ',  P_B)
    return (
        MMParams,
        P_B,
        P_RUN,
        P_TERM,
        expected_reward,
        fill_probs,
        price_increment_probs,
        unflatten_action,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reward
    The reward at time $t$ has two component, wealth increment and penalty of action
    $$r_t = r_{w, \, t} + r_{p, \, t}
    #### **Wealth and one-step PnL**

    Mark-to-market wealth is cash plus inventory valued at the current mid:

    $$W_t = c_t + q_t S_t.$$

    The one-step reward is the change in wealth:

    $$r_{w, t} = W_{t+1} - W_t = (c_{t+1} - c_t) + (q_{t+1} S_{t+1} - q_t S_t).$$

    The first term is the cash flow $dc_t$. Decomposing the inventory-value change with
    $\Delta q_t = q_{t+1} - q_t$ and $\Delta_t = S_{t+1} - S_t$,

    $$q_{t+1} S_{t+1} - q_t S_t = q_{t+1}\Delta_t + \Delta q_t\, S_t,$$

    so

    $$r_{w, \, t} = dc_t + \Delta q_t\, S_t + q_{t+1}\Delta_t.$$

    Expanding $dc_t + \Delta q_t\, S_t$ for each fill configuration, the absolute price
    $S_t$ cancels every time. Recall the ask fills at $S_t - \tfrac{1}{2} + \delta_a$ and
    the bid fills at $S_t + \tfrac{1}{2} - \delta_b$.

    | Outcome        | $\Delta q_t$ | $dc_t$ | $dc_t + \Delta q_t S_t$ |
    | -------------- | :----------: | --------------------------------- | ----------------------------- |
    | Ask only fills |     $-1$     | $+(S_t - \tfrac{1}{2} + \delta_a)$ | $\delta_a - \tfrac{1}{2}$ |
    | Bid only fills |     $+1$     | $-(S_t + \tfrac{1}{2} - \delta_b)$ | $\delta_b - \tfrac{1}{2}$ |
    | Both fill      |      $0$     | $(\delta_a + \delta_b - 1)$ | $\delta_a + \delta_b - 1$ |
    | Neither fills  |      $0$     | $0$ | $0$ |

    The reward reduces to

    $$r_{w, \, t} = \underbrace{\Big[\mathbb{1}_{\text{ask}}\big(\delta_a - \tfrac{1}{2}\big)
    +\mathbb{1}_{\text{bid}}\big(\delta_b - \tfrac{1}{2}\big)\Big]}_{\text{spread capture}}
    \;+\; \underbrace{q_{t+1}\,\Delta_t}_{\text{reval}},$$

    with no dependence on $S_t$ or $c_t$.

    > **Note on the half-tick.** Because the most aggressive quote sits $\tfrac12$ inside
    > the mid, the captured edge is $\delta - \tfrac12$, which is **negative for**
    > $\delta = 0$: quoting through the mid and filling for sure loses half a tick of
    > mark-to-market value. This is the cost of guaranteed execution and is exactly what
    > the model should weigh against inventory risk.

    #### **Total Reward**
    We'll do the comparison of three reward penalty , $r_p$, setups here

    | Terminal | Running | Both (Term+Run) |
    | --- | --- | --- |
    |  $-\gamma_{term}\,q_T^2$ at the end | $-\gamma_{run} \, \sigma^2 \, q_t^2$ each step | the sum of previous two |

    The total reward is :

    $$r_t = r_{w, \, t} + r_{p, \, t} = \underbrace{\Big[\mathbb{1}_{\text{ask}}\big(\delta_a - \tfrac{1}{2}\big)
    +\mathbb{1}_{\text{bid}}\big(\delta_b - \tfrac{1}{2}\big)\Big]}_{\text{spread capture}}\;+\; \underbrace{q_{t+1}\,\Delta_t}_{\text{reval}} + r_{p, \, t}$$
    """)
    return


@app.cell
def _(jnp, unflatten_action):
    def reward_decomposed(q, a, ask, bid, qn, dS, t, done, p):
        """
        Returns (full, reval). full = edge + reval + terminal + running.

        args: 
            q: [int] inventary
            a: [int] action idx
            ask: is ask order filled (1 for True, 0 for False)
            bid: is bid order filled (1 for True, 0 for False)
            qn: next inventary
            dS: Mid price increment
            t: time
            done: is done (1 for True, 0 for False)
            p: params
        """
        d_a, d_b = unflatten_action(a, p) # from Discrete(36) to (2,)
        edge = (ask * (d_a - 0.5) + bid * (d_b - 0.5)).astype(jnp.float32)
        reval = qn.astype(jnp.float32) * dS
        term = jnp.where(done, -p.gamma_term * qn.astype(jnp.float32) ** 2, 0.0)
        run = run = -p.gamma_run * p.sigma ** 2 * qn.astype(jnp.float32) ** 2  
        # run   = -2.0*p.gamma_run*p.sigma**2*qn.astype(jnp.float32)**2*(p.T-t).astype(jnp.float32)
        full = edge + reval + term + run
        return (full, reval)

    def reward_train(q, a, ask, bid, qn, dS, t, done, p):
        """
        Reval-free training reward (zero-mean reval removed).
        """
        full, reval = reward_decomposed(q, a, ask, bid, qn, dS, t, done, p)
        return full - reval

    return (reward_decomposed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### State
    Here we will build a market maker for a finite horizon contest.
    The state will have three component
    $$s_t = (S_t, q_t, t)$$
    Where,
    $S_t$ is the mid price at time $t$
    $q_t$ is market maker's inventary at time $t$, it is limited to $[-K_Q, K_Q]$
    $t$ is limited to $[0, T]$ (Finite horizon)
    ### Observation
    In this notebook, we will compare two observation

    | Full observation | Inventary Only |
    | --- | --- |
    | $o(s) = (S_t, q_t)$ | $o(s) = (q_t)$ |
    | More info, high volatility | Low info, maybe low performance |

    ### Action

    $$a = (\delta_a, \delta_b), \qquad \delta_a, \delta_b \in \{0, 1, \dots, K\},$$

    For a mid $S_t$, the ask at action level $\delta_a$ and the bid at level $\delta_b$
    are placed at

    $$\text{ask price} = S_t - \tfrac{1}{2} + \delta_a, \qquad
    \text{bid price} = S_t + \tfrac{1}{2} - \delta_b,$$

    with $\delta_a, \delta_b \in \{0, 1, \dots, K\}$.

    **Example** ($S_t = 101.5$): the ask levels for actions $\delta_a = 0,1,2,3,4,5$ are

    $$101,\ 102,\ 103,\ 104,\ 105,\ 106.$$

    The most aggressive ask ($\delta_a = 0$) sits at $101 = S_t - \tfrac{1}{2}$, exactly
    half a tick **inside** the mid; deeper levels step outward by one tick each.
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

        # determine if the order is filled
        pa, pb = fill_probs(d_a, d_b, p)
        u = jax.random.uniform(k_fill, (2,))
        ask = u[0] < pa; bid = u[1] < pb # is filled?

        # Inventary limitation
        at_hi = q >= p.K_Q; at_lo = q <= -p.K_Q 

        bid = bid & (~at_hi); ask = ask & (~at_lo)
        # state update
        qn = q + (-ask.astype(jnp.int32)) + bid.astype(jnp.int32)
        ks, pk = price_increment_probs(p)
        dS = p.tick * jax.random.choice(k_price, ks, p=pk).astype(jnp.float32)
        return qn, S+dS, ask, bid, dS

    # quick rollout sanity check
    def _demo_rollout():
        p = P_TERM; key = jax.random.PRNGKey(0)
        q, S, tot = jnp.int32(0), jnp.float32(p.S0), 0.0
        for t in range(p.T):
            key, k = jax.random.split(key)
            a = jnp.int32(7)  # arbitrary
            qn, Sn, ask, bid, dS = env_step(k, q, S, jnp.int32(t), a, p)
            full, _ = reward_decomposed(q, a, ask, bid, qn, dS, jnp.int32(t), (t+1)>=p.T, p)
            tot += float(full); q, S = qn, Sn
        print("demo rollout: terminal q =", int(q), " total full reward =", round(tot,3))
    _demo_rollout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Market Maker MDP — Dynamic Programming Model

    The goal of markov decision process (MDP) is to find the policy which generates series of actions (Policy trajectary, $\alpha_T$) that maximize the Value function:

    $$V^{\pi}(s) = \underset{\alpha_T \sim \pi}{E}\{R(\alpha_T)\left| s_0 = s\right.\}$$

    or Action-Value functino:

    $$ Q^{\pi}(s,a) = \underset{\alpha_T \sim \pi}{E}\{R(\alpha_T)\left| s_0 = s, a_0 = a\right.\}$$

    Where the return of a policy trajectray, $R(\alpha_T)$, is found by

    $$R(\alpha_T) = \begin{cases}
    \sum_{t=0}^T r_t & \text{for finite horizon}\\
    \sum_{t=0}^{\infty} r_t & \text{for infinite horizon}
    \end{cases}$$

    And certainly the following relation is ensured
    $$
            V^{\pi}(s) = \underset{a\sim \pi}{E} \, \{Q^{\pi}(s,a)\},
    $$
    ## Optimal and Bellman's equation
    If we find the optimal solution of policy and actions, we have:
    $$
            \max_{\pi}V^{\pi}(s) = V^*(s) = \max_a Q^* (s,a) = \max_a \{\max_{\pi} Q^{\pi} (s,a)\}
    $$

    The more important relation is Bellman's equation

    > The value of your starting point is the reward you expect to get from being there, plus the value of wherever you land next.


    The Bellman equations for the on-policy value functions are

    $$
    \begin{align*}
    V^{\pi}(s) &= \underset{a \sim \pi, \,s'\sim P}{E}\,\{r(s,a) + \gamma V^{\pi}(s')\},\\
    Q^{\pi}(s,a) &= \underset{s'\sim P}{E} \, \{r(s,a) + \gamma \underset{a'\sim \pi}{E} \, \{Q^{\pi}(s',a')\}\},
    \end{align*}
    $$
    where
    * $s' \sim P$ is shorthand for $s' \sim P(\cdot |s,a)$, indicating that the next state $s'$ is sampled from the environment's transition rules
    * $a \sim \pi$ is shorthand for $a \sim \pi(\cdot|s)$
    * $a' \sim \pi$ is shorthand for $a' \sim \pi(\cdot|s')$

    ## Derivation of DP solution
    ### Basic idea
    The dynamic programming's capibility of solving MDP comes from teardown the whole MDP into repeted one-step MDP. This can only be done by Bellman's equation and knowing the exact form of transition probability.
    > Bellman's equation is the recursion relation between current state $s$ and next state $s'$ if we know the transition probablilty.

    ### Transition and terminal condition

    $$p(s'|s, a) \doteq \Pr\{s_t=s' \mid s_{t-1}=s, A_{t-1}=a\} = \sum_{r \in \mathcal{R}} p(s', r | s, a).$$

    Note that our mid price change is independent of action and inventary change. So we can rewrite is as

    $$
    \begin{align*}
    p(s'|s, a) &= \Pr\{q_t = q' \mid q_{t=1} = q, A_{t-1} = a\}\Pr\{S_t = S' \mid S_{t-1} = S\}\\
    &= \Pr\{q_t = q' \mid q_{t=1} = q, A_{t-1} = a\}\Pr\{\Delta_S = k\tau\}
    \end{align*}
    $$

    The second term is ensured by Markov property and is a constant. Thus, we can focus only on the inventary change.

    Inventory changes by at most $\pm 1$ per step. Three reachable next-states:

    $$
    \begin{aligned}
    P(q - 1 \mid q, a) &= p_a (1 - p_b) && \text{(ask fills, bid does not; sold 1)}\\
    P(q + 1 \mid q, a) &= (1 - p_a) p_b && \text{(bid fills, ask does not; bought 1)}\\
    P(q \mid q, a) &= p_a p_b + (1 - p_a)(1 - p_b) && \text{(both or neither)}
    \end{aligned}
    $$

    These sum to $1$.

    #### **Boundary handling.**:
    At the inventory limits $\pm K_Q$, set the fill probability of the breaching side to zero (hard inventory cap) rather than letting probability mass spill off-grid. Renormalizing after the fact hides the leak.

    #### **State-space reduction**

    Two facts collapse the state to $q$ alone:

    1. **Translation invariance.** The price transition depends only on the increment
       $\Delta = S' - S$, and the fill probabilities depend only on the action — neither
       depends on $S$.

    2. **Price- and cash-free reward.** By Section 4, $R$ depends on neither $S$ nor $c$.

    Hence the value function depends only on $(q, t)$, and cash is carried purely in the
    reward stream. The transition object shrinks from $(S, q) \times a \times (S', q')$
    to $q \times a \times q'$.


    ### Backward induction

    * The normal policy iteration solves the **stationary** (infinite-horizon) problem.
    * This model is **finite-horizon** with a time-dependent value function $V_t(s_t)$, so here we choose a single backward sweep
    $V_T \to V_{T-1} \to \dots \to V_0$.
    * There is no fixed point to iterate to.
    Finite horizon $T$. Because $R$ and $p$ are time-homogeneous, they are computed
    **once**; only $V_t$ changes with $t$.

    #### Recursion

    For $t = T-1, T-2, \dots, 0$ and each state $q$:

    $$Q_t(s, a) = R(s, a) + \sum_{s'} p(s' \mid q, a)\, V_{t+1}(s'),$$

    $$V_t(s) = \max_a Q_t(s, a), \qquad \pi_t(s) = \arg\max_a Q_t(s, a).$$

    ### Expected reward $R(q, a)$

    #### Wealth reward

    $$\mathbb{E}[r_{w, \, t}] = p_a\big(\delta_a - \tfrac{1}{2}\big)
    +p_b\big(\delta_b - \tfrac{1}{2}\big)+\mathbb{E}[q_{t+1}\Delta_t]$$

    And we have

    $$\mathbb{E}[q_{t+1}\Delta_t] = \mu_\Delta\, \mathbb{E}[q_{t+1}] = 0,$$

    since the symmetric increment has $\mu_\Delta = 0$. Thus,
    $$\mathbb{E}[r_{w, \, t}] = p_a\big(\delta_a - \tfrac{1}{2}\big)
    +p_b\big(\delta_b - \tfrac{1}{2}\big)$$

    #### **Terminal penalty reward**
    $$r_{term} = -\gamma_{term} \, q_T^2$$
    By backward induction, we can avoid estimating its expectation

    #### **Running penalty reward**
    Originally from AS model, the optimal market maker can be obtained by solving the following HJB formulation

    $$H(t, x, q, S) = \sup_{(\delta^\pm_s)_{t \le s \le T} \in \mathcal{A}} \mathbb{E} \left[ X_T + q_T(S_T - \gamma_{term} q_T) - \gamma_{run} \sigma^2 \int_t^T q_s^2 \, ds \middle| X_{t-} = x, q_{t-} = q, S_{t-} = S \right]$$

    If we do the DP to teardown the expectation of whole MCP in time $[t, T]$ into single step expectation, the integral term will only contribute:

    $$r_{run,\, t} = -\gamma_{run} \, \sigma^2 \, q_t^2$$

    And its expectation should depend on the action:

    $$\mathbb{E}[r_{run,\, t}] = \sum_{q'} -\gamma_{run} \, \sigma^2 \, q'^2 \, p(q' \mid q, a)$$

    There are only three change from $q$ to $q'$ $(0, \pm 1)$

    $$\begin{align*}
    \mathbb{E}[r_{run,\, t}] &= p_a(1 - p_b)[ -\gamma_{run} \, \sigma^2 \,(q - 1) ^2 ]\\
    &+(1 - p_a)p_b[ -\gamma_{run} \, \sigma^2 \,(q + 1) ^2 ] \\
    &+\big[p_a p_b + (1 - p_a)(1 - p_b)\big][ -\gamma_{run} \, \sigma^2 \, q ^2 ]
    \end{align*}$$

    #### **Total expected reward**
    $$\begin{align*}
    R(s, a) = \mathbb{E}[r_t]
    &= \mathbb{E}[r_{w,\, t}] + \mathbb{E}[r_{term}] + \mathbb{E}[r_{run,\, t}] \\
    &= p_a(1 - p_b)\, [ \delta_a - \tfrac{1}{2}-\gamma_{run} \, \sigma^2 \,(q - 1) ^2 ]\\
    &+(1 - p_a)p_b\, [ \delta_b - \tfrac{1}{2} -\gamma_{run} \, \sigma^2 \,(q + 1) ^2 ] \\
    &+\big[p_a p_b + (1 - p_a)(1 - p_b)\big]\, [ \delta_a + \delta_b + 1 -\gamma_{run} \, \sigma^2 \, q ^2 ]\\
    &+ (-\gamma_{term}\, q_T^2) \mathbb{1}_{t=T}
    \end{align*}$$

    Let

    $$\begin{cases}
    R_+ &= \delta_b - \tfrac{1}{2} -\gamma_{run} \, \sigma^2 \,(q + 1) ^2, &p_+ = (1 - p_a)p_b\\
    R_- &= \delta_a - \tfrac{1}{2} -\gamma_{run} \, \sigma^2 \,(q - 1) ^2, &p_- = p_a(1 - p_b)\\
    R_0 &= \delta_a + \delta_b + 1 -\gamma_{run} \, \sigma^2 \, q ^2, &p_0 = p_a p_b + (1 - p_a)(1 - p_b)
    \end{cases}$$
    ### DP backward recurrion


    Expanded transition sum:

    $$\sum_{q'} p(q' \mid q, a)\, V_{t+1}(q')
    = p_a(1 - p_b)\,V_{t+1}(q - 1)
    +(1 - p_a)p_b\,V_{t+1}(q + 1)
    +\big[p_a p_b + (1 - p_a)(1 - p_b)\big]\,V_{t+1}(q).$$

    No discount factor is required for a finite horizon (a $\gamma \in (0, 1]$ may be
    added if desired).

    $$V_t(s) = \max_a Q_t(s, a) = \max_a\{ R(s, a) + \sum_{s'} p(s' \mid q, a)\, V_{t+1}(s')\}$$
    The $R(s, a)$ and $\sum_{s'} p(s' \mid q, a)\, V_{t+1}(s')$ has the same stucture, we merge them together

    $$V_t = p_+(R_+ + V_{t+1}(q+1))+p_-(R_- + V_{t+1}(q-1))+p_0(R_0 + V_{t+1}(q))$$

    This is a helpful form for accelerate the computation:

    ## Parameter summary

    | Symbol  | Meaning |
    | ------- | ------- |
    | $\tau = 1$ | tick size |
    | $S_0 = 100.5$ | initial mid (half-integer lattice) |
    | $\sigma$ | per-step price standard deviation |
    | $A = 1/dt$ | base fill intensity (calibrated so $P_{\text{fill}}(0) = 1$) |
    | $\kappa$ | fill-probability decay in spread |
    | $dt$ | step length |
    | $K$ | max half-spread level per side |
    | $K_Q$ | inventory cap, $\lvert q \rvert \le K_Q$ |
    | $\gamma_T$ | terminal inventory penalty weight |
    | $T$ | number of steps (horizon) |

    ---

    ## Summarize Modeling choices and caveats

    - **Symmetric increment assumed** ($\mu_\Delta = 0$). If drift is introduced, restore
      the reval term $\mu_\Delta(q + p_b - p_a)$ in $R(q, a)$; $R$ then depends on $q$.
    - **Terminal-only inventory control.** Without a running penalty, the agent may
      warehouse inventory mid-episode and unwind near $T$. To make it lean against
      inventory at every step, add a running term such as $-\tfrac{\gamma}{2}\sigma^2 q^2$
      to $R$ (the Avellaneda–Stoikov risk term). Out of scope for this version by design.
    - **Half-tick inside the mid.** Action $\delta = 0$ captures $-\tfrac12$ (a loss) but
      fills with certainty; the model trades this against inventory risk.
    - **One unit per side per step.** This bounds $\lvert \Delta q \rvert \le 1$ and gives
      the three-branch transition.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DP solver — ground truth (both regimes)

    Backward induction over `(q, t)`. Transition is 3-branch in `q` and
    time-homogeneous; the running penalty's `(T-t)` factor is folded into the
    next-state value at each scan step (each step knows its `t`).
    """)
    return


@app.cell
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
    return pi_dp_b, pi_dp_r, pi_dp_t, solve_dp


@app.cell
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
    ## Avellaneda–Stoikov baselines (running regime)
    The DP model we use here has nearly the same stepup as AS model. We will compare the DP-Both with it.

    * AS reservation price $r = S - q\gamma \sigma^2(T-t)$ and optimal half-spread
    $0.5(\gamma\sigma^2(T-t) + (2/\gamma)ln(1+\gamma/\kappa))$, mapped onto our env's
    quote convention (`ask` fills at $S - 1/2 + d_a$, `bid` at $S + 1/2 - d_b$).
    * **AS-continuous** posts the raw real-valued distances
    * **AS-grid** rounds to the nearest integer level (clamped to `>=1`). Their gap isolates the rounding cost.
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


@app.cell
def _(P_RUN, as_cont, as_grid, jax, jnp, np, plt):
    # ---- AS-grid / AS-continuous policy heatmaps over (q, t) ----
    qg = jnp.arange(-P_RUN.K_Q, P_RUN.K_Q + 1)          # inventory grid
    tg = jnp.arange(P_RUN.T)                             # time grid

    # evaluate AS deltas on the full (t, q) grid at the reference mid S0
    # (S cancels out of d_a, d_b in as_deltas, so S0 is representative)
    def as_skew_grids(p):
        def row(t):
            dc = jax.vmap(lambda q: as_cont(jnp.float32(p.S0), q, t, p))(qg)  # (2, n_q) tuple
            dg = jax.vmap(lambda q: as_grid(jnp.float32(p.S0), q, t, p))(qg)
            skew_c = dc[0] - dc[1]      # continuous skew d_a - d_b
            skew_g = dg[0] - dg[1]      # gridded skew
            return skew_c, skew_g
        sc, sg = jax.vmap(row)(tg)      # each (T, n_q)
        return np.array(sc), np.array(sg)

    skew_cont, skew_grid_ = as_skew_grids(P_RUN)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # AS-continuous skew (real-valued)
    im0 = axes[0].imshow(skew_cont.T, aspect='auto', origin='lower', cmap='RdBu_r',
                         extent=[0, P_RUN.T, -P_RUN.K_Q, P_RUN.K_Q],
                         vmin=-P_RUN.K, vmax=P_RUN.K)
    axes[0].set_title("AS-continuous: skew (d_a - d_b)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("q")
    plt.colorbar(im0, ax=axes[0])

    # AS-grid skew (rounded)
    im1 = axes[1].imshow(skew_grid_.T, aspect='auto', origin='lower', cmap='RdBu_r',
                         extent=[0, P_RUN.T, -P_RUN.K_Q, P_RUN.K_Q],
                         vmin=-P_RUN.K, vmax=P_RUN.K)
    axes[1].set_title("AS-grid: skew (d_a - d_b)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("q")
    plt.colorbar(im1, ax=axes[1])

    # rounding effect: grid - continuous
    diff = skew_grid_ - skew_cont
    im2 = axes[2].imshow(diff.T, aspect='auto', origin='lower', cmap='PuOr',
                         extent=[0, P_RUN.T, -P_RUN.K_Q, P_RUN.K_Q],
                         vmin=-1.5, vmax=1.5)
    axes[2].set_title("AS-grid - AS-continuous (rounding)")
    axes[2].set_xlabel("t"); axes[2].set_ylabel("q")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout(); plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * The AS Model has nearly the same result as running penalty
    * But the delta difference of AS model is much smaller, we will explore the difference betwee these two in the future works
    * One thing you can notice is that, the grid one has lots of difference to the continuous one, the possible reason is:
        * The parameter tunning is wrong
        * There is something wrong of theoretical expression of AS model
        * AS model is not suitible for discontinuous work and need further modification
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A bit wird result of running penalty
    The reason why running penalty is wird might be
    ```
    When the time goes to T, the running penalty, which is calculated by integrating
    quare quote value in the future, decrease dramatically.
    ```
    So the optimal solution will show the ignorance of penalty during the last time section. To validate my idea, a simple experiment can be leveraged. We will test it through increase the time horizon.
    """)
    return


@app.cell
def _(MMParams, plt, skew_grid, solve_dp):
    P_RUN_LT = MMParams(sigma=1.0, kappa=1.5, gamma_term=0.0, gamma_run=0.002, T=100) 
    # Only running penalty with long time horizon
    V_dp_r_lt, pi_dp_r_lt, _ = solve_dp(P_RUN_LT)
    _fig, _ax = plt.subplots(1, 1, figsize=(13, 4))

    _sg = skew_grid(pi_dp_r_lt, P_RUN_LT).T
    _im = _ax.imshow(_sg, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, P_RUN_LT.T, -P_RUN_LT.K_Q, P_RUN_LT.K_Q], vmin=-P_RUN_LT.K, vmax=P_RUN_LT.K)  # (n_q, T)
    _ax.set_xlabel('time t')
    _ax.set_ylabel('inventory q')
    _ax.set_title('DP running: skew (d_a - d_b)')
    plt.colorbar(_im, ax=_ax)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is clear to see that for time in [0, 60], the running penalty makes the policy restrict the inventary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Risk management parameter comparison
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Shared-seed evaluation harness

    Every policy faces **identical** randomness: a pre-drawn `(N,T,2)` table of
    fill-coins and an `(N,T)` table of price increments. The price path is identical
    across policies; fills use the same coin thresholded by each policy's own
    `p_fill`. Trajectories still diverge through inventory (correct). Reward at
    evaluation is the **full** reward (reval included).
    """)
    return


@app.cell
def _(
    as_cont,
    as_grid,
    jax,
    jit,
    jnp,
    lax,
    net,
    partial,
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


    return (evaluate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison of terminal penalty factor $\gamma_{term}$
    """)
    return


@app.cell
def _(MMParams, evaluate, jax, plt, skew_grid, solve_dp):
    EVAL_KEY = jax.random.PRNGKey(12345)
    N_EVAL = 3000

    EVALS_termcom = {}
    gamma_terms = [0.02, 0.2 , 2.0, 20.0]
    _fig, _axes = plt.subplots(2, 2, figsize=(13, 8))
    _axes = _axes.ravel()
    for i, g_t in enumerate(gamma_terms):
        P_TERM_dgt = MMParams(sigma=1.0, kappa=1.5, gamma_term=g_t, gamma_run=0.0, T=30) 
        V_dp_t_dgt, pi_dp_t_dgt, _ = solve_dp(P_TERM_dgt)
        EVALS_termcom[f'DP_TERM_gamma_t={g_t}'] = (evaluate(EVAL_KEY, P_TERM_dgt, kind='dp', dp_pi=pi_dp_t_dgt, N=N_EVAL), P_TERM_dgt)
        _sg = skew_grid(pi_dp_t_dgt, P_TERM_dgt).T
        _im = _axes[i].imshow(_sg, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, P_TERM_dgt.T, -P_TERM_dgt.K_Q, P_TERM_dgt.K_Q], vmin=-P_TERM_dgt.K, vmax=P_TERM_dgt.K)  # (n_q, T)
        _axes[i].set_xlabel('time t')
        _axes[i].set_ylabel('inventory q')
        _axes[i].set_title(f'DP terminal-gamma_term={g_t}: skew (d_a - d_b)')
        plt.colorbar(_im, ax=_axes[i])

    print('evaluated', len(EVALS_termcom), 'policies x', N_EVAL, 'shared-seed episodes')
    plt.tight_layout()
    plt.show()
    return EVALS_termcom, EVAL_KEY, N_EVAL


@app.cell
def _(EVALS_termcom, np):
    def pnl_of(ev): return np.array(ev[2].sum(axis=1))   # rpath sum over t, per episode
    def quant_stats(ev):
        pnl = pnl_of(ev)
        qpath = np.array(ev[0])  # (N,T)
        mean, std = pnl.mean(), pnl.std()
        sharpe = mean/ (std+1e-9)
        downside = pnl[pnl<0]
        sortino = mean / (downside.std()+1e-9) if downside.size>0 else np.inf
        # max drawdown on the MEAN cumulative-PnL path across episodes
        cum = np.array(ev[2]).mean(axis=0).cumsum()
        peak = np.maximum.accumulate(cum); dd = (peak - cum).max()
        var5 = np.percentile(pnl, 5)
        cvar5 = pnl[pnl<=var5].mean() if (pnl<=var5).any() else var5
        return dict(mean=mean, std=std, sharpe=sharpe, sortino=sortino,
                    max_dd=dd, VaR5=var5, CVaR5=cvar5,
                    mean_absq=np.abs(qpath).mean(), term_absq=np.abs(qpath[:,-1]).mean())

    import pandas as pd
    term_rows = {name: quant_stats(ev) for name,(ev,p) in EVALS_termcom.items()}
    df_termcom = pd.DataFrame(term_rows).T
    df_termcom = df_termcom[['mean','std','sharpe','sortino','max_dd','VaR5','CVaR5','mean_absq','term_absq']]
    df_termcom.columns = ['PnL mean','PnL std','Sharpe','Sortino','Max DD','VaR 5%','CVaR 5%','mean|q|','term|q|']
    df_termcom.round(3)
    return pd, pnl_of, quant_stats


@app.cell
def _(EVALS_termcom, plt, pnl_of):
    fig_termcom, ax_termcom = plt.subplots(figsize=(11,4))
    names_termcom = list(EVALS_termcom.keys())
    data_termcom = [pnl_of(EVALS_termcom[n][0]) for n in names_termcom]
    parts_termcom = ax_termcom.violinplot(data_termcom, showmeans=True, showextrema=False)
    ax_termcom.set_xticks(range(1,len(names_termcom)+1)); ax_termcom.set_xticklabels(names_termcom)
    ax_termcom.set_ylabel("terminal PnL"); ax_termcom.set_title("Realized PnL distribution (shared seeds)")
    ax_termcom.grid(alpha=0.3, axis='y'); plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison of terminal penalty factor $\gamma_{run}$
    """)
    return


@app.cell(hide_code=True)
def _(EVAL_KEY, MMParams, N_EVAL, evaluate, plt, skew_grid, solve_dp):
    EVALS_runcom = {}
    gamma_runs = [0.0004, 0.004 , 0.04, 0.4]
    _fig, _axes = plt.subplots(2, 2, figsize=(13, 8))
    _axes = _axes.ravel()
    for _i, g_r in enumerate(gamma_runs):
        P_RUN_dgr = MMParams(sigma=1.0, kappa=1.5, gamma_term=0.0, gamma_run=g_r, T=30) 
        V_dp_r_dgr, pi_dp_r_dgr, _ = solve_dp(P_RUN_dgr)
        EVALS_runcom[f'DP_RUN_gamma_r={g_r}'] = (evaluate(EVAL_KEY, P_RUN_dgr, kind='dp', dp_pi=pi_dp_r_dgr, N=N_EVAL), P_RUN_dgr)
        _sg = skew_grid(pi_dp_r_dgr, P_RUN_dgr).T
        _im = _axes[_i].imshow(_sg, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, P_RUN_dgr.T, -P_RUN_dgr.K_Q, P_RUN_dgr.K_Q], vmin=-P_RUN_dgr.K, vmax=P_RUN_dgr.K)  # (n_q, T)
        _axes[_i].set_xlabel('time t')
        _axes[_i].set_ylabel('inventory q')
        _axes[_i].set_title(f'DP running-gamma_run={g_r}: skew (d_a - d_b)')
        plt.colorbar(_im, ax=_axes[_i])

    print('evaluated', len(EVALS_runcom), 'policies x', N_EVAL, 'shared-seed episodes')
    plt.tight_layout()
    plt.show()
    return (EVALS_runcom,)


@app.cell(hide_code=True)
def _(EVALS_runcom, pd, quant_stats):
    _rows = {name: quant_stats(ev) for name,(ev,p) in EVALS_runcom.items()}
    _df = pd.DataFrame(_rows).T
    _df = _df[['mean','std','sharpe','sortino','max_dd','VaR5','CVaR5','mean_absq','term_absq']]
    _df.columns = ['PnL mean','PnL std','Sharpe','Sortino','Max DD','VaR 5%','CVaR 5%','mean|q|','term|q|']
    _df.round(3)
    return


@app.cell(hide_code=True)
def _(EVALS_runcom, plt, pnl_of):
    _fig, _ax = plt.subplots(figsize=(11,4))
    _names = list(EVALS_runcom.keys())
    _data = [pnl_of(EVALS_runcom[n][0]) for n in _names]
    _parts = _ax.violinplot(_data, showmeans=True, showextrema=False)
    _ax.set_xticks(range(1,len(_names)+1)); _ax.set_xticklabels(_names)
    _ax.set_ylabel("terminal PnL"); _ax.set_title("Realized PnL distribution (shared seeds)")
    _ax.grid(alpha=0.3, axis='y'); plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison of maximum inventary $K_Q$
    """)
    return


@app.cell(hide_code=True)
def _(EVAL_KEY, MMParams, N_EVAL, evaluate, plt, skew_grid, solve_dp):
    EVALS_MI = {}
    K_Qs = [3, 5, 10, 20]
    _fig, _axes = plt.subplots(2, 2, figsize=(13, 8))
    _axes = _axes.ravel()
    for _i, k_Q in enumerate(K_Qs):
        P_TERM_dkq = MMParams(sigma=1.0, kappa=1.5, gamma_term=2.0, gamma_run=0.0, T=30, K_Q=k_Q) 
        V_dp_t_dkq, pi_dp_t_dkq, _ = solve_dp(P_TERM_dkq)
        EVALS_MI[f'DP_TERM_K_Q={k_Q}'] = (evaluate(EVAL_KEY, P_TERM_dkq, kind='dp', dp_pi=pi_dp_t_dkq, N=N_EVAL), P_TERM_dkq)
        _sg = skew_grid(pi_dp_t_dkq, P_TERM_dkq).T
        _im = _axes[_i].imshow(_sg, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, P_TERM_dkq.T, -P_TERM_dkq.K_Q, P_TERM_dkq.K_Q], vmin=-P_TERM_dkq.K, vmax=P_TERM_dkq.K)  # (n_q, T)
        _axes[_i].set_xlabel('time t')
        _axes[_i].set_ylabel('inventory q')
        _axes[_i].set_title(f'DP terminal-K_Q={k_Q}: skew (d_a - d_b)')
        plt.colorbar(_im, ax=_axes[_i])

    print('evaluated', len(EVALS_MI), 'policies x', N_EVAL, 'shared-seed episodes')
    plt.tight_layout()
    plt.show()
    return EVALS_MI, K_Qs


@app.cell(hide_code=True)
def _(EVALS_MI, pd, quant_stats):
    _rows = {name: quant_stats(ev) for name,(ev,p) in EVALS_MI.items()}
    _df = pd.DataFrame(_rows).T
    _df = _df[['mean','std','sharpe','sortino','max_dd','VaR5','CVaR5','mean_absq','term_absq']]
    _df.columns = ['PnL mean','PnL std','Sharpe','Sortino','Max DD','VaR 5%','CVaR 5%','mean|q|','term|q|']
    _df.round(3)
    return


@app.cell(hide_code=True)
def _(EVALS_MI, plt, pnl_of):
    _fig, _ax = plt.subplots(figsize=(11,4))
    _names = list(EVALS_MI.keys())
    _data = [pnl_of(EVALS_MI[n][0]) for n in _names]
    _parts = _ax.violinplot(_data, showmeans=True, showextrema=False)
    _ax.set_xticks(range(1,len(_names)+1)); _ax.set_xticklabels(_names)
    _ax.set_ylabel("terminal PnL"); _ax.set_title("Realized PnL distribution (shared seeds)")
    _ax.grid(alpha=0.3, axis='y'); plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison of maximum inventary $K_Q$ under high variance of price update
    """)
    return


@app.cell
def _(EVAL_KEY, K_Qs, MMParams, N_EVAL, evaluate, plt, skew_grid, solve_dp):
    EVALS_MI_HV = {}
    _fig, _axes = plt.subplots(2, 2, figsize=(13, 8))
    _axes = _axes.ravel()
    for k_hv, k_Q_hv in enumerate(K_Qs):
        P_TERM_dkq_hv = MMParams(sigma=7.0, kappa=1.5, gamma_term=2.0, gamma_run=0.0, T=30, K_Q=k_Q_hv) 
        V_dp_t_dkq_hv, pi_dp_t_dkq_hv, _ = solve_dp(P_TERM_dkq_hv)
        EVALS_MI_HV[f'DP_TERM_K_Q={k_Q_hv}'] = (evaluate(EVAL_KEY, P_TERM_dkq_hv, kind='dp', dp_pi=pi_dp_t_dkq_hv, N=N_EVAL), P_TERM_dkq_hv)
        _sg = skew_grid(pi_dp_t_dkq_hv, P_TERM_dkq_hv).T
        _im = _axes[k_hv].imshow(_sg, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, P_TERM_dkq_hv.T, -P_TERM_dkq_hv.K_Q, P_TERM_dkq_hv.K_Q], vmin=-P_TERM_dkq_hv.K, vmax=P_TERM_dkq_hv.K)  # (n_q, T)
        _axes[k_hv].set_xlabel('time t')
        _axes[k_hv].set_ylabel('inventory q')
        _axes[k_hv].set_title(f'DP terminal-K_Q={k_Q_hv}: skew (d_a - d_b)')
        plt.colorbar(_im, ax=_axes[k_hv])

    print('evaluated', len(EVALS_MI_HV), 'policies x', N_EVAL, 'shared-seed episodes')
    plt.tight_layout()
    plt.show()
    return (EVALS_MI_HV,)


@app.cell
def _(EVALS_MI_HV, pd, quant_stats):
    kq_rows_hv = {name: quant_stats(ev) for name,(ev,p) in EVALS_MI_HV.items()}
    df_kqcom_hv = pd.DataFrame(kq_rows_hv).T
    df_kqcom_hv = df_kqcom_hv[['mean','std','sharpe','sortino','max_dd','VaR5','CVaR5','mean_absq','term_absq']]
    df_kqcom_hv.columns = ['PnL mean','PnL std','Sharpe','Sortino','Max DD','VaR 5%','CVaR 5%','mean|q|','term|q|']
    df_kqcom_hv.round(3)
    return


@app.cell
def _(EVALS_MI_HV, plt, pnl_of):
    _fig, _ax = plt.subplots(figsize=(11,4))
    _names = list(EVALS_MI_HV.keys())
    _data = [pnl_of(EVALS_MI_HV[n][0]) for n in _names]
    _parts = _ax.violinplot(_data, showmeans=True, showextrema=False)
    _ax.set_xticks(range(1,len(_names)+1)); _ax.set_xticklabels(_names)
    _ax.set_ylabel("terminal PnL"); _ax.set_title("Realized PnL distribution (shared seeds)")
    _ax.grid(alpha=0.3, axis='y'); plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The result of comparison shows that:
    * $\gamma_{term}$ can control the risk well (reduce the variance of PnL), however if it is too large, its affect will be dilluted (compare $\gamma_{term} = 2.0, 20.0$)
    * $\gamma_{run}$ can control the risk as well, but if it gets too large, it will make policy too restricted, even affects its PnL performance (check $\gamma_{run} = 0.04, 0.4$ for their shew heatmap and PnL chart)
    * Decrease the maximum inventary limitaion $K_Q$ can't control the risk, it restrict the policy to place deep order to avoid order fill. This experiment need to be completed by allowing policy to not place the order. This will be done in the future
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
