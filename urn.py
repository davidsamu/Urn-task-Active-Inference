# -*- coding: utf-8 -*-
"""
Active Inference solution to the Urn task.

This is a replication of the simulations of the paper:

FitzGerald TH, Schwartenbeck P, Moutoussis M, Dolan RJ, Friston K.
Active inference, evidence accumulation and the urn task.
Neural computation. 2015;27(2):306-328. doi:10.1162/NECO_a_00699.

@author: David Samu
"""

import os
import sys

from functools import reduce
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

from scipy.linalg import block_diag

fproj = '/home/david/Modelling/Urn/'  # project directory, need to be set!
sys.path.insert(1, fproj)

from urn import utils

proj_dir = fproj

os.chdir(proj_dir)


# %% Define Generative Model of task.

# Task constants.
n = 6    # maximum number of draws
p = 0.85  # probability of drawing a red ball
urn_colors = ('R', 'G')
U = ('W', 'R', 'G')
decisions = ('X',) + urn_colors

# Hidden states.
s_names = ('urn_color', 'decision', 'n_draws', 'n_green')
i_rng = range(n+1)
S = list(product(urn_colors, decisions, i_rng, i_rng))
Smi_all = pd.MultiIndex.from_tuples(S, names=s_names)
S = [s for s in S if s[2] >= s[3]]
Smi = pd.MultiIndex.from_tuples(S, names=s_names)

# Observations.
o_names = s_names[1:]
O = list(pd.Series([s[1:] for s in S]).unique())
Omi = pd.MultiIndex.from_tuples(O, names=o_names)

# Observation likelihood matrix: s -> o
A = pd.DataFrame(np.kron([1, 1], np.identity(len(O))), columns=Smi, index=Omi)

# Transition probability matrix: s -> s'
I2 = np.identity(2)
In = np.identity(n+1)
Inn = np.identity((n+1)*(n+1))
I2nn = np.identity(2*(n+1)*(n+1))
Dn = np.eye(n+1, k=-1)
B_j_red = np.kron(Dn, p * In + (1-p) * Dn)
B_j_green = np.kron(Dn, p * Dn + (1-p) * In)
B_wait = pd.DataFrame(block_diag(B_j_red, I2nn, B_j_green, I2nn),
                      index=Smi_all, columns=Smi_all)
for urn_col, i in product(urn_colors, i_rng):
    B_wait.loc[(urn_col, 'X', n, i), (urn_col, 'X', n, i)] = 1
B_dec_red = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1]])
B_dec_green = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]])
B_red, B_green = [pd.DataFrame(np.kron(np.kron(I2, B_dec), Inn),
                               index=Smi_all, columns=Smi_all)
                  for B_dec in (B_dec_red, B_dec_green)]
B = {'W': B_wait, 'R': B_red, 'G': B_green}
B = {u: Bu.loc[Bu.index.isin(Smi), Bu.columns.isin(Smi)]
     for u, Bu in B.items()}

# Policy prior.
policies = [i * ('W',) + (color,) for color in urn_colors for i in i_rng]
Pi_0 = pd.Series(1./len(policies), index=policies, name=0)  # uniform prior

# Initial state prior.
s_0 = pd.Series(0., index=Smi, name=0)
for color in urn_colors:
    s_0[color, 'X', 0, 0] = 1. / len(urn_colors)

# Utility prior.
tau = 3    # log odds ratio when comparing posterior evidence for urn color
kappa = 4  # sensitivity to a significant odds ratio
rho = np.log(p) - np.log(1-p)   # p is assumed to be known here!
lnc = pd.Series(0., index=Smi)
for color in urn_colors:
    for ntotal in range(n+1):
        for ngreen in range(ntotal+1):
            sign = 1 if color == 'G' else -1
            is_sign = sign * (2*ngreen - ntotal) * rho > tau
            lnc[color, color, ntotal, ngreen] = kappa * is_sign
c = np.exp(lnc)
c = c / c.sum()

# Precision prior.
alpha = 3
beta = 1


# %% Functions to run simulations.

# Functions for posterior on GM
# -----------------------------
def calc_T(pi, t):
    """
    Return state transition matrix for policy pi from time t till end of
    policy.
    """

    assert t < len(pi)

    T_pi = reduce(lambda x, y: x.dot(y), [B[u] for u in pi[t:]][::-1])

    return T_pi


def calc_G(policies, Smi, t, lnc):
    """
    Return G: value of each policy for each state at time t, given utility of
    final states lnc.
    """

    G = pd.DataFrame(0., index=policies, columns=Smi)
    for pi in policies:
        if len(pi) <= t:  # policy finished by this time point
            continue
        T_pi = calc_T(pi, t)  # policy's transition matrix to final state
        for si in Smi:
            sT = T_pi[si]
            # value of policy = expected utility + entropy (surprise)
            G[si][pi] = sT.dot(lnc) + utils.entropy(sT)

    # Normalize each column into a PD.
    G = G / G.sum()
    G = G.fillna(0)

    # Zero-out impossible states at time t.
    G.loc[:, G.columns.get_level_values('n_draws') != t] = 0

    return G


# Functions for variational Bayes update
# --------------------------------------
def update_s(A, o, B, u, s_prev, gamma, G, c, Pi, eps=1e-8):
    """Update hidden state estimate."""

    v = np.log(A.loc[o] + eps)

    if u is not None:  # no control (before first step)
        v += np.log(B[u].dot(s_prev) + eps)

    v += gamma * G.T.dot(Pi)

    s = utils.softmax(v)

    return s


def update_pi(gamma, G, s):
    """Update policy estimate."""

    Pi = utils.softmax(gamma * G.dot(s))
    return Pi


def update_prec(alpha, beta, Pi, G, s, eps=1e-8):
    """Update precision estimate."""

    gamma = alpha / (beta - Pi.dot(G.dot(s)) + eps)
    assert gamma > 0   # This guaranteed if G is a PD, but otherwise not...

    return gamma


# Functions for control / action selection
# ----------------------------------------
def select_MAP_control(Pi, t):
    """Select next control deterministically (maximum a posteriori)."""

    u = Pi.idxmax()[t]
    return u


def sample_control(Pi, t):
    """Sample next control probabilistically."""

    uvec, pvec = zip(*[(pi[t], pval) for pi, pval in Pi.items()
                       if len(pi) > t])
    pvec = np.array(pvec) / sum(pvec)
    u = np.random.choice(uvec, p=pvec)

    return u


# Some task-specific functions.
def step_trial(s_real, u, B, A):
    """Generate next observation."""

    sx_real = np.random.choice(B[u].index, p=B[u][s_real])  # transition state
    ox = np.random.choice(A.index, p=A[sx_real])   # generate next observation

    return sx_real, ox


# State - policy value functions (matrices).
G = {}
print('\nCalculating state - policy value function for each draw')
for t in range(n+1):
    print('draw: %i / %i' % (t+1, n))
    G[t] = calc_G(policies, Smi, t, lnc)


# %% Function to run trial.

def run_trial(s_real, o, s_0, Pi_0, alpha, beta, A, B, G, n_var_update,
              urn_colors, s_names, o_names, U, Smi, Omi, action_sel):
    """Run a trial."""

    # Inits
    n = len(G.keys()) - 1  # maximum number of draws
    u = None  # dummy action
    gamma = alpha / beta  # precision

    # History of environment.
    s_real_l = [s_real]  # list of states visited so far
    ol = [o]  # list of observations so far
    ul = []   # list of actions performed so far

    # History of agent's beliefs.
    s_l = pd.DataFrame(s_0)
    Pi_l = pd.DataFrame(Pi_0)
    gamma_l = [gamma]

    s = s_0.copy()
    Pi = Pi_0.copy()

    for t in range(n+1):  # n draw plus final decision

        s_prev = s

        # Perform variational belief update.
        for iupdt in range(n_var_update):

            s = update_s(A, o, B, u, s_prev, gamma, G[t], c, Pi)
            Pi = update_pi(gamma, G[t], s)
            gamma = update_prec(alpha, beta, Pi, G[t], s)

        # Select control.
        u = action_sel(Pi, t)  # MAP or sample?

        # Step trial: generate next state and observation.
        s_real, o = step_trial(s_real, u, B, A)

        # Save current environmental variable to history.
        s_real_l.append(s_real)
        ol.append(o)
        ul.append(u)

        # Save agent's current beliefs to history.
        s_l[t] = s
        Pi_l[t] = Pi
        gamma_l.append(gamma)

        if u in urn_colors:
            break

    # Convert results to Pandas objects.
    s_real_df = pd.DataFrame(s_real_l, columns=s_names)
    o_df = pd.DataFrame(ol, columns=o_names)
    u_df = pd.DataFrame(ul, columns=['u'])

    s_df = s_l.T
    Pi_df = Pi_l.T
    gamma_ser = pd.Series(gamma_l)

    # Add index to each value (for plotting).
    s_real_df = s_real_df.assign(idx=[Smi.get_loc(sr) for sr in s_real_l])
    o_df = o_df.assign(idx=[Omi.get_loc(o) for o in ol])
    u_df = u_df.assign(idx=[U.index(u) for u in ul])

    # Add draw color.
    s_real_df = s_real_df.assign(draw_color=s_real_df.n_green.diff())
    s_real_df['draw_color'].replace({np.nan: 'K', 0: 'R', 1: 'G'},
                                    inplace=True)

    return s_real_df, o_df, u_df, s_df, Pi_df, gamma_ser


# %% Function to plot results.

def plot_trial(s_real_df, u_df, s_df, Pi_df, gamma_ser, alpha, Smi, urn_colors,
               proj_dir):
    """Plot trial results."""

    fig, axarr = plt.subplots(2, 2, figsize=(12, 12))
    color_dict = {urn_col: urn_col for urn_col in urn_colors}

    # Plot environment state over time.
    ax = axarr[0, 0]
    colors = list(s_real_df['draw_color'])
    colors[-1] = s_real_df['decision'].iloc[-1]
    ax.scatter(s_real_df.index, s_real_df.idx, color=colors, clip_on=False)
    ax.set_ylim([-1, len(Smi)-1])
    ax.set_ylabel('state index')
    ax.set_title('Real state')

    # Plot belief state of urn color over time.
    b_urn_col_dict = {}
    idx = pd.IndexSlice
    for t, st in s_df.iterrows():
        for urn_col in urn_colors:
            # Taking only the correct number of green draws.
            ngr = s_real_df.loc[t, 'n_green']
            b_urn_col_dict[(t, urn_col)] = st[idx[urn_col, :, t, ngr]].sum()
    b_urn_col = pd.Series(b_urn_col_dict, name='p')
    b_urn_col.index.names = ['t', 'color']
    b_urn_col = b_urn_col.reset_index()

    ax = axarr[0, 1]
    ax.axhline(0.5, ls='--', c='grey')
    sns.tsplot(b_urn_col, time='t', unit='color', condition='color', value='p',
               color=color_dict, err_style='unit_points', clip_on=False, ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title('Belief about urn color')

    ax = axarr[1, 0]
    pPi_df = Pi_df.T
    # pPi_df.index = [''.join(pi) for pi in pPi_df.index]
    pPi_df.index = [str(len(pi)-1) + '-' + pi[-1] for pi in pPi_df.index]
    pPi_df.columns = [str(t) for t in pPi_df.columns]
    sns.heatmap(pPi_df, linewidths=.5, vmin=0, vmax=1, cbar=False, annot=True,
                ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')
    ax.set_title('Belief about policy')

    # Plot belief about precision.
    ax = axarr[1, 1]
    ax.plot(gamma_ser[:-1], '-o', clip_on=False)
    # ax.set_ylim([0, None])
    ax.set_ylabel('gamma')
    ax.set_title('Precision')

    # Set global settings.
    my_xaxis_locator = mpl.ticker.MaxNLocator(integer=True)
    for i in range(axarr.shape[0]):
        for j in range(axarr.shape[1]):
            ax = axarr[i, j]
            ax.set_xlabel('t')
            if i != 1 or j != 0:
                ax.xaxis.set_major_locator(my_xaxis_locator)

    draw_colors = ''.join([dc for dc in s_real_df['draw_color'][1:-1]])
    pi = str(len(u_df['u'])-1) + u_df['u'].iloc[-1]
    title = 'alpha: %i, draws: %s, policy: %s' % (alpha, draw_colors, pi)
    fig.suptitle(title, y=1.02, fontsize=20)

    # Save figure.
    fname = utils.format_to_fname(title)
    ffig = proj_dir + 'results/%i_draws/%s.png' % (n, fname)
    utils.save_fig(ffig, fig)


# %% Simulations.

# Parameters.
n_var_update = 8  # number of variational updates per draw
action_sel = select_MAP_control  # select_MAP_control or sample_control

# Init environment.
s_real_0 = ('R', 'X', 0, 0)
o_0 = s_real_0[1:]

nruns = 10

print('\n')
for i in range(nruns):

    print('run: %i / %i' % (i+1, nruns))

    # Run trial.
    res = run_trial(s_real_0, o_0, s_0, Pi_0, alpha, beta, A, B, G,
                    n_var_update, urn_colors, s_names, o_names,
                    U, Smi, Omi, action_sel)
    s_real_df, o_df, u_df, s_df, Pi_df, gamma_ser = res

    # Plot results.
    plot_trial(s_real_df, u_df, s_df, Pi_df, gamma_ser, alpha, Smi,
               urn_colors, proj_dir)


# %% Dynamic Programming solution.

# Optimal solution intuitively:
# - draw as many balls as possible, then decide to most frequently drawn color


# POMDP solution
# --------------

# Standard solution for Belief MDP: perform value iteration on belief states.
# V_opt(s) = max_u [ r_exp(s, u) + sum_o P(o | s,u) * V_opt(s') ]
# where s' is the updated belief state using information from observation o.

def V_belief(s, n):
    """
    Return value function and optimal action from belief state s for n steps
    ahead.
    """

    if n == 0:
        return 0, ''
    else:
        v = {}
        sx_u = {}
        for ui in U:   # for each possible action
            sx = B[ui].dot(s)  # next belief state
            r = lnc.dot(sx)    # expected immediate reward
            v[ui] = r + V_belief(sx, n-1)[0]
            sx_u[ui] = sx

        vmax = max(v.values())
        umax = max(v, key=v.get)
        sxmax = sx_u[umax]

        return vmax, umax, sxmax


# Generate solution by forward playing generative belief model with specific
# instantiations of observations, using their probability, immediate utility
# and future expected reward.

V_belief(s_0, n)
