#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for urn project.

@author: David Samu
"""


import os
import copy
import string
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import entropy as sp_entropy


# %% System I/O functions.

def create_dir(f):
    """Create directory if it does not already exist."""

    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return


def write_objects(obj_dict, fname):
    """Write out dictionary object into pickled data file."""

    create_dir(fname)
    pickle.dump(obj_dict, open(fname, 'wb'))


def read_objects(fname, obj_names=None):
    """Read in objects from pickled data file."""

    data = pickle.load(open(fname, 'rb'))

    # Unload objects from dictionary.
    if obj_names is None:
        objects = data  # all objects
    elif isinstance(obj_names, str):
        objects = data[obj_names]   # single object
    else:
        objects = [data[oname] for oname in obj_names]  # multiple objects

    return objects


def get_copy(obj, deep=True):
    """Returns (deep) copy of object."""

    copy_obj = copy.deepcopy(obj) if deep else copy.copy(obj)
    return copy_obj


def save_fig(ffig, fig=None, dpi=300, close=True, tight_layout=True):
    """Save composite (GridSpec) figure to file."""

    # Init figure and folder to save figure into.
    create_dir(ffig)

    if fig is None:
        fig = plt.gcf()

    if tight_layout:
        fig.tight_layout()

    fig.savefig(ffig, dpi=dpi, bbox_inches='tight')

    if close:
        plt.close(fig)


# %% String formatting functions.

def format_to_fname(s):
    """Format string to file name compatible string."""

    valid_chars = "_ %s%s" % (string.ascii_letters, string.digits)
    fname = ''.join(c for c in s if c in valid_chars)
    fname = fname.replace(' ', '_')
    return fname


def form_str(v, njust):
    """Format float value into string for reporting."""

    vstr = ('%.1f' % v).rjust(njust)
    return vstr


# %% Pandas functions.

def vectorize(sel_idx, index):
    """
    Return indexed vector with all zero values except for selected index,
    which is set to 1.
    """

    vec = pd.Series(0., index=index)
    vec[sel_idx] = 1

    return vec


# %% Maths functions.

def softmax(x):
    """Compute softmax values for vector x."""

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def entropy(p, base=None):
    """Compute entropy of probability distribution p."""

    H = sp_entropy(p, base=base)
    return H


def D_KL(p, q, base=None):
    """Compute  Kullback-Leibler divergence between PDs p and q."""

    D = sp_entropy(p, q, base=base)
    return D
