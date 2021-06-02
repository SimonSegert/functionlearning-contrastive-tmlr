import torch
import numpy as np
from scipy.special import logsumexp


#implements the horizontal jitter augmentation
def position_jitter(inp, sigma=.1, xs=None, strength=1, use_scaling=True):
    # if use_scaling, then the weights will be normalized to sum to 1
    # if not, then the overall scale is essentially arbitrary for each curve
    # however, if the curves will later be fed to rand_rescale, then this
    # arbitrary scaling factor will wash out anyway
    lx = max(xs) - min(xs)
    left = min(xs) - lx * np.random.uniform(low=0, high=strength, size=len(inp))
    right = max(xs) + lx * np.random.uniform(low=0, high=strength, size=len(inp))

    new_pts = np.random.uniform(size=len(xs))
    new_pts = new_pts * (right - left)[:, None] + left[:, None]
    new_pts = np.sort(new_pts, axis=1)
    # dd=-.5*np.add.outer(new_pts,-xs)**2/sigma**2
    dd = -.5 * (new_pts[:, :, None] - xs[None, None, :]) ** 2 / sigma ** 2

    if use_scaling:
        dd = np.exp(dd - logsumexp(dd, axis=1)[:, None, :])
    else:
        dd = np.exp(dd - np.max(dd, axis=1)[:, None, :])
    outp = np.einsum('ijk,ij->ik', dd, inp, optimize='optimal')
    return outp

#implements the horizontal flipping augmentation, and optionally an extra nonlinear vertical warping
def y_jitter(inp, strength=2):
    #when strength=1, then the vertical warping is not performed
    #all experiments have strength=1
    rand_signs = np.random.choice([1, -1], size=len(inp))
    mm = np.mean(inp, axis=1)[:, None]
    st = np.std(inp, axis=1)[:, None]
    inp_zsc = (inp - mm) / st
    exps = np.random.uniform(low=1, high=strength + 1, size=len(inp))
    inp_zsc = np.sign(inp_zsc) * np.abs(inp_zsc) ** (exps[:, None])
    yy = mm + (inp_zsc * st)
    return yy * rand_signs[:, None]

#implements random rescaling augmentation
def rand_rescale(inp, min_ht=.8):
    # rescale each curve to lie completely the interval [0,1], with difference between min and max at least min_ht

    new_mins = np.random.uniform(size=len(inp), low=0, high=(1 - min_ht) / 2)
    new_maxs = np.random.uniform(size=len(inp), low=(1 + min_ht) / 2, high=1)

    # first rescale so min=0 and max=1
    outp = inp - np.min(inp, axis=1)[:, None]
    outp = outp / np.max(outp, axis=1)[:, None]

    outp = outp * (new_maxs - new_mins)[:, None] + new_mins[:, None]
    return outp


#performs all three augmentations in sequence
def full_jitter(y,y_strength=2,x_strength=.8,xs=None):
    yhat = y_jitter(y, strength=y_strength)
    yhat = position_jitter(yhat, sigma=.1, strength=x_strength,xs=xs)
    yhat = rand_rescale(yhat)
    return yhat

