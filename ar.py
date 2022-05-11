import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

#utilities for fitting linear autoregressive models

def split_ar(x, window_size=10):
    # split time series into array of windows, with next point being predictors
    # x should be 1 dimensional array
    try:
        X = np.vstack([x[i:i + window_size] for i in range(0, len(x) - window_size)])
    except:
        print(x.shape)
        raise ValueError('')
    Y = x[window_size:]
    return X, Y

def split_ar_batch(xs, window=10):
    #similar to split_ar, except xs canhave multiple rows
    X = []
    Y = []
    curve_ids = []
    for i, x in enumerate(xs):
        x, y = split_ar(x, window_size=window)
        X.append(x)
        Y.append(y)
        curve_ids = curve_ids + [i] * len(x)
    X = np.vstack(X)
    Y = np.concatenate(Y)
    return X, Y, np.array(curve_ids)


def extrapolate_ar(reg, xs, window=10, n_pts=20):
    # use a single fitted ar model to extrapolate the series
    # xs should be shape (n curves)x (curve lengths)

    # extrapolation for each curve
    exts = []
    inp = xs[:, -window:]
    for _ in range(n_pts):
        exts.append(reg.predict(inp)[:, None])  # shape (n curves x 1)
        inp = np.concatenate((inp[:, 1:], exts[-1]), 1)
    return np.concatenate(exts, 1)


def fit_hlm(ys, cl, probs=None, window=10, n_clusters=14, use_probs=False):
    #fits hierarchical linear autoregressive model, using given cluster assignments
    #can optionally provide soft cluster assignments

    # probs should be of shape n curves x n clusters, if provided
    # is ignored is use_probs=False
    # if use_probs=True, then they are used as sample weights
    X, Y, c_id = split_ar_batch(ys, window=window)
    # create a separate indicator variable for each curve
    # shape=n windows x n curves*window
    # curve_ind=1*np.concatenate([np.repeat((c_id==i)[:,None],window,1) for i in range(max(c_id)+1)],1)
    # create a separate indicator variable for each class

    # create regressors for each lag position in each class
    # shape n windows x window*n clusters
    # for example if first curve is in first cluster, then the row would look like
    # [1,...1,0,...0], where number of ones=size of window
    # each window from this curve would have the same row
    # if using probs, then indicator functions are replaced with probabilities
    if not use_probs:
        class_ind = 1 * np.concatenate([np.repeat((cl[c_id] == i)[:, None], window, 1) for i in range(n_clusters)], 1)
        # adding "1" for class specific intercept
        intercepts = 1 * np.concatenate([(cl[c_id] == i)[:, None] for i in range(n_clusters)], 1)
    else:
        class_ind = np.zeros((len(X), X.shape[1] * n_clusters))
        for r_id in range(probs.shape[0]):
            for col_id in range(probs.shape[1]):
                class_ind[r_id, col_id * window:col_id * (window + 1)] = probs[r_id, col_id]
    X1 = np.concatenate([X for _ in range(max(c_id) + 1)], 1)
    X2 = np.concatenate([X for _ in range(n_clusters)], 1)
    assert X2.shape == class_ind.shape, print(X2.shape, class_ind.shape)
    # construct regressors
    # first "block"
    X = np.concatenate((X, class_ind * X2, intercepts), 1)  # ,curve_ind*X1),1)
    reg = Ridge(alpha=.0001)
    grid_search = GridSearchCV(
        reg, {
            'alpha': [10.0 ** i for i in range(-6, 4)]
        },
        cv=5, n_jobs=5
    )
    grid_search.fit(X, Y)
    reg = grid_search.best_estimator_
    reg.fit(X, Y)
    c0 = reg.coef_[:window]
    # coef/intercept for each class
    main_effects = np.vstack([reg.coef_[window + i * window:window + (1 + i) * window] for i in range(n_clusters)])
    intercepts = reg.coef_[-n_clusters:]
    # package into separate regression objects
    regs = []
    for j in range(n_clusters):
        r = Ridge(alpha=reg.alpha)
        r.coef_ = c0 + main_effects[j]
        r.intercept_ = reg.intercept_ + intercepts[j]
        regs.append(r)
    return regs, reg, class_ind, X, X2


def forecast_hlm(regs, ys, cl, probs=None, window=10, n_pts=20, use_probs=False):
    #given a fitted hlm and cluster assignments, forecast the curves
    #using the respective autoregressive models
    if not use_probs:
        return np.vstack(
            [extrapolate_ar(regs[cl[i]], ys[i][None, :], window=window, n_pts=n_pts) for i in range(len(cl))])
    else:
        exts = np.concatenate([extrapolate_ar(r, ys, window=window, n_pts=n_pts)[:, :, None] for r in regs], 2)
        return np.sum(exts * probs[:, None, :], 2)


