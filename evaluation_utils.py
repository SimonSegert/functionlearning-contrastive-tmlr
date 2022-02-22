import torch
import numpy as np
xs=np.linspace(0,10,100)
device='cuda' if torch.cuda.is_available() else 'cpu'
from GP_utils import *
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from scipy.special import logsumexp

#extract global time series representation from each model type
def get_reps(model, ys, model_hparams, auto_reg=None):
    # ys should be shape (n curves) x (length of curve)
    # auto_reg is ignored unless the model is cpc
    mn = model_hparams['model_name']
    # split up input into several chunks, to avoid gpu memory errors
    dl = torch.utils.data.DataLoader(ys, batch_size=256, shuffle=False)
    if mn == 'native':
        inps = ys.clone()
    elif mn == 'contrastive' or mn=='contrastive-cnp-encoder':
        # inps=model(ys)
        inps = torch.cat([model(x) for x in dl], 0)
    elif mn=='cnp':
        inps=torch.cat([model.enc(x) for x in dl],0)

    elif mn == 'vae' or mn == 'hopf':
        # _,inps,_=model(ys)
        inps = torch.cat([model(x)[1] for x in dl], 0)
    elif mn == 'native':
        inps = ys
    elif mn == 't-loss':
        inps = np.vstack([model.encode(x[:, None, :].cpu().numpy()) for x in dl])
        inps = torch.from_numpy(inps).float().to(ys.device)
    elif mn.startswith('tnc') or mn == 'cpc':
        # first extract the representation for each window
        window_size = model_hparams['tnc_window']
        ys_windowed = []
        # the offset needed to fit as many windows as possible in the time series
        offset = ys.shape[-1] % window_size
        for r in ys:
            windows = []
            for j in range(offset // 2, ys.shape[-1] - offset // 2, window_size):
                windows.append(r[j:j + window_size][None, :])
            # shape: windows per curve x window length
            windows = torch.vstack(windows)
            ys_windowed.append(windows[None, None, :, :])
        # shape: n curves x 1 x windows per curve x window length
        ys_windowed = torch.cat(ys_windowed, axis=0)
        dl_w = torch.utils.data.DataLoader(ys_windowed, batch_size=256, shuffle=False)
        if mn.endswith('avwindow'):
            # average the representations over all window
            z_windowed = []
            for window_id in range(ys_windowed.shape[2]):
                # representation of the ith window for each curve
                # shape: n curves x hidden size
                # zw=model(ys_windowed[:,:,window_id])
                zw = torch.cat([model(x[:, :, window_id]) for x in dl_w], 0)
                z_windowed.append(zw.unsqueeze(2))
            z_windowed = torch.cat(z_windowed, axis=2)
            z_windowed = torch.mean(z_windowed, axis=2)
            inps = z_windowed
        elif mn.endswith('catwindow') or mn == 'cpc':
            # concatenate the representations for all windows
            z_windowed = []
            for window_id in range(ys_windowed.shape[2]):
                # representation of the ith window for each curve
                # shape: n curves x hidden size
                # zw=model(ys_windowed[:,:,window_id])
                zw = torch.cat([model(x[:, :, window_id]) for x in dl_w], 0)
                z_windowed.append(zw)
            if mn.startswith('tnc'):
                z_windowed = torch.cat(z_windowed, axis=1)
                # for tnc, we concatenate all representations together
                inps = z_windowed
            else:
                # shape: n_curves x n_windows x hidden size
                z_seq = torch.cat([x.unsqueeze(1) for x in z_windowed], axis=1)
                # recall auto_reg is batch_first
                _, c_t = auto_reg(z_seq)
                inps = c_t.squeeze(0)
            # for cpc, we feed this to the autoregressor
        else:
            raise ValueError('invalid tnc name')
    else:
        raise ValueError('')
    return inps

def generate_curves(kernel_samples=2048, n_obs=80):
    labs = []
    y0s = []
    ys = []
    cov_mats = []
    kernels = []
    for k_id in range(14):
        K, n = sample_comp_kernel(k_i=k_id) if k_id != 13 else sample_mix_kernel()
        cov_mats.append(K.cov(xs, xs) + n * np.eye(len(xs)))
        y0 = sample_gaussian(kernel_samples, C=K.cov(xs, xs) + n * np.eye(len(xs)))
        y0s.append(y0)
        y = y0 - np.min(y0, 1)[:, None]
        y = y / np.max(y, 1)[:, None]
        y = torch.from_numpy(y).float().to(device)
        labs = labs + [k_id] * len(y)
        kernels.append(K)
        ys.append(y.cpu())
    y0s = np.concatenate(y0s, 0)
    ys = torch.cat(ys, 0).numpy()
    labs = np.array(labs)
    rp = np.random.permutation(len(labs))
    y0s = y0s[rp]
    ys = ys[rp]
    labs = labs[rp]
    llh = np.vstack([-.5 * np.einsum('ij,jk,ik->i', y0s[:, :n_obs], np.linalg.inv(cov_mats[i][:n_obs, :n_obs]),
                                     y0s[:, :n_obs]) - .5 * np.linalg.slogdet(cov_mats[i][:n_obs, :n_obs])[1] for i in
                     range(14)]).T
    Cinvs = [np.linalg.inv(cov_mats[i][:n_obs, :n_obs]) for i in range(len(cov_mats))]
    y0s_obs = y0s[:, :n_obs]
    mix_comps = np.vstack(
        [kernels[-1].posterior(xs[:n_obs], y0s[i, :n_obs], xs[n_obs:], K11_inverse=Cinvs[-1])[0] for i in
         range(len(y0s))])
    comp_comps = np.vstack([kernels[3 + np.argmax(llh[i, 3:13] + 3)].posterior(xs[:n_obs], y0s[i, :n_obs], xs[n_obs:],
                                                                               K11_inverse=Cinvs[
                                                                                   3 + np.argmax(llh[i, 3:13])])[0] for
                            i in range(len(y0s))])

    rbf_comps = np.vstack(
        [kernels[1].posterior(xs[:n_obs], y0s[i, :n_obs], xs[n_obs:], K11_inverse=Cinvs[1])[0] for i in
         range(len(y0s))])
    # posterior using the true generative kernel
    true_comps = np.vstack(
        [kernels[labs[i]].posterior(xs[:n_obs], y0s[i, :n_obs], xs[n_obs:], K11_inverse=Cinvs[labs[i]])[0] for i in
         range(len(y0s))])
    y0s_mix = np.concatenate((y0s_obs, mix_comps), 1)
    y0s_comp = np.concatenate((y0s_obs, comp_comps), 1)
    y0s_rbf = np.concatenate((y0s_obs, rbf_comps), 1)
    y0s_true = np.concatenate((y0s_obs, true_comps), 1)
    # rescale so that the sampled curve, the compositional completion, and mixture completion
    # all line in [0,1]
    all_vals = np.array([y0s_mix, y0s_comp, y0s])
    all_min = np.min(all_vals, axis=(0, 2))[:, None]
    all_max = np.max(all_vals, axis=(0, 2))[:, None]
    # use consistent rescaling so all curves agree on first portion
    new_mins = np.random.uniform(size=len(y0s), low=0, high=(1 - .8) / 2)
    new_maxs = np.random.uniform(size=len(y0s), low=(1 + .8) / 2, high=1)
    y0s_mix = (y0s_mix - all_min) / (all_max - all_min)
    y0s_mix = y0s_mix * (new_maxs - new_mins)[:, None] + new_mins[:, None]
    y0s_rbf = (y0s_rbf - all_min) / (all_max - all_min)
    y0s_rbf = y0s_rbf * (new_maxs - new_mins)[:, None] + new_mins[:, None]
    y0s_comp = (y0s_comp - all_min) / (all_max - all_min)
    y0s_comp = y0s_comp * (new_maxs - new_mins)[:, None] + new_mins[:, None]
    y0s_true = (y0s_true - all_min) / (all_max - all_min)
    y0s_true = y0s_true * (new_maxs - new_mins)[:, None] + new_mins[:, None]
    dnew_to_old = np.add.outer(np.linspace(0, n_obs, 100), -np.arange(n_obs))
    sigma = .001
    ker_new_to_old = -dnew_to_old ** 2 / sigma ** 2
    ker_new_to_old = np.exp(ker_new_to_old - logsumexp(ker_new_to_old, axis=1)[:, None])
    ys_rsc = (y0s - all_min) / (all_max - all_min)
    ys_rsc = ys_rsc * (new_maxs - new_mins)[:, None] + new_mins[:, None]
    y0s_inp = np.dot(y0s_mix[:, :n_obs], ker_new_to_old.T)
    return y0s, ys, labs, y0s_obs, y0s_inp, y0s_mix, y0s_comp, y0s_rbf, y0s_true, ys_rsc


def train_mc(hs_comp,hs_mix,hs_obs,multchoice_targets,batch_size=256,n_epochs=50,lr=.01,device='cpu',weight_decay=10**-6):
    #training on multiple choice questions using quadratic model
    #hs_comp: the h vectors of the composition completion
    #hs_mix: the h vectors of the mixture completions
    #hs_obs: the h vectors of the input curve (not seeing the completed part)
    #multchoice targets:0 for comp and 1 for mixture
    mc_head=torch.nn.Linear(hs_comp.shape[1],32).to(device)
    mc_opt=torch.optim.Adam(mc_head.parameters(),lr=lr,weight_decay=weight_decay)
    n_batches=(n_epochs*len(hs_comp))//batch_size
    for ii in range(n_batches):
        ids=np.random.choice(len(hs_comp),batch_size,replace=False)
        inp_h=hs_obs[ids].detach().to(device)
        mc_targets=multchoice_targets[ids]

        mc_opt.zero_grad()
        choices=torch.cat([mc_head(hh.detach().to(device)).unsqueeze(0) for hh in [hs_comp[ids].to(device),hs_mix[ids].to(device)]],0)
        logits=torch.sum(mc_head(inp_h).unsqueeze(0)*choices,2).T
        l=torch.nn.CrossEntropyLoss()(logits,torch.from_numpy(mc_targets).long().to(device))
        l.backward()
        mc_opt.step()
        if ii%500==0:
            print(l)
    return mc_head


def latex_table(df, res_name, sigfigs=2, ci='sd', mult_100=False):
    #prints dataframe in latex table format
    # df should hold the average scores for each model,run id,task id,train size combination
    # will print the latex code for the corresponding table
    # should have entires for 'model name' and 'train size'
    # res_name: the name of the column that holds the result
    # ci can be 'sd' or 'sem' corresponding to standard deviation or standard error of mean
    #if mult_100, then values will be multiplied by 100
    mn = 'model name' if 'model name' in df.columns else 'model'
    aa = df.groupby([mn, 'train size']).mean().reset_index()[[mn, 'train size', res_name]]
    if ci == 'sd':
        bb = df.groupby([mn, 'train size']).std().reset_index()[[mn, 'train size', res_name]]
    elif ci == 'sem':
        bb = df.groupby([mn, 'train size']).sem().reset_index()[[mn, 'train size', res_name]]
    vals = pd.merge(aa, bb, on=[mn, 'train size'])
    if mult_100:
        vals[f'{res_name}_x'] *= 100
        vals[f'{res_name}_y'] *= 100
    if ci == 'sem':
        vals[f'{res_name}_y'] *= 1.96  # factor for 95% confidence interval
    fmt_str = "{:." + str(sigfigs) + "f}"
    vals['mean_str'] = vals[f'{res_name}_x'].map(lambda x: fmt_str.format(x))
    vals['std_str'] = vals[f'{res_name}_y'].map(lambda x: fmt_str.format(x))
    vals['res'] = vals['mean_str'] + '$\pm$ ' + vals['std_str']
    vals = vals[[mn, 'train size', 'res']].reset_index()
    tab_str = pd.pivot(vals, columns='train size', index=mn, values='res').to_latex(escape=False)
    print(tab_str)


