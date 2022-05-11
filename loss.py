import torch
import numpy as np

def vae_loss(preds,targets,mu,logsigma):
    #preds: outputs from vae, assumed to be logits
    #targets: reconstruction targets, assumed to be in [0,1]

    recon_loss=torch.nn.BCEWithLogitsLoss(reduction='sum')(preds,targets)/len(preds)
    kl=.5*torch.sum(torch.exp(logsigma)**2+mu**2-2*logsigma,axis=1)
    return recon_loss+torch.mean(kl)


def simclr_loss(z, tau=1):
    # z is assumed to be a tensor of shape 2N x K
    # we assume that row i and row i+N are generated from the same image

    N = z.shape[0] // 2  # number of pairs
    zn = z / (torch.sum(z ** 2, axis=1)[:, None] + 10 ** -8) ** .5
    sim = torch.mm(zn, zn.T) / tau
    mask = torch.zeros(*sim.shape) - torch.eye(sim.shape[0]) * 10 ** 9
    mask = mask.to(z.device)
    # index of positive pair for each image
    pos_sim = torch.mean(torch.cat((torch.diagonal(sim, N), torch.diagonal(sim, -N))))
    neg_sim = torch.mean(torch.logsumexp(sim + mask, axis=1))
    contrastive_pred = torch.argmax(sim + mask, 1).cpu().numpy()
    contrastive_acc = np.mean(contrastive_pred == np.concatenate((np.arange(N) + N, np.arange(N))))
    return -(pos_sim - neg_sim), contrastive_acc
