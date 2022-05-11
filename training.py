import argparse
from data import *
from GP_utils import *
from models import *
import os
import torch
import loss
import time
import json
from comparison_implementations.tloss import scikit_wrappers
from comparison_implementations.TNC.tnc import models as tncmodels
from comparison_implementations.TNC.tnc import tnc
#impl of cpc provided with tnc
from comparison_implementations.TNC.baselines import cpc

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',type=str) #will create a folder in this location to store the results
parser.add_argument('--batch_size',type=int,default=512)
parser.add_argument('--n_epochs',type=int,default=100)
parser.add_argument('--tau',type=float,default=.5)
parser.add_argument('--lr',type=float,default=.01)
parser.add_argument('--weight_decay',type=float,default=10**-6)
parser.add_argument('--x_jitter_strength',type=float,default=.8)
parser.add_argument('--y_jitter_strength',type=float,default=2)
parser.add_argument('--model_name',type=str,default='contrastive')
parser.add_argument('--h_size',type=int,default=512)
parser.add_argument('--tnc_window',type=int,default=20)

#ablations for different classes of augmentation-ignored if not contrastive model
parser.add_argument('--incl_y_jitter',type=int,default=1)
parser.add_argument('--incl_position_jitter',type=int,default=1)
parser.add_argument('--incl_rescale',type=int,default=1)
device='cuda' if torch.cuda.is_available() else'cpu'
xs=np.linspace(0,10,100)

args= parser.parse_args()
os.mkdir(args.save_dir)

losses=[]
accs=[]
cl_accs=[]




h_size=args.h_size
z_size=128

incl_y_jitter=args.incl_y_jitter>0
incl_position_jitter=args.incl_position_jitter>0
incl_rescale=args.incl_rescale>0

if args.model_name=='contrastive':
    model=ConvEncoder(convolutional=True,h_size=h_size).to(device)
    head=torch.nn.Sequential(*[torch.nn.Linear(h_size,z_size),torch.nn.LeakyReLU(),torch.nn.Linear(z_size,z_size)]).to(device)
    opt=torch.optim.Adam(list(model.parameters())+list(head.parameters()),lr=args.lr)
    head.train()
elif args.model_name=='contrastive-fc-encoder':
    model=ConvEncoder(convolutional=False,h_size=h_size).to(device)
    head=torch.nn.Sequential(*[torch.nn.Linear(h_size,z_size),torch.nn.LeakyReLU(),torch.nn.Linear(z_size,z_size)]).to(device)
    opt=torch.optim.Adam(list(model.parameters())+list(head.parameters()),lr=args.lr)
    head.train()

elif args.model_name=='contrastive-cnp-encoder':
    model=CNPEncoder(h_size=h_size).to(device)
    head=torch.nn.Sequential(*[torch.nn.Linear(h_size,z_size),torch.nn.LeakyReLU(),torch.nn.Linear(z_size,z_size)]).to(device)
    opt=torch.optim.Adam(list(model.parameters())+list(head.parameters()),lr=args.lr)
    head.train()
elif args.model_name=='cnp':
    model=CNP(h_size=h_size).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)

elif args.model_name=='vae':
    model=VAE(h_size=args.h_size).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)

elif args.model_name=='t-loss':
    #triplet loss model, from unsupervised scalable representation learning paper
    model = scikit_wrappers.CausalCNNEncoderClassifier()
    #we set nb_steps so that the model proceeds through the entire batch once
    #and the same hidden size as the contrastive model

    #other params copied from the "default hparams" file
    hyperparameters = {
        "batch_size": args.batch_size,
        "channels": 40,
        "compared_length": None,
        "depth": 10,
        "nb_steps": 1,
        "in_channels": 1,
        "kernel_size": 3,
        "penalty": None,
        "early_stopping": None,
        "lr": args.lr,
        "nb_random_samples": 10,
        "negative_penalty": 1,
        "out_channels": h_size,
        "reduced_size": h_size,
        "cuda": device=='cuda',
        "gpu": 0
    }
    model.set_params(**hyperparameters)
    #we do not need to explicitly create an optimizer in this case,
    #because it is a class variable in the model
elif args.model_name=='tnc':
    #temporal neighborhood coding
    #params copied from the code for "simulation" except for the encoding size
    window_size = args.tnc_window
    w = .05
    model = tncmodels.RnnEncoder(hidden_size=100, in_channel=1, encoding_size=h_size, device=device)
    disc_model = tnc.Discriminator(model.encoding_size, device)
    params = list(disc_model.parameters()) + list(model.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)#, weight_decay=10**-5)
elif args.model_name=='cpc':
    #use the cpc implemtation from the tnc repo
    window_size = args.tnc_window
    model = tncmodels.RnnEncoder(hidden_size=100, in_channel=1, encoding_size=h_size, device=device)
    ds_estimator = torch.nn.Linear(h_size, h_size)
    auto_regressor = torch.nn.GRU(input_size=h_size, hidden_size=h_size, batch_first=True)
    params = list(ds_estimator.parameters()) + list(model.parameters())+list(auto_regressor.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)#, weight_decay=10**-5)





else:
    raise ValueError('invalid model_name')



try:
    model.train()
except:
    pass
curves_per_epoch=50000
iters_per_epoch=curves_per_epoch//args.batch_size
n_iters=args.n_epochs*iters_per_epoch

STARTTIME=time.time()


for ii in range(n_iters):
    try:
        opt.zero_grad()
    except:
        pass
    y = []
    labs = []
    for _ in range(50):
        K, n = sample_comp_kernel() if np.random.rand() < .5 else sample_mix_kernel()
        n = 10 ** -5
        for n_tries in range(1000):
            try:
                y1 = sample_gaussian(args.batch_size // 50, C=K.cov(xs, xs) + n * np.eye(len(xs)), use_svd=True)
                break
            except:
                continue
        if n_tries==999:
            raise ValueError('kernel was not invertible,giving up after 1000 tries')
        y.append(y1)
        labs = labs + [K.id()] * len(y1)
    labs = torch.Tensor(labs).long()
    y = np.concatenate(y, axis=0)
    if args.model_name=='contrastive' or args.model_name=='contrastive-cnp-encoder' or args.model_name=='contrastive-fc-encoder':
        yhat=full_jitter(y,x_strength=args.x_jitter_strength,y_strength=args.y_jitter_strength,xs=xs,
                         incl_y_jitter=incl_y_jitter,incl_position_jitter=incl_position_jitter,incl_rescale=incl_rescale)
        y=full_jitter(y,x_strength=args.x_jitter_strength,y_strength=args.y_jitter_strength,xs=xs,
                      incl_y_jitter=incl_y_jitter,incl_position_jitter=incl_position_jitter,incl_rescale=incl_rescale)
        y = torch.from_numpy(y).float().to(device)
        yhat = torch.from_numpy(yhat).float().to(device)

        h0 = model(y)
        z0 = head(h0)
        h0_hat= model(yhat)
        z0_hat = head(h0_hat)
        zz = torch.cat((z0, z0_hat), 0)
        l, acc = loss.simclr_loss(zz, tau=args.tau)
        l.backward()
        opt.step()
        losses.append(l.item())
        accs.append(acc)
    elif args.model_name=='cnp':
        y=rand_rescale(y)
        y=torch.from_numpy(y).float().to(device)
        xpos=1.0*torch.arange(y.shape[1])/y.shape[1]
        xpos=torch.cat([xpos[None,:] for _ in range(len(y))],0).to(device)
        #select 10 points at random from each curve for testing
        #since the encoder is order-invariant we can do this by shuffling each row independently
        yshuff=[]
        xposshuff=[]
        for ii in range(len(y)):
            rp=np.random.permutation(y.shape[1])
            yshuff.append(y[ii][rp].unsqueeze(0))
            xposshuff.append(xpos[ii][rp].unsqueeze(0))
        yshuff=torch.cat(yshuff,0)
        xposshuff=torch.cat(xposshuff,0)

        xobs=xposshuff[:,:90]
        yobs=yshuff[:,:90]

        xtest=xposshuff[:,90:]
        ytest=yshuff[:,90:]

        mu,logsigma=model(xobs,yobs,xtest)
        #maximize normal log likelihoood of test points
        dy=ytest-mu
        llh=-.5*torch.exp(-2*logsigma)*dy*dy-logsigma
        l=-llh.mean()
        l.backward()
        opt.step()
        losses.append(l.item())
        accs.append(-1)


    elif args.model_name=='vae':
        y=rand_rescale(y)
        y=torch.from_numpy(y).float().to(device)
        recon,mu,logsigma=model(y)
        l=loss.vae_loss(recon,y,mu,logsigma)
        l.backward()
        opt.step()
        losses.append(l.item())
        accs.append(-1)
    #in both of these cases, the optimizer.zero_grad() and .step() methods are done in the training functions
    #so we do not have to explicitly do it here
    elif args.model_name=='t-loss':
        y=rand_rescale(y)
        y=y[:,None,:]
        model.fit_encoder(y, save_memory=True, verbose=True)
        losses.append(-1)
        accs.append(-1)
    elif args.model_name=='tnc':
        y=rand_rescale(y)
        y=y[:,None,:]
        #set epochs=0 because of off-by-one in training loop
        #other params copied from the code for "simulation" case
        tnc.learn_encoder(y, model, disc_model,opt,window_size,w, gp_batch_size=args.batch_size,lr=args.lr, decay=1e-5, n_epochs=0,
                          mc_sample_size=40, path='gp', device=device, augmentation=5, n_cross_val=1)
        losses.append(-1)
        accs.append(-1)


    elif args.model_name=='cpc':
        y = rand_rescale(y)
        y = y[:, None, :]
        #no off-by-one in the training loop here, compared with in the above
        #other params copied from "simulation" case
        cpc.learn_encoder(y,window_size,model,ds_estimator,auto_regressor,
                                  opt,n_size=15,n_epochs=1,device=device)
        losses.append(-1)
        accs.append(-1)


    if ii>0 and ii%iters_per_epoch==0:
        elapsed=time.time()-STARTTIME
        av_time=elapsed/ii
        remaining=round((n_iters-ii)*av_time/3600,2)
        print(f'est remaining={remaining}')

        print(f'loss={np.mean(losses[-iters_per_epoch:])}')
        print(f'contrastive acc={np.mean(accs[-iters_per_epoch:])}')


if args.model_name=='t-loss':
    torch.save(model.encoder.state_dict(),f'{args.save_dir}/model.pth')
else:
    torch.save(model.state_dict(),f'{args.save_dir}/model.pth')

if args.model_name=='cpc':
    torch.save(auto_regressor.state_dict(),f'{args.save_dir}/autoregressor.pth')


np.savetxt(f'{args.save_dir}/loss.txt',losses)
np.savetxt(f'{args.save_dir}/contrastive_acc.txt',accs)

hparams=vars(args)
with open(f'{args.save_dir}/hparams.json','w') as f:
    json.dump(hparams,f)
