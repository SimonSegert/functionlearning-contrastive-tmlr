import os
from models import ConvEncoder,VAE
import json
from evaluation_utils import *
from GP_utils import *
np.random.seed(1234)
from sklearn.linear_model import SGDClassifier
import pandas as pd
from ar import *



device='cuda' if torch.cuda.is_available() else 'cpu'




xs=np.linspace(0,10,100)







from comparison_implementations.tloss.networks.causal_cnn import CausalCNNEncoder
from comparison_implementations.tloss.scikit_wrappers import CausalCNNEncoderClassifier
from comparison_implementations.TNC.tnc.models import RnnEncoder



from copy import deepcopy



#load in the saved models
models_dir='saved_models'
models=[]
auto_regs=[]
model_hparams=[]
for folder_name in os.listdir(models_dir):
    for runname in os.listdir(f'{models_dir}/{folder_name}'):
        dd=f'{models_dir}/{folder_name}/{runname}'
        try:
            hparams=json.load(open(f'{dd}/hparams.json','r'))
        except:
            continue
            
       
            
            
        if hparams['model_name']=='contrastive':
            if hparams['x_jitter_strength'] not in [.2,.4] or hparams['y_jitter_strength']!=0:
                continue
            model=ConvEncoder(convolutional=True,h_size=hparams['h_size']).to(device)
        elif hparams['model_name']=='vae':
            model=VAE(h_size=hparams['h_size']).to(device)
        elif hparams['model_name']=='t-loss':
            enc=CausalCNNEncoder(1, 40, 10, hparams['h_size'],hparams['h_size'],3).to(device)
            model=CausalCNNEncoderClassifier(compared_length=None, nb_random_samples=10,
                 negative_penalty=1, batch_size=hparams['batch_size'], nb_steps=1, lr=0.001,
                 penalty=None, early_stopping=None, channels=40, depth=10,
                 reduced_size=hparams['h_size'], out_channels=hparams['h_size'], kernel_size=3,
                 in_channels=1, cuda=device=='cuda', gpu=0)
            enc.eval()
            model.encoder=enc
        elif hparams['model_name']=='tnc':
            window_size = hparams['tnc_window']
            w = .05
            model = RnnEncoder(hidden_size=100, in_channel=1, encoding_size=hparams['h_size'], device=device)
            
        elif hparams['model_name']=='cpc':
            model = RnnEncoder(hidden_size=100, in_channel=1, encoding_size=hparams['h_size'], device=device)
            autoregressor=torch.nn.GRU(input_size=hparams['h_size'], hidden_size=hparams['h_size'], batch_first=True)

        else:
            raise ValueError('')
        hparams['run_id']=runname.split('run')[1]
        
        
        if hparams['model_name']=='t-loss':
            enc.load_state_dict(torch.load(f'{dd}/model.pth'))
        else:
            model.load_state_dict(torch.load(f'{dd}/model.pth'))

            
        if hparams['model_name']=='cpc':
            autoregressor.load_state_dict(torch.load(f'{dd}/autoregressor.pth'))
            autoregressor.eval()
            
            
            
            
        if hparams['model_name']!='t-loss':
            model.eval()
        
            
        if hparams['model_name']!='tnc':
            models.append(model)
            model_hparams.append(hparams)
        else:
            #load either the concat or average variant depending on the size of the hidden layer
            
            appended_name='catwindow' if hparams['h_size']==25 else 'avwindow'
            hp=deepcopy(hparams)
            hp['model_name']=f'tnc_{appended_name}'
            models.append(model)
            model_hparams.append(hp)
        if hparams['model_name']=='cpc':
            auto_regs.append(autoregressor.to(device))
        else:
            auto_regs.append(None)

#as a control, add a dummy model which just copies the input
models.append(torch.nn.Sequential())
auto_regs.append(None)
model_hparams.append({'model_name':'native','run_id':-1,'h_size':-1})

as_ids=np.argsort([x['model_name'] for x in model_hparams])
models=[models[j] for j in as_ids]
auto_regs=[auto_regs[j] for j in as_ids]
model_hparams=[model_hparams[j] for j in as_ids]


#evaluating the categorization and multiple choice tasks

n_obs=80
n_eval_samples=200 #samples per kernel used for evaluations
train_sample_range=[3,10,30,100,300]#different numbers of samples used for training
mc_res=[]
cl_res=[]

for task_id in range(10):
    ns=n_eval_samples+max(train_sample_range)
    y0s,ys,labs,y0s_obs,ys_inp,ys_mix,ys_comp,ys_rbf,ys_true_ker,ys_rsc=generate_curves(kernel_samples=ns,n_obs=n_obs)
    #y0s: raw curves 
    #ys: curves projected to [0,1] (only used for kernel classification)
    #labs: ground truth kernel class of each curves
    #y0s_obs: the first e.g. 80 points of each curve (NOT in [0,1])
    #ys_inp: ys_obs stretched out to 100 points using interpolation and projected to [0,1]
    #ys_mix: the mixture completion of the observed curve ys_inp
    #ys_comp: similar 
    #ys_rbf: similar
    #ys_true_ker: similar,but using the true generative kernel
    #ys_rsc: original curves, but rescaled to agree with the ys_inp etc.
    ys_comp=torch.from_numpy(ys_comp).float().to(device)
    ys_mix=torch.from_numpy(ys_mix).float().to(device)
    ys_inp=torch.from_numpy(ys_inp).float().to(device)
    ys_true_ker=torch.from_numpy(ys_true_ker).float().to(device)
    ys=torch.from_numpy(ys).float().to(device)
    
    assert torch.max(ys_comp)<1+10**-6
    assert torch.min(ys_comp)>-10**-6
    assert torch.max(ys_mix)<1+10**-6
    assert torch.min(ys_mix)>-10**-6
    assert torch.max(ys)<1+10**-6
    assert torch.min(ys)>-10**-6
    
    assert len(ys)==ns*14
    for model,hparams,autoreg in zip(models,model_hparams,auto_regs):
        #autoreg is "None" unless the model is cpc
        
        #classification accuracy
        clf=SGDClassifier(loss='log',alpha=10**-5)

        mn=hparams['model_name']
        print(mn)
        xj=hparams['x_jitter_strength'] if mn=='contrastive' else np.nan
        yj=hparams['y_jitter_strength'] if mn=='contrastive' else np.nan
        with torch.no_grad():
            inps=get_reps(model,ys,hparams,autoreg)
            inps=inps.cpu()
            h_obs=get_reps(model,ys_inp,hparams,autoreg)
            h_obs=h_obs.cpu()
            #h vector of the initial portion of the curve
            h_comp=get_reps(model,ys_comp,hparams,autoreg)
            h_comp=h_comp.cpu()
            h_mix=get_reps(model,ys_mix,hparams,autoreg)
            h_mix=h_mix.cpu()


        for n_train_samples in train_sample_range:

            #n_train_samples refers to average number of samples per kenel, so multiply by number of kernels
            tr_ids=np.arange(n_train_samples*14)
            val_ids=np.arange(n_train_samples*14,(n_train_samples+n_eval_samples)*14)

            clf=SGDClassifier(loss='log')
            grid_search =GridSearchCV(
                clf, {
                    'alpha': [10.0**i for i in range(-6,4)],
                },
                cv=5, n_jobs=5
            )
            grid_search.fit(inps[tr_ids],labs[tr_ids])
            clf = grid_search.best_estimator_


            preds=clf.predict(inps[val_ids])
            targets=labs[val_ids]

            train_score,test_score=clf.score(inps[tr_ids],labs[tr_ids]),clf.score(inps[val_ids],labs[val_ids])

            cl_res.append([test_score,mn,task_id,hparams['run_id'],hparams['h_size'],xj,yj,n_train_samples])





            #multiple choice completions
            
            #include curves that are only from composition or from mixture
            mc_ids=np.array([i for i,l in enumerate(labs) if l>2])
            #number of included mixture examples
            n_mc_mix=np.sum(labs[mc_ids]==13)
            #randomly select equal number of compositional examples
            mc_comp_ids=np.array([i for i in mc_ids if labs[i]!=13])[:n_mc_mix]
            mc_mix_ids=np.array([i for i in mc_ids if labs[i]==13])
            
            #construct balanced set of mutliple choice problems
            mc_ids=np.zeros(2*n_mc_mix).astype('int')
            mc_ids[::2]=mc_comp_ids
            mc_ids[1::2]=mc_mix_ids
            
            #further divide the set of multiple choice questions
            #into training and validation sets
            mc_tr=np.arange(0,n_train_samples*2)
            mc_val=np.arange((n_train_samples)*2,(n_train_samples+n_eval_samples)*2)
            #0 for comp and 1 for mixture
            multchoice_targets=1*(labs[mc_ids]==13)

            mc_head=train_mc(h_comp[mc_ids][mc_tr],h_mix[mc_ids][mc_tr],h_obs[mc_ids][mc_tr],multchoice_targets[mc_tr],device=device,
                            n_epochs=10,weight_decay=10**-6,batch_size=min(256,len(mc_ids[mc_tr])))

            inp_h=h_obs[mc_ids][mc_val].detach()
            mc_targets=multchoice_targets[mc_val]
            
            mc_head=mc_head.cpu()
            with torch.no_grad():
                choices=torch.cat([mc_head(hh.detach().cpu()).unsqueeze(0) for hh in [h_comp[mc_ids][mc_val],h_mix[mc_ids][mc_val]]],0)
                logits=torch.sum(mc_head(inp_h).unsqueeze(0)*choices,2).T


            probs=torch.softmax(logits,axis=1)
            for p,t,l in zip(probs,mc_targets,labs[mc_ids][mc_val]):
                ktype=['comp','mix'][t]
                
                mc_res.append([p[t].item(),mn,l,task_id,hparams['run_id'],hparams['h_size'],xj,yj,n_train_samples,ktype])

cl_res=pd.DataFrame(cl_res,columns=['score','model name','task id','run id','h size','xjitter','yjitter','train size'])
mc_res=pd.DataFrame(mc_res,columns=['pr correct','model name','kernel','task id','run id','h size','xjitter','yjitter','train size','ktype'])


print('categorization results:')
latex_table(cl_res,'score',ci='sem',sigfigs=2,mult_100=True)


rr=mc_res.groupby(['model name','run id','task id','train size']).mean().reset_index()
print('mc results:')
latex_table(rr,'pr correct',mult_100=True,ci='sem')


#show bias towards compositional vs mixture complextions
qq=mc_res.groupby(['ktype','model name','run id','task id','train size']).mean().reset_index()
qq=pd.pivot_table(index=['model name','run id','task id','train size'],columns='ktype',values='pr correct',data=qq)
qq=qq.reset_index()
qq['comp minus mix']=qq['comp']-qq['mix']
print('difference in accs on mc:')
latex_table(qq,'comp minus mix',mult_100=True,ci='sem')

#evaluation on the freeform task

n_train_samples=300 #samples for training logistic regressor
n_val_samples_range=[1,3,10,30,100]
n_test_samples=300
res2=[]
window_size=20 #autoregressino window



for task_id in range(10):
    ns=n_train_samples+max(n_val_samples_range)+n_test_samples
    y0s,ys,labs,y0s_obs,ys_inp,ys_mix,ys_comp,ys_rbf,ys_true_ker,ys_rsc=generate_curves(kernel_samples=ns,n_obs=n_obs)
    ys_comp=torch.from_numpy(ys_comp).float().to(device)
    ys_mix=torch.from_numpy(ys_mix).float().to(device)
    ys_inp=torch.from_numpy(ys_inp).float().to(device)
    ys=torch.from_numpy(ys).float().to(device)
    
    assert torch.max(ys_comp)<1+10**-6
    assert torch.min(ys_comp)>-10**-6
    assert torch.max(ys_mix)<1+10**-6
    assert torch.min(ys_mix)>-10**-6
    assert torch.max(ys)<1+10**-6
    assert torch.min(ys)>-10**-6
    
    assert len(ys)==ns*14
    
    #compute errors for "oracle" gp baseline
    for i in range(len(ys)):
        d=np.mean((ys_true_ker[i,n_obs:]-ys_rsc[i,n_obs:])**2)**.5
        c=np.corrcoef(ys_true_ker[i,n_obs:],ys_rsc[i,n_obs:])[0,1]
        res2.append([d,c,labs[i],'gp oracle',-1,task_id,-1,'true',False])

    
    for m_id,(model,hparams,autoreg) in enumerate(zip(models,model_hparams,auto_regs)):
        
        #first, fit the classifier on the training input curves 
        #maybe it makes mores sense to use k-means? might be fairer to models that had poor classification
        tr_ids=np.arange(n_train_samples*14)

        clf=SGDClassifier(loss='log')
        mn=hparams['model_name']
        with torch.no_grad():
            h_obs=get_reps(model,ys_inp,hparams,autoreg)
            h_obs=h_obs.cpu()
        #n_train_samples refers to average number of samples per kenel, so multiply by number of kernels
        clf=SGDClassifier(loss='log')
        grid_search =GridSearchCV(
            clf, {
                'alpha': [10.0**i for i in range(-6,4)],
            },
            cv=5, n_jobs=5
        )
        grid_search.fit(h_obs[tr_ids],labs[tr_ids])
        clf = grid_search.best_estimator_
        for n_val_samples in n_val_samples_range:

            val_ids=np.arange(n_train_samples*14,(n_train_samples+n_val_samples)*14)
            test_ids=np.arange((n_train_samples+n_val_samples)*14,(n_train_samples+n_val_samples+n_test_samples)*14)


            ext_targets_val=ys_rsc[val_ids,n_obs:]

            true_cl=np.zeros((len(val_ids),14))
            for i in range(len(val_ids)):
                true_cl[i,labs[val_ids][i]]=1
                
            cl_val=clf.predict(h_obs[val_ids])
            pr_val=clf.predict_proba(h_obs[val_ids])
            cl_test=clf.predict(h_obs[test_ids])
            pr_test=clf.predict_proba(h_obs[test_ids])
            for use_probs in [False]:
                #probs variant does pretty terribly, so not included
                #hierarchical lniear model, with group assignments predicted using the hidden rep
                regs,*_=fit_hlm(ys_rsc[val_ids],cl_val,probs=pr_val,window=window_size,n_clusters=14,use_probs=use_probs)
                preds=forecast_hlm(regs,ys_rsc[test_ids,:n_obs],cl_test,probs=pr_test,window=window_size,n_pts=20,use_probs=use_probs)

                # measure distance on testing set

                target_labels=['true','comp','mix']
                targets=[ys_rsc[test_ids,n_obs:],ys_comp[test_ids,n_obs:].cpu().numpy(),ys_mix[test_ids,n_obs:].cpu().numpy()]

                for target_name,target in zip(target_labels,targets):
                    for i in range(len(test_ids)):
                        d=np.mean((preds[i]-target[i])**2)**.5
                        c=np.corrcoef(preds[i],target[i])[0,1]
                        res2.append([d,c,labs[test_ids][i],mn,n_val_samples,task_id,hparams['run_id'],target_name,use_probs])

            if m_id==0:
                #comparison non-hierarchical autoregression, trained on same amount of data
                #m_id=0 ensures this is just run once 
                trval_ids=np.concatenate((tr_ids,val_ids))
                X,Y,_=split_ar_batch(ys_rsc[trval_ids],window=window_size)
                reg=Ridge(alpha=.0001)
                grid_search =GridSearchCV(
                    reg, {
                        'alpha': [10.0**i for i in range(-6,4)]
                    },
                    cv=5, n_jobs=5
                )
                grid_search.fit(X,Y)
                reg = grid_search.best_estimator_

                reg.fit(X,Y)
                preds_ar=forecast_hlm([reg],ys_rsc[test_ids,:n_obs],[0]*len(test_ids),window=window_size,n_pts=20,use_probs=False)

                for target_name,target in zip(target_labels,targets):
                    for i in range(len(test_ids)):
                        d=np.mean((preds_ar[i]-target[i])**2)**.5
                        c=np.corrcoef(preds_ar[i],target[i])[0,1]
                        res2.append([d,c,labs[test_ids][i],'ar',n_val_samples,task_id,hparams['run_id'],target_name,False])


            
res2=pd.DataFrame(res2,columns=['dist','corr','kernel','model','train size','task id','run id','target','use probs'])

print('freeform results, correlation:')
bb=res2[res2['target']=='true'].groupby(['model','train size','task id','run id']).mean().reset_index()
latex_table(bb,'corr',sigfigs=2,ci='sem',mult_100=True)
print('freeform results, mse:')
latex_table(bb,'dist',sigfigs=4,ci='sem',mult_100=False)





