import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import squareform,pdist,cdist
import pyGPs

#a fixed ordering of the kernel names
KERNELNAMES=['linear','rbf','periodic',
             'linear+periodic','linear+rbf','rbf+periodic',
             'linear*periodic','linear*rbf','periodic*rbf',
             'linear+rbf+periodic','periodic*rbf+linear','linear*rbf+periodic',
             'linear*rbf*periodic','mix']
#the depth of each kernel, defined as the number of atomic pieces
KERNELDEPTHS=[1]*3+[2]*6+[3]*4+[-1]

class Linear2(pyGPs.Core.cov.Kernel):
    '''
    Linear kernel. hyp = [ theta ].
    '''
    def __init__(self, theta=0.):
        self.hyp = [ theta ]

    def getCovMatrix(self,x=None,z=None,mode=None):
        self.checkInputGetCovMatrix(x,z,mode)
        t=self.hyp[0]
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.reshape(np.sum((z-t)*(z-t),1), (nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            n,D = x.shape
            A = np.dot(x-t,(x-t).T) + np.eye(n)*1e-10    # required for numerical accuracy
        elif mode == 'cross':             # compute covariance between data sets x and z
            A = np.dot(x-t,(z-t).T)
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        #(xi-t)(xj-t); t^2-t(xi+xj)
        self.checkInputGetDerMatrix(x,z,mode,der)
        t=self.hyp[0]
        if der == 0:
            if mode == 'self_test':           # self covariances for the test cases
                nn,D = z.shape
                A=z+z.T #A_{ij}=z_i+z_j (broadcasting)
            elif mode == 'train':             # compute covariance matix for dataset x
                n,D = x.shape
                A=x+x.T + np.eye(n)*1e-16    # required for numerical accuracy
            elif mode == 'cross':             # compute covariance between data sets x and z
                A = x+z.T
            A = t*t-A
        else:
            raise Exception("Wrong derivative index in covLinear")
        return A

def get_cov_pygp(k_id):
    #returns pygp class of the given kernel type; used to optimize the hyperparameters
    #can pass either id or one of the strings in KERNELNAMES

    #initialize hyperparameters from same distribution-helps with optimization


    l=Linear2(theta=np.random.randn()*2)
    r=pyGPs.cov.RBF(log_ell=np.log(np.random.uniform(1,5)),log_sigma=.5*np.log(np.random.uniform(1,3)))
    #1/p=2theta5; logp=-log(2theta5)
    #2/theta6^2=1/l^2; log(l)=-.5*log (2/theta6^2)
    p=pyGPs.cov.Periodic(log_p=-np.log(np.random.rand()),log_ell=-.5*np.log(2/np.random.uniform(1,5)**2),log_sigma=.5*np.log(np.random.uniform(1,3)))
    if k_id==0:
        k = l
    elif k_id==1:
        k=r
    elif k_id==2:
        k=p
    elif k_id==3: #l+p
        k=l+p
    elif k_id==4: #l+r
        k=l+r
    elif k_id==5: #r+p
       k=r+p
    elif k_id==6: #l*p
        k=l*p
    elif k_id==7: #l*r
        k=l*r
    elif k_id==8: #p*r
        k=p*r
    elif k_id==9: #l+r+p
        k=l+r+p
    elif k_id==10: #p*r+l
        k=p*r+l
    elif k_id==11:#l*r+p
        k=l*r+p
    elif k_id==12: #l*r*p
        k=l*r*p
    elif k_id==13:
        w=np.random.rand(6)
        m=np.random.rand(6)*.01
        v=np.random.rand(6)*.02
        hyp=np.concatenate((np.log(w),np.log(m),.5*np.log(v)))
        k=pyGPs.cov.SM(Q=6,hyps=list(hyp))

    return k+pyGPs.cov.Noise(log_sigma=-4*np.log(10)) #add noise term for numerical stability


def spectral_ent(C):
    evals=np.linalg.svd(C)[1]
    evals_n=evals/np.sum(evals)
    spectral_ent=-np.sum(evals_n*np.log(evals_n)) #measure of predictability; the higher this entropy, the more like white noise
    return spectral_ent

def sample_gaussian(n,m=None,C=None,use_svd=False):
    #n: number of samples to take
    #m: array of shape (n), giving mean for each point (if None, will be zero)
    #C: array of shape (n,n), giving covariance (if None, will be identity)
    #if use_svd, then will compute diagonalization of C
    #can be faster depending on number of curves to sample
    if use_svd:
      eigs,evals,_=np.linalg.svd(C,full_matrices=True)
      #each column of eigs is an eigenvector
      w=np.random.randn(n,len(evals))
      wts=w*evals**.5
      return np.dot(wts,eigs.T)

    return multivariate_normal.rvs(mean=m,cov=C,size=n)

class Kernel():
    #base class for a kernel, not meant to be instantiated directly
    def __init__(self,**kwargs):
        raise NotImplementedError()
    def id(self):
        #unique numerical id for each kernel type
        return KERNELNAMES.index(self.name)
    def depth(self):
        return KERNELDEPTHS[self.id()]
    def cov(self,x,y):
        #given vectors x and y of possibly different lengths, return the covariance matrix K(x_i,y_j)
        raise NotImplementedError()
    def posterior(self,x_obs,y_obs,x_test,K11_inverse=None,inv_reg=0):
        #given array x_obs and values y_obs, return the mean and covariance matrix of the posterior at x_test
        #can optionally give precomputed value for K11_inverse= K(x_obs,x_obs)^{-1}; if None then will be computed directly
        #inv_reg: regularization for K(x,x)+inv_reg for computing inverse
        #is ignored if K11_inverse is provided
        K12 = self.cov(x_obs, x_test)
        # compute the inverse covariance matrix of the data points, if not suppled
        K11 = self.cov(x_obs, x_obs) if K11_inverse is None else 0
        K11_inv = np.linalg.inv(K11+inv_reg*np.eye(len(x_obs))) if K11_inverse is None else K11_inverse
        K22 = self.cov(x_test, x_test)
        mu = np.dot(K12.T, np.dot(K11_inv, y_obs))
        sigma = K22 - np.dot(K12.T, np.dot(K11_inv, K12))
        return mu, sigma

    def llh(self,x_obs, y_obs,inv_reg=0):
        #the log-likelihood of the observations
        C=self.cov(x_obs,x_obs)+inv_reg*np.eye(len(x_obs))
        return -.5*np.sum(y_obs*np.dot(np.linalg.inv(C),y_obs))-.5*np.linalg.slogdet(C)[1]-.5*len(x_obs)*np.log(2*np.pi)


class WhiteNoiseKernel(Kernel):
    def __init__(self,sigma=1):
        self.sigma=sigma
        self.name='white noise'
    def cov(self,x,y):
        dist=cdist(x[:, None], y[:, None])
        return np.where(dist<10**-8,self.sigma,0)

class SumKernel(Kernel):
    #a class for a kernel defined as the sum of two other kernels
    def __init__(self,K1,K2):
        self.K1=K1
        self.K2=K2
        self.name=K1.name+'+'+K2.name
    def cov(self,x,y):
        return self.K1.cov(x,y)+self.K2.cov(x,y)

class ProductKernel(Kernel):
    def __init__(self,K1,K2):
        self.K1=K1
        self.K2=K2
        self.name=K1.name+'*'+K2.name
    def cov(self,x,y):
        return self.K1.cov(x,y)*self.K2.cov(x,y)

class RBFKernel(Kernel):
    def __init__(self,sigma=1,scaling=1):
        self.sigma=sigma
        self.scaling=scaling
        self.name='rbf'
    def cov(self,x,y):
        dist = cdist(x[:, None], y[:, None])
        return self.scaling*np.exp(-dist ** 2 / self.sigma ** 2)

class LinearKernel(Kernel):
    def __init__(self,theta=0):
        self.theta=theta
        self.name='linear'
    def cov(self,x,y):
        return np.outer(x-self.theta,y-self.theta)

class PeriodicKernel(Kernel):
    def __init__(self,frequency=1,sigma=1,scaling=1):
        self.frequency=frequency
        self.sigma=sigma
        self.scaling=scaling
        self.name='periodic'
    def cov(self,x,y):
        abs_diffs=np.abs(np.add.outer(x,-y))
        return self.scaling*np.exp(-2*np.sin(np.pi*abs_diffs*self.frequency)**2/self.sigma**2)

class SpectralMixtureKernel(Kernel):
    def __init__(self,means=None,covs=None,weights=None):
        #let q = number of mixture components
        #means: array of len q, giving mean of each component in mixture
        #covs=array of len q, giving variance of each component in mixture
        #w=array of len q, giving weighting of each component in mixture
        self.means=means
        self.covs=covs
        self.weights=weights
        self.name='mix'
    def cov(self,x,y):
        tau=np.add.outer(x,-y)
        k=np.zeros((len(x),len(y)))
        for w,mu,v in zip(self.weights,self.means,self.covs):
            k=k+w*np.exp(-2*v*(np.pi*tau)**2)*np.cos(2*np.pi*tau*mu)
        return k

def sample_comp_kernel(k_i=None):
    k_id = np.random.choice(13) if k_i is None else k_i
    lin_params = dict({})
    lin_params['theta'] = np.random.randn() * 2
    rbf_params = dict({})
    rbf_params['sigma'] = 1 + np.random.rand() * 4
    rbf_params['scaling'] = 1 + 2 * np.random.rand()
    periodic_params = dict({})
    periodic_params['scaling'] = 1 + 2 * np.random.rand()
    periodic_params['frequency'] = .5 + np.random.rand()
    periodic_params['sigma'] = 1 + np.random.rand() * 4
    hparams = dict({})
    hparams['linear'] = lin_params
    hparams['rbf'] = rbf_params
    hparams['periodic'] = periodic_params
    KL = LinearKernel(**hparams['linear'])
    KP = PeriodicKernel(**hparams['periodic'])
    KRBF = RBFKernel(**hparams['rbf'])
    comps = get_comp_kernels(KL, KRBF, KP)
    K = comps[k_id]
    noise = np.random.rand()
    multiplier = .005
    # if K.name=='linear':
    #    multiplier=1
    # elif K.name=='periodic':
    #    multiplier=.1
    noise = noise * multiplier
    return K, noise


def get_comp_kernels(linear_kernel, rbf_kernel, periodic_kernel):
    # returns a list of kernels generated according to the grammar in the schulz paper
    # the three arguments should be instances of the respective 3 classes
    # the ordering is the same as in the KERNELNAMES variable
    lin = linear_kernel
    rbf = rbf_kernel
    per = periodic_kernel  # aliases for typing convenience
    kernels = [lin, rbf, per]
    kernels.append(SumKernel(lin, per))
    kernels.append(SumKernel(lin, rbf))
    kernels.append(SumKernel(rbf, per))
    kernels.append(ProductKernel(lin, per))
    kernels.append(ProductKernel(lin, rbf))
    kernels.append(ProductKernel(per, rbf))
    kernels.append(SumKernel(SumKernel(lin, rbf), per))
    kernels.append(SumKernel(ProductKernel(per, rbf), lin))
    kernels.append(SumKernel(ProductKernel(lin, rbf), per))
    kernels.append(ProductKernel(ProductKernel(lin, rbf), per))
    assert [k.name for k in kernels] == KERNELNAMES[:-1]
    return kernels

def sample_mix_kernel():
    q = np.random.choice(np.arange(2, 7))
    means = .01 * np.random.rand(5)
    covs = np.random.rand(q) * .02
    weights = np.random.rand(q) * 1
    # weights=weights**10/np.sum(weights**10)
    hparams = dict({})
    hparams['means'] = means
    hparams['covs'] = covs
    hparams['weights'] = weights
    K = SpectralMixtureKernel(**hparams)
    noise = np.random.rand() * .005
    return K, noise
