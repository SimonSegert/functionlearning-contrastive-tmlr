import torch
class CNP(torch.nn.Module):
    def __init__(self,h_size=128):
        super(CNP,self).__init__()

        self.h_size=h_size
        self.enc=CNPEncoder(h_size=h_size)
        #for each query point, decodes mean and variance conditional on the "r" vector
        self.dec=torch.nn.Sequential(*[torch.nn.Linear(h_size+1,32),torch.nn.LeakyReLU(),
                                       torch.nn.Linear(32,2)])
    def forward(self,xobs,yobs,xt):
        #mean and logsigma of each point at xt, conditional on observed points

        #xobs:(n sequences) x (n observed points)
        #yobs: (n sequences) x (n observed points)
        #xt: (n sequences) x (n query points)
        #all sequences are assumed to have same number of observed/query points, but
        #locations of these points can vary

        #(n sequences) x (h size)
        r=self.enc(yobs,x_positions=xobs)
        mu=[]
        logsigma=[]
        #decode each point one by one
        for ii in range(xt.shape[1]):
            inp=torch.cat((r,xt[:,ii].unsqueeze(1)),1)
            phi=self.dec(inp)
            mu.append(phi[:,0][:,None])
            logsigma.append(phi[:,1][:,None])
        mu=torch.cat(mu,axis=1)
        logsigma=torch.cat(logsigma,axis=1)
        return mu,logsigma

class CNPEncoder(torch.nn.Module):
    #permutation-invariant encoder used in Neural Processes
    #for simplicity, we assume that the position of observation points is always constant
    #so we do not include it as input
    def __init__(self,h_size=128):
        super(CNPEncoder,self).__init__()
        self.h_size=h_size
        self.bn0=torch.nn.BatchNorm1d(2)
        self.mlp=torch.nn.Sequential(*[torch.nn.Linear(2,64),torch.nn.LeakyReLU(),
                                       torch.nn.BatchNorm1d(64),
                                       torch.nn.Linear(64,128),torch.nn.LeakyReLU(),
                                       torch.nn.BatchNorm1d(128),
                                       torch.nn.Linear(128,h_size)])

    def forward(self,x,x_positions=None):
        #input should be shape (batch size) x (sequence length)
        #xpositions should be (batch size) x (sequence length)
        #defaults to uniform grid on [0,1]
        seq_len=x.shape[1]
        #create an array of x positions
        #since all positions are assumed to be constant over different sequences, we use dummy coding of pos integers
        if x_positions is None:
            xpos=1.0*torch.arange(seq_len).to(x.device)/seq_len
            xpos=torch.cat([xpos for _ in range(len(x))])
            xpos=xpos.unsqueeze(1)
        else:
            xpos=1.0*torch.cat([a for a in x_positions])
            xpos=xpos.unsqueeze(1)
        #all entries from first sequence, then all entries of second sequence, etc.
        xf=torch.cat([x[i] for i in range(x.shape[0])])
        xf=xf.unsqueeze(1)

        xf=torch.cat((xf,xpos),axis=1)
        xf=self.bn0(xf)

        rn=self.mlp(xf)
        #average over entries of each sequence
        r=torch.cat([torch.mean(rn[i:i+seq_len,:],axis=0).unsqueeze(0) for i in range(0,len(xf),seq_len)],axis=0)
        return r





class ConvEncoder(torch.nn.Module):
    # reconstruct an image, but using the inductive bias of passing through a pattern completion module
    def __init__(self, convolutional=False,h_size=512):
        super(ConvEncoder, self).__init__()
        self.convolutional = convolutional
        self.bn0 = torch.nn.BatchNorm1d(100)
        self.h_size=h_size
        if convolutional:
            self.seq1 = torch.nn.Sequential(*[torch.nn.Conv1d(1, out_channels=64, kernel_size=5, stride=2),
                                              torch.nn.MaxPool1d(2), torch.nn.LeakyReLU(),
                                              torch.nn.BatchNorm1d(64),
                                              torch.nn.Conv1d(64, out_channels=64, kernel_size=5, stride=1),
                                              torch.nn.MaxPool1d(2), torch.nn.LeakyReLU(),
                                              torch.nn.BatchNorm1d(64),
                                              torch.nn.Conv1d(64, out_channels=64, kernel_size=3, stride=1),
                                              torch.nn.Flatten(),
                                              torch.nn.LeakyReLU(),
                                              torch.nn.Linear(512,h_size)
                                                            ])

        else:
            self.seq1 = torch.nn.Sequential(*[torch.nn.Linear(100, 32), torch.nn.LeakyReLU(),
                                              torch.nn.Linear(32, 32), torch.nn.LeakyReLU(),
                                              torch.nn.Linear(32, 512)])

    def forward(self, x):
        x=self.bn0(x)
        x=x.unsqueeze(1)
        return self.seq1(x).squeeze(1)
class VAE(torch.nn.Module):
    def __init__(self,h_size=512):
        super(VAE, self).__init__()
        self.h_size=h_size
        self.bn0 = torch.nn.BatchNorm1d(100)
        self.conv1=torch.nn.Conv1d(1, out_channels=64, kernel_size=5, stride=2)
        self.mp1=torch.nn.MaxPool1d(2,return_indices=True)
        self.relu1=torch.nn.LeakyReLU()
        self.bn1=torch.nn.BatchNorm1d(64)
        self.conv2=torch.nn.Conv1d(64, out_channels=64, kernel_size=5, stride=1)
        self.mp2=torch.nn.MaxPool1d(2,return_indices=True)
        self.relu2=torch.nn.LeakyReLU()
        self.bn2=torch.nn.BatchNorm1d(64)
        self.conv3=torch.nn.Conv1d(64, out_channels=64, kernel_size=3, stride=1)
        self.to_h=torch.nn.Linear(512,h_size)
        self.to_logsigma=torch.nn.Linear(512,h_size)
        self.from_h=torch.nn.Linear(h_size,512)
        self.deconv3=torch.nn.ConvTranspose1d(64,out_channels=64,kernel_size=3,stride=1)
        self.relu1t=torch.nn.LeakyReLU()
        self.mpt2=torch.nn.MaxUnpool1d(2)
        self.deconv2=torch.nn.ConvTranspose1d(64,out_channels=64,kernel_size=5,stride=1)
        self.relu2t=torch.nn.LeakyReLU()
        self.mpt1=torch.nn.MaxUnpool1d(2)
        self.deconv1=torch.nn.ConvTranspose1d(64,out_channels=1,kernel_size=6,stride=2)

    def forward(self,x):
        eps=torch.randn((len(x),self.h_size)).to(x.device)
        x=self.bn0(x)
        x=x.unsqueeze(1)
        x=self.conv1(x)
        x,ids1=self.mp1(x)
        x=self.relu1(x)
        x=self.bn1(x)
        x=self.conv2(x)
        x,ids2=self.mp2(x)
        x=self.relu2(x)
        x=self.bn2(x)
        x=self.conv3(x).reshape((len(x),-1))
        h=self.to_h(x)
        logsigma=self.to_logsigma(x)
        x=h+eps*torch.exp(logsigma)
        x=self.from_h(x)
        x=x.reshape((len(x),64,8))
        x=self.deconv3(x)
        x=self.relu2t(x)
        x=self.mpt2(x,indices=ids2)
        x=self.deconv2(x)
        x=self.relu1t(x)
        x=self.mpt1(x,indices=ids1)
        x=self.deconv1(x)
        return x.squeeze(1),h,logsigma



