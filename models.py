import torch

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