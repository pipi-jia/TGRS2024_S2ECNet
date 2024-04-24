import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        # self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s): 
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * x + beta
        #return (1+gamma)*(x)+beta

    
class STN(nn.Module):
    def __init__(self, imdim, imsize, class_num):
        super(STN,self).__init__()    
        self.zdim = class_num*imsize*imsize
        self.localization=nn.Sequential(nn.Conv2d(imdim,8,kernel_size=5,padding=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(8,class_num,kernel_size=3,padding=1),
                                        nn.ReLU(True))
        self.fc_loc=nn.Sequential(nn.Linear(self.zdim,32),
                                  nn.ReLU(True),
                                  nn.Linear(32,3*2))
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0]))
        
    def forward(self,x):
        xs=self.localization(x)
        xs=xs.view(-1,self.zdim)
        theta=self.fc_loc(xs)
        theta=theta.view(-1,2,3)
        grid=F.affine_grid(theta,x.size())
        x=F.grid_sample(x,grid)
        return x

    
class Generator(nn.Module):
    def __init__(self, num_class, n=16, kernelsize=3, imdim=3, imsize=[13, 13], zdim=16, device=0):
        super().__init__()
        stride = (kernelsize-1)//2
        self.zdim = zdim
        self.imdim = imdim
        self.imsize = imsize
        self.device = device
        self.zdim = n

        # 提取空间信息
        self.conv_spa1 = nn.Conv2d(imdim, 3, 1, 1)
        self.conv_spa2 = nn.Conv2d(3, n, 1, 1)
        # 使用STN改变content
        self.stn = STN(3, imsize[0], num_class)
        # 提取光谱信息
        self.conv_spe1 = nn.Conv2d(imdim, n, imsize[0], 1)
        self.conv_spe2 = nn.ConvTranspose2d(n, n, imsize[0])       
        # 使用AdaIN改变style
        self.adain = AdaIN2d(n, n)

        # 还原维度
        self.conv1_stn = nn.Conv2d(n, n, kernelsize, 1, stride)
        self.conv2_stn = nn.Conv2d(n, imdim, kernelsize, 1, stride)
        self.conv1_adain = nn.Conv2d(n, n, kernelsize, 1, stride)
        self.conv2_adain = nn.Conv2d(n, imdim, kernelsize, 1, stride)       

    def forward(self, x):
        x_spa = self.conv_spa1(x)
        x_spe = self.conv_spe1(x)

        z = torch.randn(len(x), self.zdim).to(self.device)
        x_adain = self.adain(x_spe, z)
        x_adain = self.conv_spe2(x_adain)

        x_stn = self.stn(x_spa)
        x_stn = self.conv_spa2(x_stn)

        x_stn = F.relu(self.conv1_stn(x_stn))
        x_stn = torch.sigmoid(self.conv2_stn(x_stn))

        x_adain = F.relu(self.conv1_adain(x_adain))
        x_adain = torch.sigmoid(self.conv2_adain(x_adain))        

        return x_adain, x_stn      


if __name__=='__main__':
    x = torch.randn(32, 3, 13, 13).to('cuda')
    G_net = Generator(num_class=7).to('cuda')
    G_net.eval()
    y1, y2 = G_net(x)
    print(y1.shape, y2.shape)

