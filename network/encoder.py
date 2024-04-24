import torch
import torch.nn as nn
import torch.nn.functional as F

    
class Featurizer(nn.Module):
    
    def __init__(self, inchannel, in_dim, patch_size):
        super(Featurizer, self).__init__()
        dim = in_dim
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.PReLU()
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.PReLU()
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.PReLU()

    def _get_final_flattened_size(self):
        
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            # out1 = self.relu1(self.conv1(x))
            # out2 = self.relu2(self.conv2(out1))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        # out1 = self.relu1(self.conv1(x))
        # out2 = self.relu2(self.conv2(out1))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))
        
        return out4
    

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.layers = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        scores = self.layers(features)
        return scores