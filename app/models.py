import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from constants import *

def check_frozen_layers(model):
    def count_parameters(model: nn.Module):
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        non_frozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return frozen_params, non_frozen_params
    
    frozen_params, non_frozen_params = count_parameters(model)
    print(f"Number of frozen parameters: {frozen_params}")
    print(f"Number of non-frozen parameters: {non_frozen_params}")
    print(f"Proportion of frozen parameters: {frozen_params / (frozen_params + non_frozen_params)}")

class ResNet18Regressor(nn.Module):
    def __init__(self, num_outputs=1, bias=None):
        super(ResNet18Regressor, self).__init__()
        
        # Load the pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        
        # Remove the last fully connected layer (classification layer)
        # It consists of several layers, including Convolution, BatchNorm, ReLU, etc.
        # We remove the final fully connected layer.
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Append a new fully connected layer for regression
        self.fc = nn.Linear(512, num_outputs)

        if bias is not None:
            self.fc.bias.data.fill_(bias)
        
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet50Regressor(nn.Module):
    def __init__(self, num_outputs=1, bias=None):
        super(ResNet50Regressor, self).__init__()
        
        # Load the pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)
        
        # Remove the last fully connected layer (classification layer)
        # It consists of several layers, including Convolution, BatchNorm, ReLU, etc.
        # We remove the final fully connected layer.
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Append a new fully connected layer for regression
        self.fc = nn.Linear(2048, num_outputs)

        if bias is not None:
            self.fc.bias.data.fill_(bias)
        
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = F.sigmoid(self.fc(x))
        return x

class SuperBasic(nn.Module):
    def __init__(self, bias=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc_out = nn.Linear(84, 1)

        if bias is not None:
            self.fc_out.bias.data.fill_(bias)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class ImageSize128(nn.Module):
    def __init__(self, bias=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)

        self.fc1 = nn.Linear(2304, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc_out = nn.Linear(84, 1)

        if bias is not None:
            self.fc_out.bias.data.fill_(bias)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x

class ImageSize224(nn.Module):
    def __init__(self, bias=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)

        self.fc1 = nn.Linear(9216, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc_out = nn.Linear(84, 1)

        if bias is not None:
            self.fc_out.bias.data.fill_(bias)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class Chonko224(nn.Module):
    def __init__(self, bias=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 128, 5)

        self.fc1 = nn.Linear(12800, 224)
        self.fc2 = nn.Linear(224, 84)
        self.fc_out = nn.Linear(84, 1)

        if bias is not None:
            self.fc_out.bias.data.fill_(bias)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc_out(x))
        return x

def vit_model(model_name, bias=None, freeze=True, hidden_head_size=None):
    model = timm.create_model(model_name, pretrained=True)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    if hidden_head_size is not None:
        fc1 = nn.Linear(model.head.in_features, hidden_head_size)
        fc2 = nn.Linear(hidden_head_size, 1)

        if bias is not None:
            fc2.bias.data.fill_(bias)

        model.head = nn.Sequential(fc1, fc2)
    else: 
        fc = nn.Linear(model.head.in_features, 1)

        if bias is not None:
            fc.bias.data.fill_(bias)

        model.head = nn.Sequential(fc, nn.Sigmoid())
    
    return model

def build_model(model_name, bias, freeze, hidden_head_size):
    if model_name == 'super-basic-convnet':
        m = SuperBasic(bias=bias)
    elif model_name == '128-convnet':
        m = ImageSize128(bias=bias)
    elif model_name == '224-convnet':
        m = ImageSize224(bias=bias)
    elif model_name == '224-chonk':
        m = Chonko224(bias=bias)
    elif model_name == 'resnet-18':
        m = ResNet18Regressor(bias=bias)
    elif model_name == 'resnet-50':
        m = ResNet50Regressor(bias=bias)
    elif model_name in TIMM_MODELS:
        m = vit_model(model_name, bias, freeze, hidden_head_size)
    else: 
        raise Exception('unknown model: ' + model_name)
    
    check_frozen_layers(m)

    return m