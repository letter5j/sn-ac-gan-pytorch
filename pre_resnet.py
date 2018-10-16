import torch
import torch.nn as nn
from torch.autograd import Variable
import pretrainedmodels

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False
def build_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrained model


    model_name = 'resnet152'
    model = pretrainedmodels.resnet152(pretrained='imagenet')
    
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.last_linear.in_features
    class CustomModel(nn.Module):
        def __init__(self, model):
            super(CustomModel, self).__init__()
            self.features = nn.Sequential(*list(model.children())[:-1]  )
            self.classifier = nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.Dropout(0.3),  # drop 50% of the neuron
                torch.nn.Linear(128, 7)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    model = CustomModel(model)
    freeze_layer(model.features)
    num_ftrs = list(model.classifier.children())[-1].out_features

    model.load_state_dict(torch.load('resnet152.pth'))
    model.to(device)
    model.name = model_name
    return model, num_ftrs
