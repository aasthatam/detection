import torch.nn as nn
from torchvision.models import resnet50

class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        base_model = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(2048, 128)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
