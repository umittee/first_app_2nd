import torch.nn as nn
from torchvision.models import resnet18

class Resnet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 7)

    def forward(self, images):
        h = self.feature(images)
        outputs = self.fc(h)
        return outputs