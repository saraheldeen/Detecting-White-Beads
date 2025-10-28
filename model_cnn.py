import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNetwork(nn.Module):
    """
    Matches the architecture used in the PDF:
    conv3x3 -> pool2 -> conv3x3 -> pool2 -> flatten -> FC(120) -> FC(84) -> FC(2)
    For 300x300 input, the flatten size is 73*73*16.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1)   # 3->6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1)  # 6->16
        self.fc1 = nn.Linear(73*73*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 73*73*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
