from torch import nn

class ModelBaseline(nn.Module):
    def __init__(self):
        super(ModelBaseline, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(90, 180),
            nn.ReLU(),
            nn.Linear(180, 360),
            nn.ReLU(),
            nn.Linear(360, 90),
            nn.ReLU(),
            nn.Linear(90, 45),
            nn.ReLU(),
            nn.Linear(45, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class LargeParamModel(nn.Module):
    def __init__(self):
        super(LargeParamModel, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(90, 180),
            nn.ReLU(),
            nn.Linear(180, 720),
            nn.ReLU(),
            nn.Linear(720, 2040),
            nn.ReLU(),
            nn.Linear(2040, 2040),
            nn.ReLU(),
            nn.Linear(2040, 1024),
            nn.ReLU(),
            nn.Linear(1024, 720),
            nn.ReLU(),
            nn.Linear(720, 360),
            nn.ReLU(),
            nn.Linear(360, 90),
            nn.ReLU(),
            nn.Linear(90, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits