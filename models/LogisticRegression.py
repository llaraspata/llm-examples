import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, use_bias):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=use_bias)


    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(0)


    @torch.inference_mode()
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.linear.weight.device)

        print(f"output predict : {torch.sigmoid(self.linear(x)).squeeze(0)}")
        return torch.sigmoid(self.linear(x)).squeeze(0)
    