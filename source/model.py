import torch
import torch.nn as nn

class meanMLP(nn.Module):
    """
    meanMLP model for fMRI data.
    Expected input shape: [batch_size, time_length, n_components].
    Output: [batch_size, n_classes]

    Hyperparameters expected in model_cfg:
        dropout: float
        hidden_size: int
    Data info expected in model_cfg:
        input_size: int - input n_components
        output_size: int - n_classes
    """

    def __init__(self, model_cfg):
        super().__init__()

        input_size = model_cfg["input_size"]
        output_size = model_cfg["output_size"]
        dropout = model_cfg["dropout"]
        hidden_size = model_cfg["hidden_size"]

        layers = [
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size),
        ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # bs, tl, fs = x.shape  # [batch_size, time_length, input n_components]

        fc_output = self.fc(x)
        logits = fc_output.mean(1)
        return logits