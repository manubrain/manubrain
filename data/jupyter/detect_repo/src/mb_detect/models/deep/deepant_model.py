import torch
import torch.nn as nn


class DeepAntNet(nn.Module):
    """Network as specified in DeepAnt model.

    Two convolution layers followed
    by a linear layer with ReLU activation and maxpool after every conv-
    olutional layer
    """

    def __init__(self, predict_window, n_dim) -> None:
        super(DeepAntNet, self).__init__()
        self.predict_window = predict_window
        # self.lin1 = torch.nn.Linear(32, n_dim, bias=False)
        self.features = nn.Sequential(
            nn.Conv1d(n_dim, 32, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(1),
            nn.Conv1d(32, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Linear(16, n_dim)
        )

    def forward(self, x):
        """Forward pass connections for the model.

        Args:
            x (Tensor): Tensor containing the input values

        Returns:
            [Tensor]: Output of the network
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = x.float()
        print("THIS IS IT", x.dtype)
        f = self.features(x)[..., -1]
        y = x[..., -1:] + self.classifier(f).unsqueeze(-1)
        prediction_list = [y]
        context = torch.cat([x[..., 1:], y], -1)
        for _ in range(self.predict_window - 1):
            y = y + self.classifier(self.features(context)[..., -1]).unsqueeze(-1)
            context = torch.cat([context[..., 1:], y], -1)
            prediction_list.append(y)
        return torch.cat(prediction_list, axis=-1)
