import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.optim import Adam

from ...dataloader.model_loader import RNNPredictionLoader


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, predict_window=1, history_window=5, num_channels=[75] * 3, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], input_size, bias=True)
        self.predict_window = predict_window
        self.history_window = history_window
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def predict(self, x):
        tx = torch.Tensor(x.to_numpy()).T
        tx = tx[..., -self.history_window :]
        ty = self.forward(tx)
        y = ty.detach().numpy()
        return y

    def forward(self, x):
        y = x[..., -1:] + self.linear(self.tcn(x)[..., -1]).unsqueeze(-1)
        prediction_list = [y]
        context = torch.cat([x[..., 1:], y], -1)
        for _ in range(self.predict_window - 1):
            y = y + self.linear(self.tcn(context)[..., -1]).unsqueeze(-1)
            context = torch.cat([context[..., 1:], y], -1)
            prediction_list.append(y)
        return torch.cat(prediction_list, axis=-1)

    def fit(self, X, epochs=10, use_cuda=False, lr=0.1):
        if use_cuda:
            self.cuda()
        optimizer = Adam(self.parameters(), lr=lr)
        loss_fun = torch.nn.MSELoss(reduction="sum")
        dataloader = RNNPredictionLoader(
            X,
            context_length=self.history_window,
            prediction_length=self.predict_window,
        )
        # Training loop
        self.train()
        for epoch in range(1, epochs + 1):
            # Batch training
            for idx in range(len(dataloader)):
                train_data, target_predictions = dataloader[idx]
                train_data = torch.Tensor(train_data)
                target_predictions = torch.Tensor(target_predictions)
                if use_cuda:
                    train_data = train_data.cuda()
                    target_predictions = target_predictions.cuda()
                optimizer.zero_grad()
                # Forward pass
                predictions = self.forward(train_data)
                loss = loss_fun(predictions, target_predictions)
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.parameters(), 0.5)
                optimizer.step()
