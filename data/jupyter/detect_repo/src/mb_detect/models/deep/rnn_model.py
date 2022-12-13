import torch


class GatedCell(torch.nn.Module):
    """A single layer lstm prediction model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        window_size: int,
    ):
        """Create an LSTM prediction model.

        Args:
            input_size (int): The number of input dimensions or channels.
            hidden_size (int): The size of the LSTM cell.
            output_size (int): The number of output channels.
            window_size (int): How many future samples to predict.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.cell = torch.nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input: torch.Tensor):
        """Compute the lstm layer forward pass.

        Args:
            input (torch.Tensor): The input tensor of shape
                [batch_size, channels, time]

        Returns:
            torch.Tensor: Using the prediction window
                predictions of shape
                [batch_size, channels, window_size]
                are returned.
        """
        batch_size = input.shape[0]
        time_steps = input.shape[-1]

        # state = torch.zeros([batch_size, self.hidden_size])
        out = torch.zeros([batch_size, self.hidden_size])
        if input.device.type == "cuda":
            out = out.cuda()

        # input encoding
        for t in range(time_steps):
            out = self.cell(input[:, :, t], out)
            residual = input[:, :, t] + self.linear(out)

        # compute prediction
        prediction_list = [residual]
        for _ in range(self.window_size - 1):
            out = self.cell(prediction_list[-1], out)
            residual = prediction_list[-1] + self.linear(out)
            prediction_list.append(residual)

        prediction = torch.stack(prediction_list, dim=-1)
        return prediction
