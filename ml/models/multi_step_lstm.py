import torch
import torch.nn as nn

class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, forecast_steps):
        super(MultiStepLSTM, self).__init__()
        self.forecast_steps = forecast_steps
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, forecast_steps * output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # last timestep's hidden output
        out = self.fc(last_hidden)
        return out.view(-1, self.forecast_steps, self.output_size)  # [B, 6, 5]
