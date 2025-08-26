import torch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, forecast_steps, num_layers=1):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.forecast_steps = forecast_steps
        self.output_size = output_size
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Decoder LSTM
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)

        # Output projection
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_input, teacher_forcing_ratio=0.5, decoder_targets=None):
        batch_size = encoder_input.size(0)
        device = encoder_input.device

        # Encoding
        _, (hidden, cell) = self.encoder(encoder_input)

        decoder_input = torch.zeros(batch_size, 1, self.output_size).to(device)  # Initial input = zeros
        outputs = []

        for t in range(self.forecast_steps):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            step_output = self.fc(out.squeeze(1))  # [batch, output_size]
            outputs.append(step_output.unsqueeze(1))

            # Scheduled teacher forcing
            if decoder_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = decoder_targets[:, t].unsqueeze(1)
            else:
                decoder_input = step_output.unsqueeze(1)

        return torch.cat(outputs, dim=1)  # [batch, forecast_steps, output_size]
