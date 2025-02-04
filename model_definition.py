# model_definition.py

import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    '''Input/Output Shapes:
        Input: (batch_size, seq_length, input_size)

        Output:
        outputs: (batch_size, seq_length, hidden_size * num_directions)
        hidden: (num_layers * num_directions, batch_size, hidden_size)
        cell: (num_layers * num_directions, batch_size, hidden_size)
    '''

    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        """
        Encoder module using LSTM.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            num_layers (int): Number of LSTM layers.
            bidirectional (bool): If True, use bidirectional LSTM.
        """
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=0.5)

    # X is  A tensor of shape (batch_size, seq_length, input_size) (e.g., (32, 81, 34)).
    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F).

        Returns:
            tuple: (outputs, hidden, cell)
        """
        outputs, (hidden, cell) = self.lstm(x)
        outputs = self.dropout(outputs)
        return outputs, hidden, cell


class Decoder(nn.Module):
    '''Input/Output Shapes:
        Input: (batch_size, seq_length, input_size)


        Output:
        predictions: (batch_size, seq_length, output_size)
        hidden: (num_layers * num_directions, batch_size, hidden_size)
        cell: (num_layers * num_directions, batch_size, hidden_size)'''

    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional=True):
        """
        Decoder module using LSTM.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            output_size (int): Number of output features/classes.
            num_layers (int): Number of LSTM layers.
            bidirectional (bool): If True, use bidirectional LSTM.
        """
        super(Decoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        # Fully Connected Layer (fc): Maps hidden units to output units
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, hidden, cell):
        """
        Forward pass for the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, F).
            hidden (torch.Tensor): Hidden state.
            cell (torch.Tensor): Cell state.

        Returns:
            tuple: (predictions, hidden, cell)
        """
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        outputs = self.dropout(outputs)
        predictions = self.fc(outputs)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.3):
        """
        Seq2Seq model combining encoder and decoder.

        Args:
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            device (torch.device): Device to run the model on.
            teacher_forcing_ratio (float): Probability to use teacher forcing.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, source, target=None):
        """

        Forward pass for the Seq2Seq model.

        Args:
            source (torch.Tensor): Source tensor of shape (B, T, F).
            target (torch.Tensor, optional): Target tensor for teacher forcing. Defaults to None.

        Returns:
            torch.Tensor: Predictions tensor of shape (B, T, output_size).
        """
        # Runs the encoder over the entire input sequence. Returns final hidden/cell states for the decoder.
        encoder_outputs, hidden, cell = self.encoder(source)

        batch_size = source.size(0)
        seq_length = source.size(1)
        output_size = self.decoder.fc.out_features

        # An empty tensor to store the decoder’s outputs over each time step
        predictions = torch.zeros(
            batch_size, seq_length, output_size).to(self.device)

        # Initialize with first input at First timestep
        decoder_input = source[:, :1, :]

        if target is not None:
            # Training mode with teacher forcing
            for t in range(seq_length):
                output, hidden, cell = self.decoder(
                    decoder_input, hidden, cell)
                predictions[:, t:t + 1, :] = output
                # Decide whether to use teacher forcing
                if random.random() < self.teacher_forcing_ratio:
                    if target.dim() == 3:
                        # Use the true target
                        decoder_input = target[:, t:t + 1, :]
                    else:
                        # If targets are class indices
                        decoder_input = output
                else:
                    # Use the model's own prediction
                    decoder_input = output
        else:
            # Inference mode without teacher forcing
            for t in range(seq_length):
                output, hidden, cell = self.decoder(
                    decoder_input, hidden, cell)
                predictions[:, t:t + 1, :] = output
                decoder_input = output  # Next input is current prediction

        # predictions: Shape (batch_size, seq_length, output_size).
        return predictions
