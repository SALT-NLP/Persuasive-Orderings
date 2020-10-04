import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vae_train.vae_utils import *

class Encoder(nn.Module):
    def __init__(self, embedding_size=128, n_highway_layers=0, encoder_hidden_size=128, n_class=None, encoder_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.n_class = n_class

        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_layers = encoder_layers
        self.bidirectional = 2 if bidirectional else 1

        if n_class is None:
            self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=encoder_hidden_size,
                                num_layers=encoder_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(input_size=embedding_size + n_class, hidden_size=encoder_hidden_size,
                                num_layers=encoder_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, y=None, sent_len=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]


        if self.n_class is not None and y is not None:
            y = torch.cat([y]*seq_len, 1).view(batch_size,
                                               seq_len, self.n_class)
            x = torch.cat([x, y], dim=2)

        output, (h_n, c_n) = self.lstm(x)
        hidden = output[torch.arange(output.shape[0]), sent_len]
        return hidden


