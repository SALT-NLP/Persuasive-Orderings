import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vae_train.vae_utils import *

class Generator(nn.Module):
    def __init__(self, vocab_size, z_size=128, embedding_size=128, generator_hidden_size=128, n_class=None, generator_layers=1):
        super(Generator, self).__init__()
        self.n_class = n_class
        self.generator_hidden_size = generator_hidden_size
        self.generator_layers = generator_layers
        self.vocab_size = vocab_size
        self.z_size = z_size

        if n_class is None:
            self.lstm = nn.LSTM(input_size=embedding_size + z_size,
                                hidden_size=generator_hidden_size, num_layers=generator_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=embedding_size + z_size + n_class,
                                hidden_size=generator_hidden_size, num_layers=generator_layers, batch_first=True)

        self.linear = nn.Linear(generator_hidden_size, vocab_size)

    def forward(self, x, z, y=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.z_size)
        x = torch.cat([x, z], dim=2)

        if self.n_class is not None and y is not None:
            y = torch.cat([y] * seq_len, 1).view(batch_size,
                                                 seq_len, self.n_class)
            x = torch.cat([x, y], dim=2)

        output, hidden = self.lstm(x)
        
        output = self.linear(output)
        
        
        return output