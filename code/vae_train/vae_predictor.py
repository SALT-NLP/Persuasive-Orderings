import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vae_train.vae_utils import *

class Predictor(nn.Module):
    def __init__(self, n_class, out_class, z_size ,hard = True, hidden_size=64, num_layers=1, bidirectional=False):
        super(Predictor, self).__init__()
        self.z_size = z_size
        self.bidirectional = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hard = hard
        if hard:
            self.embedding = nn.Linear(n_class, hidden_size, bias=False)
            self.lstm = nn.LSTM(input_size=n_class + z_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(input_size=n_class + z_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        self.predict = nn.Linear(hidden_size, out_class)

    def forward(self, x, z, doc_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        if self.hard:
            x = torch.cat([x, z], dim=2)
            output, (h_n, c_n) = self.lstm(x)
            hidden = output[torch.arange(output.shape[0]), doc_len.view(-1)]
            return self.predict(hidden), self.embedding.weight
        else:
            x = torch.cat([x, z], dim=2)
            output, (h_n, c_n) = self.lstm(x)
            hidden = output[torch.arange(output.shape[0]), doc_len.view(-1)]
            return self.predict(hidden), None