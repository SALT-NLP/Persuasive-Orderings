import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_train.vae_encoder import *
from vae_train.vae_classifier import *
from vae_train.vae_generator import *
from vae_train.vae_predictor import *
from vae_train.vae_utils import *

class HierachyVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size=128, n_highway_layers=0, encoder_hidden_size=128, encoder_layers=1, generator_hidden_size=128, generator_layers=1, z_size=128, n_class=None, out_class = None, bidirectional=False, pretrained_embedding=None, hard = True):
        super(HierachyVAE, self).__init__()
        self.z_size = z_size
        self.n_class = n_class

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if vocab_size is not None and pretrained_embedding is not None:
            pretrained_embedding = np.array(pretrained_embedding)
            self.embedding.weight.data.copy_(
                torch.Tensor(pretrained_embedding))

        self.encoder = Encoder(embedding_size, n_highway_layers, encoder_hidden_size,
                                     n_class=None, encoder_layers=encoder_layers, bidirectional=bidirectional)

        double = 2 if bidirectional else 1
        
        self.hidden_to_mu = nn.Linear(z_size + n_class, z_size)
        self.hidden_to_logvar = nn.Linear(z_size + n_class, z_size)
        
        self.hidden_linear = nn.Linear(double * encoder_hidden_size, z_size)
        
        self.classifier = nn.Linear(double * encoder_hidden_size, n_class)

        self.predictor = Predictor(n_class, out_class, z_size, hard)
        self.generator = Generator(
            vocab_size, z_size, embedding_size, generator_hidden_size, n_class, generator_layers)

    def encode(self, x, sent_len):
        x = self.embedding(x)
        encoder_hidden = self.encoder(x, sent_len = sent_len)    
        q_y = self.classifier(encoder_hidden)
        
        encoder_hidden = self.hidden_linear(encoder_hidden)
        
        return q_y, encoder_hidden
        
    def sample_gumbel(self, logits, eps=1e-8):
        U = torch.rand(logits.shape)
        if logits.is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, tau):
        y = logits + self.sample_gumbel(logits)
        return F.softmax(y / tau, dim=-1)

    def gumbel_softmax(self, logits, tau, hard=True):
        y = self.gumbel_softmax_sample(logits, tau)
        
        if not hard:
            return y
        
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, y.shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*y.shape)

        y_hard = (y_hard - y).detach() + y

        return y_hard.view(-1, y.shape[-1])

    def gaussian_sample(self, mu, logvar, batch_size):
        z = torch.randn([batch_size, self.z_size]).cuda()
        z = mu + z * torch.exp(0.5 * logvar)
        return z
    
    def forward(self, x, prob, tau, mask1, mask2, hard=True, y=None, doc_len = None, sent_len = None):
        batch_size = x.shape[0]
        seq_num = x.shape[1]
        seq_len = x.shape[2]
        
        n_labels = y.shape[-1]

        x = x.view(batch_size * seq_num, seq_len)
        y = y.view(batch_size*seq_num, self.n_class).float()
        mask1 = mask1.view(batch_size*seq_num).float()
        mask2 = mask2.view(batch_size * seq_num).float()

        sent_len = sent_len.view(batch_size * seq_num)

        q_y, encoder_hidden = self.encode(x, sent_len)
        y_sample = self.gumbel_softmax(q_y, tau, hard).float()
        y_in = y.float() * mask1.view(-1, 1) + y_sample * mask2.view(-1, 1)
        
        hidden = torch.cat([encoder_hidden, y_in], dim = -1)
        
        mu = self.hidden_to_mu(hidden)
        logvar = self.hidden_to_logvar(hidden)
        
        y_in2 = y.float() * mask1.view(-1,1) + F.softmax(q_y, dim=-1).float() * mask2.view(-1,1)
        y_in3 = F.softmax(q_y, dim=-1)

        z = self.gaussian_sample(mu, logvar, batch_size*seq_num)
        
        t, strategy_embedding = self.predictor(
            y_in2.view(batch_size, seq_num, self.n_class), encoder_hidden.view(batch_size, seq_num, self.z_size), doc_len)

        kld_z = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, 1).mean()
        prob = self.embedding(prob)
        logits = self.generator(prob, z, y_in)
        
        return logits, kld_z, q_y, F.softmax(q_y, dim=-1), t, strategy_embedding, y_in2, encoder_hidden









        
        
