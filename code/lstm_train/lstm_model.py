import torch
import torch.nn as nn
import torch.nn.functional as F

d1 = 64
d2 = 128
d3 = 256
d4 = 512

class AttnContentStrategy(nn.Module):
    def __init__(self, n_labels):
        super(AttnContentStrategy, self).__init__()
        self.linearStrategy = nn.Linear(n_labels, d1)
        self.linearStrategy2 = nn.Linear(d1, d1)
        
        self.linearContent = nn.Linear(d1, d1)
        self.linearContent2 = nn.Linear(d1, d1)
        
        self.strategyContentContext = nn.Parameter(torch.randn([d1, 1]).float())
        self.content_proj = nn.Linear(d1, d1)
        self.strat_proj = nn.Linear(d1, d1)

        self.linear1 = nn.Linear(d2, d3)
        self.linear2 = nn.Linear(d3, d4)
        self.lstm = nn.LSTM(d4, d4)
        self.s_proj = nn.Linear(d4, d4)
        self.softmax = nn.Softmax(dim = 1)

        self.s_context_vector = nn.Parameter(torch.randn([d4, 1]).float())
        self.sent_linear1 = nn.Linear(d4, d3)
        self.sent_linear2 = nn.Linear(d3, d2)
        self.sent_linear3 = nn.Linear(d2, 1)

    def forward(self, content, strategy):
        linearContent = self.linearContent(content)
        linearContent = F.relu(self.linearContent2(linearContent))

        linearStrategy = self.linearStrategy(strategy)
        linearStrategy = F.relu(self.linearStrategy2(linearStrategy))

        out = torch.cat((linearContent, linearStrategy), axis=2)
        
        Hcontent = torch.tanh(self.content_proj(linearContent))
        Hstrategy = torch.tanh(self.strat_proj(linearStrategy))

        Wcontent = Hcontent.matmul(self.strategyContentContext)
        Wstrategy = Hstrategy.matmul(self.strategyContentContext)
        
        temp = torch.cat((Wcontent, Wstrategy), dim=2)
        temp = torch.softmax(temp, dim = 2)  
        out = torch.cat((temp[:,:,0].unsqueeze(2) * linearContent, temp[:,:,1].unsqueeze(2) * linearStrategy), axis=2)

        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out, _ = self.lstm(out)
        Hs = torch.tanh(self.s_proj(out))
        s_score = self.softmax(Hs.matmul(self.s_context_vector))
        out = out.mul(s_score)
        out = torch.sum(out, dim = 1)
        out = F.relu(self.sent_linear1(out))
        out = F.relu(self.sent_linear2(out))
        out = self.sent_linear3(out)
        out = F.sigmoid(out)

        return out, temp[:,:,0].unsqueeze(2), temp[:,:,1].unsqueeze(2), s_score