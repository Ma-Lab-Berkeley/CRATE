import torch
import torch.nn as nn
# from einops import rearrange
class CodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(CodingRate, self).__init__()
        self.eps = eps
    
    def forward(self, X):
        #normalize over the dim_heads dimension
        '''
        X with shape (b, h, m, d)
        W with shape (b*h, d, m)
        I with shape (m, m)
        logdet2 with shape (b*h)
        '''
        b, h, _, _ = X.shape
        # X = rearrange(X, 'b h m d -> (b h) m d')
        X = X/torch.norm(X, dim=-1, keepdim=True)
        # print((X @ X.transpose(1,2))[0])
        W = X.transpose(-1,-2)
        
        
        _,_, p, m = W.shape
        I = torch.eye(m,device=W.device)
        scalar = p / (m * self.eps)
        
        product = W.transpose(-1,-2) @ W
        logdet2 = torch.logdet(I + scalar * product)
        # print(logdet2.shape)
        mcr2s = logdet2.sum(dim=-1)/(2.)
        # print(mcr2s.shape)
        mean_mcr2 = mcr2s.mean()
        stdev = mcr2s.std()
        return (mean_mcr2, stdev)