import torch
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from torch import nn


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 5.0

    def forward(self, x):
        # A@x
        output = F.linear(x, self.weight, bias=None)
        return output


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., use_softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1) if use_softmax else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale  # (b h n n)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Block_CRATE(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0., ista=0.1, use_softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.dim = dim
        self.layers.append(
            nn.ModuleList(
                [
                    PreNorm(dim, FeedForward(dim, dim, dropout=dropout, step_size=ista)),
                    PreNorm(
                        dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, use_softmax=use_softmax)
                    ),
                ]
            )
        )

    def forward(self, x):
        for ff, attn in self.layers:
            grad_x = ff(x)
            x = grad_x - attn(grad_x)
        return x
