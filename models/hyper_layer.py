import torch
import copy
from torch import nn, einsum
from einops import rearrange, repeat


class HyperLearningDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dropout = 0., emb_dropout = 0.):
        super().__init__()

        # seq_len: list[t,a,v]

        multi_decoder_layer = MultimodalTransformerDecoderLayer(dim=dim,heads=heads,mlp_dim=4*dim,dropout=dropout)
        self.multi_decoder = _get_clones(multi_decoder_layer,depth)


        self.dropout = nn.Dropout(emb_dropout)



    def forward(self, n_t, n_a, n_v):

        n_t = self.dropout(n_t)

        n_a = self.dropout(n_a)

        n_v = self.dropout(n_v)

        for dec_m in self.multi_decoder:
            n_t = dec_m(n_t, n_a, n_v)

        return n_t

class MultimodalTransformerDecoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout = 0.):
        super().__init__()

        self.attn_ca_ta = PreNormAttention(dim, Attention(dim, heads = heads, dropout = dropout))
        self.attn_ca_tv = PreNormAttention(dim, Attention(dim, heads = heads, dropout = dropout))
        self.attn_sa_t = PreNormAttention(dim, Attention(dim, heads = heads, dropout = dropout))
        self.ffn  = PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))

    def forward(self, n_t, n_a, n_v):

        n_ta = self.attn_ca_ta(n_t, n_a, n_a)
        n_tv = self.attn_ca_tv(n_t, n_v, n_v)
        n_tt = self.attn_sa_t(n_t,n_t,n_t)
        #sum
        n_t = n_tt + n_tv + n_ta
        n_t = self.ffn(n_t) + n_t
        return n_t


class PreNormForward(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, mask=None):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v,mask)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        assert dim % heads == 0, "Error: 'dim' must be divisible by 'heads'."
        dim_head = int(dim/heads)
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim_head * heads, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v, mask=None):
        b, n, _= q.shape
        h = self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None: #padding mask
            padding_mask = mask.unsqueeze(1).expand(b, q.size(2), k.size(2)).unsqueeze(1).repeat(1, h, 1, 1)
            dots = dots.masked_fill(padding_mask, -1e9) # -np.inf
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
