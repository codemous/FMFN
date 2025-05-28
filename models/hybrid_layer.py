import torch
import copy
from torch import nn, einsum
from einops import rearrange, repeat


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

    def forward(self, q, k, v):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)


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

    def forward(self, q, k, v):
        b, n, _= q.shape
        h = self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class HybridLearningEncoder(nn.Module):
    def __init__(self, seq_len, neck_size, dim, depth, heads, dropout = 0., emb_dropout = 0.):
        super().__init__()

        # seq_len: list[t,a,v]

        text_encoder_layer = TextTransformerEncoderLayer(dim=dim,heads=heads,mlp_dim=4*dim,dropout=dropout)
        self.text_encoder = _get_clones(text_encoder_layer,depth)


        audio_encoder_layer =  AudioTransformerEncoderLayer(dim=dim,heads=heads,mlp_dim=4*dim,dropout=dropout)
        self.audio_encoder = _get_clones(audio_encoder_layer,depth)


        visual_encoder_layer =  VisualTransformerEncoderLayer(dim=dim,heads=heads,mlp_dim=4*dim,dropout=dropout)
        self.visual_encoder = _get_clones(visual_encoder_layer,depth)


        self.pos_embedding_t = nn.Parameter(torch.randn(1, seq_len[0], dim))
        self.pos_embedding_neck_t = nn.Parameter(torch.randn(1, neck_size, dim))
        self.neck_text = nn.Parameter(torch.zeros(1, neck_size, dim))

        self.pos_embedding_a = nn.Parameter(torch.randn(1, seq_len[1], dim))
        self.pos_embedding_neck_a = nn.Parameter(torch.randn(1, neck_size, dim))
        self.neck_audio = nn.Parameter(torch.zeros(1, neck_size, dim))

        self.pos_embedding_v = nn.Parameter(torch.randn(1, seq_len[2], dim))
        self.pos_embedding_neck_v = nn.Parameter(torch.randn(1, neck_size, dim))
        self.neck_visual = nn.Parameter(torch.zeros(1, neck_size, dim))


        self.dropout = nn.Dropout(emb_dropout)

        # self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        #
        # self.pool = pool
        # self.to_latent = nn.Identity()

    def forward(self, h_t, h_a, h_v, mask_t, mask_a, mask_v):

        neck_text = repeat(self.neck_text, '1 n d -> b n d', b = h_t.size(0))
        h_t = h_t + self.pos_embedding_t
        neck_text = neck_text + self.pos_embedding_neck_t

        neck_audio = repeat(self.neck_audio, '1 n d -> b n d', b = h_a.size(0))
        h_a = h_a + self.pos_embedding_a
        neck_audio = neck_audio + self.pos_embedding_neck_a

        neck_visual = repeat(self.neck_visual, '1 n d -> b n d', b = h_v.size(0))
        h_v = h_v + self.pos_embedding_v
        neck_visual = neck_visual + self.pos_embedding_neck_v

        h_t = self.dropout(h_t)
        neck_text = self.dropout(neck_text)

        h_a = self.dropout(h_a)
        neck_audio = self.dropout(neck_audio)

        h_v = self.dropout(h_v)
        neck_visual = self.dropout(neck_visual)

        for enc_t, enc_a, enc_v in zip(self.text_encoder,self.audio_encoder,self.visual_encoder):
            neck_audio = enc_a(neck_audio, h_a, neck_text)
            neck_visual = enc_v(neck_visual, h_v, neck_text)
            neck_text = enc_t(neck_text, h_t)
        return neck_text,neck_audio,neck_visual

class TextTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout = 0.):
        super().__init__()


        self.attn_ca = PreNormAttention(dim, Attention(dim, heads = heads, dropout = dropout))
        self.attn_sa = PreNormAttention(dim, Attention(dim, heads = heads, dropout = dropout))
        self.ffn  = PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))


    def forward(self, n_t, x_t):

        n_t = self.attn_ca(n_t, x_t, x_t) + n_t
        n_t = self.attn_sa(n_t, n_t, n_t) + n_t
        n_t = self.ffn(n_t) + n_t

        return n_t


class VisualTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout = 0.):
        super().__init__()


        self.attn_ca = PreNormAttention(dim, Attention(dim, heads = heads, dropout = dropout))
        self.attn_sa = PreNormAttention(dim, Attention(dim, heads = heads, dropout = dropout))
        self.ffn  = PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))


    def forward(self, n_v, x_v, n_t):

        n_v = self.attn_ca(n_v, x_v, x_v) + n_v
        n_v = self.attn_sa(n_v + n_t, n_v, n_v) + n_v
        n_v = self.ffn(n_v) + n_v

        return n_v


class AudioTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout = 0.):
        super().__init__()


        self.attn_ca = PreNormAttention(dim, Attention(dim, heads = heads, dropout = dropout))
        self.attn_sa = PreNormAttention(dim, Attention(dim, heads = heads, dropout = dropout))
        self.ffn  = PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))


    def forward(self, n_a, x_a, n_t):

        n_a = self.attn_ca(n_a, x_a, x_a) + n_a
        n_a = self.attn_sa(n_a + n_t, n_a, n_a) + n_a
        n_a = self.ffn(n_a) + n_a

        return n_a

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

