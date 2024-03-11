import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from builders.vision_embedding_builder import build_vision_embedding
from utils.instance import Instance

class UniTNT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device = torch.device(config.DEVICE)

        self.vision_encoder = build_vision_embedding(config.VISION_EMBEDDING)

        # TODO:

    def forward(self, inputs: Instance):
        features = inputs.grid_features

        vision_features, vision_padding_mask = self.vision_encoder(features)

        # TODO:

class CausalSelfAttention(nn.Module):

    def __init__(self, d_k, d_model, n_heads, max_len):
        super().__init__()

        # Assume d_v = d_k
        self.d_k = d_k
        self.n_heads = n_heads

        self.key = nn.Linear(in_features=d_model, out_features=d_k * n_heads) # including bias
        self.query = nn.Linear(in_features=d_model, out_features=d_k * n_heads)
        self.value = nn.Linear(in_features=d_model, out_features=d_k * n_heads)

        # final linear layer
        self.final = nn.Linear(in_features=d_k * n_heads, out_features=d_model)

        # causal mask
        # make it so that diagonal is 0 too
        # this way we don't have to shift the inputs to make targets
        cm = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("causal_mask", cm.view(1, 1, max_len, max_len))


    def forward(self, q, k, v, pad_mask=None):
        # input: (N, T, d_model)
        q = self.query(q)   # (N, T, h * d_k), where h is the number of attention heads
        k = self.key(k)     # (N, T, h * d_k)
        v = self.value(v)   # (N, T, h * d_v)

        N = q.shape[0] # batch size
        T = q.shape[1] # sequence length

        # change the shape from (N, T, h * d_k) -> (N, T, h, d_k) -> (N, h, T, d_k) (swap second and third dim)
        q = q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2)

        # compute attention weights
        # (N, h, T, d_k) x (N, h, d_k, T) -> (N, h, T, T) (broadcasting)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        if pad_mask is not None:
            attn_scores = attn_scores.masked_fill(
                pad_mask[:, None, None, :] == 0, float('-inf')
            )
        # do causal mask
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf')
        )
        attn_weights = F.softmax(attn_scores, dim = -1)

        # compute attention-weighted values
        # (N, h, T, T) x (N, h, T, d_k) -> (N, h, T, d_k)
        A = attn_weights @ v

        # reshape it back before final linear layer
        A = A.transpose(1, 2) # (N, T, h, d_k)
        A = A.contiguous().view(N, T, self.n_heads * self.d_k) # (N, T, h * d_k)

        # projection
        return self.final(A)