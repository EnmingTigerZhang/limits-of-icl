import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# ==============================================================================
# Base Class for Attention Mechanisms
# ==============================================================================

class Attention(nn.Module):
    """
    A simple base class for attention mechanisms.
    This provides a common interface for different attention implementations.
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # A boolean flag to determine if Linear layers should have a bias.
        self.use_linear_bias = config.bias

    def forward(self, x):
        raise NotImplementedError("Each attention subclass must implement its own forward pass.")

# ==============================================================================
# Attention Implementations
# ==============================================================================

class SoftmaxCausalSelfAttention(Attention):
    """
    The standard, scaled dot-product causal self-attention mechanism
    that uses a softmax function. This is the default attention in GPT-2.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=self.use_linear_bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.use_linear_bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class LocalGlobalCausalSelfAttention(Attention):
    """
    Local + Global (hybrid sparse) causal self-attention.

    - local_window_size: int or None
        If None: full causal attention (equivalent to standard).
        If int W: each non-global token attends to [i-W+1 .. i] intersect [0..i].
    - global_attn_indices: iterable of ints or None
        Positions that are "global":
          * as keys: can be attended to by any later token (subject to causality)
          * as queries: can attend to all past positions 0..i
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        assert self.n_embd % self.n_head == 0

        # hyperparameters are now passed via kwargs
        self.local_window_size = kwargs.get("local_window_size", None)
        self.global_attn_indices = list(kwargs.get("global_attn_indices", []))

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=self.use_linear_bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.use_linear_bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # flash attention support (we will use attn_mask, so no is_causal)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # precompute local+global causal mask for max block_size
        block_size = config.block_size
        mask = torch.zeros(block_size, block_size, dtype=torch.bool)

        # first, local (or full) causal pattern
        for i in range(block_size):
            if self.local_window_size is None:
                start = 0  # full causal
            else:
                start = max(0, i - self.local_window_size + 1)
            # attend to [start..i]
            mask[i, start:i+1] = True

        # global behavior
        if len(self.global_attn_indices) > 0:
            # ensure global queries can attend to full past
            for g in self.global_attn_indices:
                if 0 <= g < block_size:
                    mask[g, :g+1] = True  # query at g attends to all 0..g
            # ensure any query can attend to global keys in the past
            for i in range(block_size):
                for g in self.global_attn_indices:
                    if 0 <= g <= i < block_size:
                        mask[i, g] = True

        # reshape to broadcast over (B, n_head, T, T)
        mask = mask.view(1, 1, block_size, block_size)
        self.register_buffer("local_global_mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # slice mask to current sequence length
        # shape: (1, 1, T, T) -> broadcast to (B, nh, T, T)
        attn_mask = self.local_global_mask[:, :, :T, :T]

        if self.flash:
            # Use PyTorch SDPA with a boolean attention mask encoding:
            # - causality
            # - local window
            # - global tokens
            # NOTE: we set is_causal=False because the mask already enforces causality.
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,  # bool mask, True = allowed
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            # manual implementation with sparse causal + local + global mask
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # attn_mask: True = allowed, False = masked
            # convert to float mask by -inf where masked
            att = att.masked_fill(~attn_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
