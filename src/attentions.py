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

class MQACausalSelfAttention(Attention):
    """
    Multi-Query Attention (MQA):
    - each head has its own Query projection (Q)
    - all heads share Key and Value projections (K, V)
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        assert self.n_embd % self.n_head == 0

        head_dim = config.n_embd // config.n_head

        # Q: separate for each head → output = (B, T, n_head * head_dim)
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.use_linear_bias)

        # K,V: shared across heads → output = (B, T, head_dim)
        self.k_proj = nn.Linear(self.n_embd, head_dim, bias=self.use_linear_bias)
        self.v_proj = nn.Linear(self.n_embd, head_dim, bias=self.use_linear_bias)

        # final projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.use_linear_bias)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            # causal mask for fallback path
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.size()
        head_dim = C // self.n_head

        # ---- Compute projections ----
        q = self.q_proj(x)                     # (B, T, C)
        k = self.k_proj(x)                     # (B, T, head_dim)
        v = self.v_proj(x)                     # (B, T, head_dim)

        # reshape Q: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # reshape K,V shared: (B, 1, T, head_dim)
        k = k.unsqueeze(1)   # insert head dim, but always 1
        v = v.unsqueeze(1)

        # ---- Flash Attention path ----
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # ---- Manual attention ----
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, head_dim)

        # ---- Combine heads ----
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # final projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FAVORCausalSelfAttention(Attention):
    """
    Performer-style FAVOR+ causal self-attention using positive random features
    to approximate the scaled softmax kernel in (roughly) linear time.

    - rf_dim (int, optional via kwargs):
        Number of random features m for the feature map phi.
        Default: 2 * head_dim.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        assert self.n_embd % self.n_head == 0
        self.head_dim = self.n_embd // self.n_head

        # number of random features for phi
        self.rf_dim = int(kwargs.get("rf_dim", 2 * self.head_dim))
        assert self.rf_dim > 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=self.use_linear_bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.use_linear_bias)

        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Positive random feature weights for FAVOR+ (shared across heads)
        # omega ~ N(0, I), saved in state_dict
        w = torch.randn(self.head_dim, self.rf_dim)
        self.register_buffer("omega", w, persistent=True)

        # small epsilon to avoid division by zero
        self.eps = 1e-6

    def _phi(self, x):
        """
        Positive random feature map phi for the softmax kernel.

        x: (B, nh, T, head_dim)
        returns: (B, nh, T, rf_dim)

        Conceptually:
          phi(x)_j = exp(omega_j^T x - ||x||^2 / 2) / sqrt(m)
        """
        B, nh, T, d = x.shape
        m = self.rf_dim

        # (B*nh*T, d)
        x_flat = x.reshape(B * nh * T, d)

        # projection: (B*nh*T, m)
        proj = x_flat @ self.omega  # omega: (d, m)

        # ||x||^2 / 2 term: (B*nh*T, 1)
        norm_sq_half = 0.5 * (x_flat * x_flat).sum(dim=-1, keepdim=True)

        # positive random features: (B*nh*T, m)
        phi_flat = torch.exp(proj - norm_sq_half) / math.sqrt(m)

        # reshape back: (B, nh, T, m)
        phi = phi_flat.view(B, nh, T, m)
        return phi

    def forward(self, x):
        """
        x: (B, T, C = n_embd)
        returns: (B, T, C)
        """
        B, T, C = x.size()
        assert C == self.n_embd

        # project to q, k, v: (B, T, 3C) -> (B, T, C) each
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # (B, T, nh, head_dim) -> (B, nh, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scale q, k for "scaled" dot product
        scale = 1.0 / math.sqrt(self.head_dim)
        q = q * scale
        k = k * scale

        # (optional) dropout on values
        v = self.attn_dropout(v)

        # random feature maps: (B, nh, T, m)
        phi_q = self._phi(q)
        phi_k = self._phi(k)

        B, nh, T, d_h = v.shape
        m = self.rf_dim

        # ---- Causal FAVOR via vectorized prefix sums ----
        # S_t = sum_{j<=t} phi_k_j ⊗ v_j  in R^{m x d_h}
        # Z_t = sum_{j<=t} phi_k_j        in R^{m}
        #
        # kv = phi_k ⊗ v -> (B, nh, T, m, d_h)
        kv = phi_k.unsqueeze(-1) * v.unsqueeze(-2)   # (B, nh, T, m, d_h)

        # flatten last two dims, cumsum over time, then reshape back
        kv_flat = kv.view(B, nh, T, m * d_h)        # (B, nh, T, m*d_h)
        S_flat = torch.cumsum(kv_flat, dim=2)       # (B, nh, T, m*d_h)
        S = S_flat.view(B, nh, T, m, d_h)           # (B, nh, T, m, d_h)

        # normalizer prefix sums: (B, nh, T, m)
        Z = torch.cumsum(phi_k, dim=2)

        # num_t = phi_q_t^T S_t  -> (B, nh, T, d_h)
        # den_t = phi_q_t^T Z_t  -> (B, nh, T)
        #
        # Explicit contractions over "m" using einsum:
        num = torch.einsum("bhtm,bhtmd->bhtd", phi_q, S)   # (B, nh, T, d_h)
        den = torch.einsum("bhtm,bhtm->bht", phi_q, Z)     # (B, nh, T)
        den = den.unsqueeze(-1) + self.eps                 # (B, nh, T, 1)

        y = num / den                                      # (B, nh, T, d_h)

        # (B, nh, T, d_h) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection + residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y