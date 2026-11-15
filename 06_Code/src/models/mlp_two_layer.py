"""Two-layer MLP for the isolated Salah & Yevick Dropout setup.

Paper architecture (arXiv:2507.11645, Section 2):
  - Embed each of i, j to a 128-d vector (shared embedding table).
  - Concatenate → 256-d input.
  - Hidden layer: Linear(256, 256) + ReLU + Dropout.
  - Output layer: Linear(256, p) with p = 53.
  - Xavier Normal init: W ~ N(0, sqrt(2 / (n_in + n_out))).

Dropout rate is adjustable at measurement time via set_dropout_rate.
Training itself uses dropout_rate=0.0 (paper applies dropout at eval).
This file is not the shared Nanda Transformer.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class MLPTwoLayer(nn.Module):
    def __init__(
        self,
        modulus=53,
        embed_dim=128,
        hidden_dim=256,
        dropout_rate=0.0,
        init_scale=1.0,
    ):
        super().__init__()
        self.modulus = modulus
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(modulus, embed_dim)
        # After concat of two embeddings: 2 * embed_dim == hidden_dim.
        assert 2 * embed_dim == hidden_dim, (
            "Salah & Yevick concat two 128-d embeddings into a 256-d MLP input"
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, modulus)

        self._init_xavier_normal(init_scale)

    def _init_xavier_normal(self, init_scale):
        """Xavier Normal as in the paper: sigma = sqrt(2 / (n_in + n_out))."""
        for module in (self.embedding, self.fc1, self.fc2):
            if isinstance(module, nn.Embedding):
                n_in, n_out = module.num_embeddings, module.embedding_dim
                # Embedding: treat as (vocab → dim); paper uses Xavier on embeddings.
                std = init_scale * math.sqrt(2.0 / (n_in + n_out))
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Linear):
                n_in, n_out = module.in_features, module.out_features
                std = init_scale * math.sqrt(2.0 / (n_in + n_out))
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_dropout_rate(self, rate):
        """Change the measurement-time dropout probability."""
        self.dropout.p = float(rate)

    def forward(self, x):
        """x: LongTensor of shape (batch, 2) holding (i, j)."""
        i_emb = self.embedding(x[:, 0])
        j_emb = self.embedding(x[:, 1])
        h = torch.cat([i_emb, j_emb], dim=-1)
        h = self.fc1(h)
        h = self.activation(h)
        h = self.dropout(h)
        return self.fc2(h)


def cross_entropy_loss(logits, labels):
    """Standard cross-entropy for the modular-addition classification head."""
    return F.cross_entropy(logits, labels)
