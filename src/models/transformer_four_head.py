from torch import arange, nn
import torch

# ======================================================================
# 4-HEAD VARIANT — matches Nanda et al.'s actual architecture
# (d_model=128, 4 attention heads, head dimension = 128/4 = 32 each).
#
# The original src/models/transformer.py is UNCHANGED and still uses
# single-head attention (one Query/Key/Value block covering the full
# d_model=128 at once). That file is left exactly as it is — this is a
# separate, new model class, following the same "shadow copy" pattern
# already used in this project for the LayerNorm side-experiment
# (see src/models/model_shadow_with_layernorm.py).
#
# This is the SHARED model architecture for the Nanda-Unified benchmark
# protocol (configs/nanda_unified.yaml). It is used by training, every
# predictor, and evaluation — it is not owned by any one predictor.
#
# Design: 1-layer, 4-head, same token + position embedding, MLP (4x
# expansion + ReLU), NO LayerNorm, learned positional embeddings,
# untied embed/unembed, same dropout1/dropout2 hooks (used by
# predictors/dropout.py's stress-test), same output head.
#
# Two Nanda-alignment details baked in here:
#   1. WEIGHT INIT. Every weight matrix — embeddings included — is drawn
#      from N(0, init_std) with init_std = 0.8 / sqrt(d_model) (~0.0707
#      for d_model=128). This is the TransformerLens / Nanda scheme.
#      PyTorch defaults (nn.Embedding = N(0,1), nn.Linear =
#      Kaiming-uniform) would otherwise make the token-embedding table
#      ~94% of the starting weight norm and force a large early collapse
#      under weight decay that has nothing to do with grokking.
#   2. MLP BIASES. query/key/value and the output head have NO bias
#      (Nanda: no biases in attention or unembedding), but the MLP keeps
#      b_in / b_out (Nanda keeps them). MLP biases are zero-initialised.
# ======================================================================


class TransformerFourHead(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads=4, init_std=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # TransformerLens / Nanda init scale: N(0, 0.8 / sqrt(d_model)).
        if init_std is None:
            init_std = 0.8 / (d_model ** 0.5)
        self.init_std = init_std

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(3, d_model)

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.mlp_in = nn.Linear(d_model, 4 * d_model, bias=True)
        self.mlp_activation = nn.ReLU()
        self.mlp_out = nn.Linear(4 * d_model, d_model, bias=True)

        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self._init_weights()

    def _init_weights(self):
        # Match Nanda et al. / TransformerLens: every weight matrix,
        # embeddings included, drawn from N(0, init_std). MLP biases start
        # at zero.
        for module in [self.token_embedding, self.position_embedding,
                       self.query, self.key, self.value,
                       self.mlp_in, self.mlp_out, self.output_head]:
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
        for module in [self.mlp_in, self.mlp_out]:
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def split_into_heads(self, x):
        # x: (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, head_dim)
        # This is the step that turns one big d_model-wide attention into
        # num_heads smaller, independent attention computations running
        # side by side.
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        # x: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, d_model)
        # Reverses split_into_heads — puts all heads' outputs back side by
        # side into one d_model-wide vector, exactly as Nanda et al.
        # describe combining the 4 heads' outputs before the MLP.
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x):
        token_vectors = self.token_embedding(x)
        position_vectors = self.position_embedding(arange(x.size(1), device=x.device))
        combined_vector = token_vectors + position_vectors

        query_vector = self.split_into_heads(self.query(combined_vector))
        key_vector = self.split_into_heads(self.key(combined_vector))
        value_vector = self.split_into_heads(self.value(combined_vector))

        # Scaling uses head_dim (32), not the full d_model (128) — each
        # head does its own scaled dot-product attention independently.
        scores = torch.matmul(query_vector, key_vector.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value_vector)
        attention_output = self.combine_heads(attention_output)

        attention_values = combined_vector + self.dropout1(attention_output)
        mlp_output = attention_values + self.dropout2(self.mlp_out(self.mlp_activation(self.mlp_in(attention_values))))

        logits = self.output_head(mlp_output)

        return logits


if __name__ == "__main__":
    # Nanda-Unified: p = 113 -> vocab_size = 113 + 1 = 114
    model = TransformerFourHead(vocab_size=114, d_model=128, num_heads=4)

    x = torch.eye(2, 3, dtype=torch.long)
    model.forward(x)
