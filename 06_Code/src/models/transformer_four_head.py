from torch import nn
import torch
import torch.nn.functional as F

# ======================================================================
# Shared model for the Nanda-Unified benchmark (configs/nanda_unified.yaml).
# Used by training, every predictor, and evaluation. Not owned by any one
# predictor.
#
# Training setup (retrain, Nov 2026), matched to Nanda's released
# transformers.py (progress-measures-paper) for p = 113, d_model = 128,
# 4 heads, d_mlp = 512, d_vocab = 114, n_ctx = 3, ReLU, no LayerNorm:
#   - W_O: attention output projection, 128 x 128, no bias, after the
#     heads are concatenated and before the residual add.
#   - Causal mask: lower-triangular mask on the 3 x 3 scores, upper
#     triangle set to -1e10, then scale by 1/sqrt(d_head), then softmax.
#     Same formula as Nanda's Attention.forward (tril, then -1e10).
#   - Init: W_E, W_pos, W_Q, W_K, W_V, W_O, W_in, W_out drawn as
#     randn / sqrt(d_model). W_U drawn as randn / sqrt(d_vocab).
#     b_in and b_out are zeros. No attention, embedding, or unembed bias.
#   - No dropout modules. Nanda's model has none. The Dropout predictor
#     applies its own measurement-time dropout; it does not live here.
# ======================================================================


def cross_entropy_high_precision(logits, labels):
    """Final-position cross entropy in float64, same reduction as
    helpers.cross_entropy_high_precision: log_softmax, gather the label,
    minus the mean.

    MPS has no float64, so the logits are cast to CPU float64 for the
    loss. The cast is differentiable, so gradients flow back to the model.
    """
    logits64 = logits.to(dtype=torch.float64, device="cpu")
    labels_cpu = labels.to(device="cpu")
    logprobs = F.log_softmax(logits64, dim=-1)
    chosen = torch.gather(logprobs, index=labels_cpu[:, None], dim=-1)
    return -torch.mean(chosen)


class TransformerFourHead(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads=4, n_ctx=3):
        super().__init__()
        # nn.Linear / nn.Embedding draw their own default init. Save the RNG
        # and restore it before the Nanda draws, so those defaults do not
        # move the stream. The caller seeds torch before constructing.
        rng_state = torch.get_rng_state()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_ctx, d_model)

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        # W_O. nn.Linear weight is (out, in) = (d_model, d_model), which is
        # Nanda's W_O shape (d_model, d_head * num_heads).
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        self.mlp_in = nn.Linear(d_model, 4 * d_model, bias=True)
        self.mlp_activation = nn.ReLU()
        self.mlp_out = nn.Linear(4 * d_model, d_model, bias=True)

        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)))
        self._init_weights(rng_state)

    def _init_weights(self, rng_state):
        """Same draws as Nanda's constructors, in the same order.

        W_E and W_U are drawn as (d_model, d_vocab) and then transposed
        into nn.Embedding / nn.Linear, which store the other axis first.
        W_K, W_Q, W_V are drawn as (heads, d_head, d_model) and reshaped
        into the Linear weight (d_model, d_model).
        """
        torch.set_rng_state(rng_state)
        scale_model = self.d_model ** -0.5
        scale_vocab = self.vocab_size ** -0.5
        with torch.no_grad():
            w_e = torch.randn(self.d_model, self.vocab_size) * scale_model
            self.token_embedding.weight.copy_(w_e.T)
            w_pos = torch.randn(self.position_embedding.num_embeddings, self.d_model) * scale_model
            self.position_embedding.weight.copy_(w_pos)
            for linear in (self.key, self.query, self.value):
                drawn = torch.randn(self.num_heads, self.head_dim, self.d_model) * scale_model
                linear.weight.copy_(drawn.reshape(self.d_model, self.d_model))
            w_o = torch.randn(self.d_model, self.d_model) * scale_model
            self.attn_out.weight.copy_(w_o)
            w_in = torch.randn(self.mlp_in.out_features, self.d_model) * scale_model
            self.mlp_in.weight.copy_(w_in)
            self.mlp_in.bias.zero_()
            w_out = torch.randn(self.d_model, self.mlp_out.in_features) * scale_model
            self.mlp_out.weight.copy_(w_out)
            self.mlp_out.bias.zero_()
            w_u = torch.randn(self.d_model, self.vocab_size) * scale_vocab
            self.output_head.weight.copy_(w_u.T)

    def split_into_heads(self, x):
        batch_size, seq_len, _d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size, _num_heads, seq_len, _head_dim = x.shape
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

    def forward(self, x):
        token_vectors = self.token_embedding(x)
        positions = torch.arange(x.size(1), device=x.device)
        combined_vector = token_vectors + self.position_embedding(positions)

        query_vector = self.split_into_heads(self.query(combined_vector))
        key_vector = self.split_into_heads(self.key(combined_vector))
        value_vector = self.split_into_heads(self.value(combined_vector))

        # Scores are unscaled here. Nanda divides by sqrt(d_head) after the mask.
        scores = torch.matmul(query_vector, key_vector.transpose(-2, -1))
        seq = scores.size(-1)
        causal = self.mask[:seq, :seq]
        scores = torch.tril(scores) - 1e10 * (1.0 - causal)
        attention_weights = torch.softmax(scores / (self.head_dim ** 0.5), dim=-1)
        mixed = torch.matmul(attention_weights, value_vector)
        z_flat = self.combine_heads(mixed)
        attention_output = self.attn_out(z_flat)

        attention_values = combined_vector + attention_output
        mlp_hidden = self.mlp_activation(self.mlp_in(attention_values))
        mlp_output = self.mlp_out(mlp_hidden)
        residual = attention_values + mlp_output
        return self.output_head(residual)


if __name__ == "__main__":
    model = TransformerFourHead(vocab_size=114, d_model=128, num_heads=4)
    x = torch.eye(2, 3, dtype=torch.long)
    model.forward(x)
