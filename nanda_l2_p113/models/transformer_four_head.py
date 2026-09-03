from torch import arange, nn
import torch

# ======================================================================
# 4-HEAD VARIANT — matches Nanda et al.'s actual architecture
# (d_model=128, 4 attention heads, head dimension = 128/4 = 32 each).
#
# This is a byte-for-byte copy of src/models/transformer_four_head.py.
# It lives here so the nanda_l2_p113/ experiment is fully self-contained
# and does not import anything from src/. The ONLY change from the src/
# copy is the vocab_size number in the __main__ demo (98 -> 114), because
# this experiment runs on p = 113 (113 number tokens 0..112 + one "="
# token with id 113).
#
# Architecture is otherwise identical to the main experiment: same token +
# position embedding, same MLP (4x expansion + ReLU), no LayerNorm, no
# Linear biases, same dropout1/dropout2 hooks (kept at p=0.0 here — this
# experiment has no Dropout predictor, but the hooks are harmless and
# keeping them keeps the class identical to src/), same output head.
# ======================================================================


class TransformerFourHead(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(3, d_model)

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.mlp_in = nn.Linear(d_model, 4 * d_model, bias=False)
        self.mlp_activation = nn.ReLU()
        self.mlp_out = nn.Linear(4 * d_model, d_model, bias=False)

        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

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
    # p = 113 -> vocab_size = 113 + 1 = 114
    model = TransformerFourHead(vocab_size=114, d_model=128, num_heads=4)

    x = torch.eye(2, 3, dtype=torch.long)
    model.forward(x)
