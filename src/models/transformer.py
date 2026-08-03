from torch import arange, nn
import torch


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(3, d_model)

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.mlp_in = nn.Linear(d_model, 4 * d_model, bias=False)
        self.mlp_activation = nn.ReLU()
        self.mlp_out = nn.Linear(4 * d_model, d_model, bias=False)


        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        token_vectors = self.token_embedding(x)
        position_vectors = self.position_embedding(arange(x.size(1), device=x.device))
        combined_vector = token_vectors + position_vectors
        query_vector = self.query(combined_vector)
        key_vector = self.key(combined_vector)
        value_vector = self.value(combined_vector)

        scores = torch.matmul(query_vector, key_vector.transpose(-2, -1)) / (query_vector.size(-1) ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = combined_vector + torch.matmul(attention_weights, value_vector)
        mlp_output = attended_values + self.mlp_out(self.mlp_activation(self.mlp_in(attended_values)))


        logits = self.output_head(mlp_output)

        return logits
        
        
if __name__ == "__main__":
    model = Transformer(vocab_size=98, d_model=128)
    
    x = torch.eye(2, 3, dtype=torch.long)
    model.forward(x)
    