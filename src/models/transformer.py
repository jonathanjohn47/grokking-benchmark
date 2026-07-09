import pandas as pd

from torch import arange, nn
import torch
torch.set_printoptions(threshold=20, edgeitems=3)


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(3, d_model)
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.mlp = nn.Linear(d_model, d_model)
        
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        token_vectors = self.token_embedding(x)
        print("token_vectors:", token_vectors.shape, "\n", token_vectors)
        position_vectors = self.position_embedding(arange(x.size(1)))
        print("position_vectors:", position_vectors.shape, "\n", position_vectors)
        combined_vector = token_vectors + position_vectors
        print("combined_vector:", combined_vector.shape, "\n", combined_vector)
        query_vector = self.query(combined_vector)
        print("query_vector:", query_vector.shape, "\n", query_vector)
        key_vector = self.key(combined_vector)
        print("key_vector:", key_vector.shape, "\n", key_vector)
        value_vector = self.value(combined_vector)
        print("value_vector:", value_vector.shape, "\n", value_vector)

        scores = torch.matmul(query_vector, key_vector.transpose(-2, -1)) / torch.sqrt(torch.tensor(query_vector.size(-1), dtype=torch.float32))
        print("scores:", scores.shape, "\n", scores)
        attention_weights = torch.softmax(scores, dim=-1)
        print("attention_weights:", attention_weights.shape, "\n", attention_weights)
        attended_values = torch.matmul(attention_weights, value_vector)
        print("attended_values:", attended_values.shape, "\n", attended_values)
        mlp_output = self.mlp(attended_values)
        print("mlp_output:", mlp_output.shape, "\n", mlp_output)

        logits = self.output_head(mlp_output)
        print("logits:", logits.shape, "\n", logits)
        
        return logits
        
        
if __name__ == "__main__":
    model = Transformer(vocab_size=98, d_model=128)
    
    x = torch.eye(2, 3, dtype=torch.long)
    model.forward(x)
    