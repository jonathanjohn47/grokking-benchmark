import pandas as pd

from torch import arange, nn
import torch
torch.set_printoptions(profile="full")


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
        position_vectors = self.position_embedding(arange(x.size(1)))
        combined_vector = token_vectors + position_vectors
        query_vector = self.query(combined_vector)
        key_vector = self.key(combined_vector)
        value_vector = self.value(combined_vector)
        
        scores = torch.matmul(query_vector, key_vector.transpose(-2, -1)) / torch.sqrt(torch.tensor(query_vector.size(-1), dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value_vector)
        mlp_output = self.mlp(attended_values)
        
        logits = self.output_head(mlp_output)
        
        print("Logits shape:", logits.shape)
        
        return logits
        
        
if __name__ == "__main__":
    model = Transformer(vocab_size=98, d_model=128)
    
    x = torch.eye(2, 3, dtype=torch.long)
    print(x)
    model.forward(x)
    