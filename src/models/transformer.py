import pandas as pd

from torch import arange, nn
import torch
torch.set_printoptions(profile="full")


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(3, d_model)
        
    def forward(self, x):
        token_vectors = self.token_embedding(x)
        position_vectors = self.position_embedding(arange(x.size(1)))
        return token_vectors + position_vectors
        
        
        
        
if __name__ == "__main__":
    model = Transformer(vocab_size=97, d_model=128)