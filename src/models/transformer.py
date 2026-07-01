import torch.nn as nn


class Transformer(nn.Module):
    
    def __init__(self, num_tokens, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, d_model)
        self.position_embedding = nn.Embedding(2, d_model)