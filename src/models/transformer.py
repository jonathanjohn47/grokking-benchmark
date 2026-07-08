import pandas as pd

from torch import nn
import torch
torch.set_printoptions(profile="full")


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        print(self.token_embedding.weight)
        
if __name__ == "__main__":
    model = Transformer(vocab_size=97, d_model=128)
    
    dataframe = pd.DataFrame(model.token_embedding.weight.detach().numpy())
    print(dataframe)