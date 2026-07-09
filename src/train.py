from torch import nn
from torch.optim import Adam

from data.modular_arithmetic import get_dataloaders
from models.transformer import Transformer

data_loader = get_dataloaders(97, batch_size=32)
model = Transformer(vocab_size=98, d_model=128)
optimizer = Adam(model.parameters(), lr=0.001)

print("Optimizer:", optimizer)
cross_entropy_loss = nn.CrossEntropyLoss()

for epoch in range(100):
    for x, y in data_loader[0]:
        logit = model.forward(x)
        equal_sign_logit = logit[:, 2, :]

        loss = cross_entropy_loss(equal_sign_logit, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
