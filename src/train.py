from torch import nn

from data.modular_arithmetic import get_dataloaders
from models.transformer import Transformer

data_loader = get_dataloaders(97, batch_size=32)
model = Transformer(vocab_size=98, d_model=128)

x, y = next(iter(data_loader[0]))
logit = model.forward(x)
equal_sign_logit = logit[:, 2, :]

print(equal_sign_logit.shape)

cross_entropy_loss = nn.CrossEntropyLoss()
loss = cross_entropy_loss(equal_sign_logit, y)

print(loss)


