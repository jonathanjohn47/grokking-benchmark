# archive/p97_default/ — pre-Nanda-Unified four-head runs

These are the 3 seeded four-head runs from the OLD default substrate,
kept for the record. They were produced BEFORE the September 4, 2026
switch to the Nanda-Unified protocol (see context.md).

Old substrate (what produced these runs):
  - prime p = 97
  - weight init = PyTorch defaults (nn.Embedding = N(0,1), nn.Linear = Kaiming-uniform)
  - AdamW betas = (0.9, 0.999)  [PyTorch default]
  - MLP biases = False
  - lr = 1e-3, weight_decay = 1.0, full-batch, 40000 epochs, 30/70 split, no LayerNorm

Known artefacts in these runs (why the substrate was changed):
  - ~94% of the initial weight norm is the token-embedding table; weight
    decay crushes it in the first ~2000 epochs (init transient, not grokking).
  - beta2 = 0.999 produces a slingshot-type weight-norm spike at the grok
    transition on the long flat plateau.

The live experiment (src/) now trains on configs/nanda_unified.yaml.
This folder is gitignored (kept on local disk only), same as runs/.
