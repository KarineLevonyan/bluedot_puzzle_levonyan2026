# Weirder representations

Two engineered MLP heads (on top of frozen MiniLM embeddings) that hide
`country` in deliberately "weirder" ways than the original puzzle, plus
diagnostics comparing them against the puzzle model and a normally-trained one.

- **Model C — conditional alignment by sentiment.** At `h2`, `country` is
  aligned with `food` when `sentiment=0` and with `person` when `sentiment=1`.
  This is a genuine 3-way interaction: slicing on a single feature fails to
  recover `country`; only `(sentiment × food)` or `(sentiment × person)`
  double-slicing does.
- **Model X — cross-layer rotating alignment.** `country` is aligned with a
  different feature at each layer (`h1↔food`, `h2↔sentiment`, `h3↔person`), so
  there is no clean `country` direction at any single hidden layer and the
  velocity field shows multiple shocks across the trajectory.

## Run

```bash
python weirder.py
```

Trains the normal, Model C, and Model X heads, loads the puzzle model, prints
the diagnostic tables (per-layer linear probes, velocity divergence, direction
rotation, single- and double-feature slicing), and writes a few figures.

## Files

| file | role |
|------|------|
| `weirder.py` | training + diagnostics (entry point) |
| `model.pt` | the original puzzle model (used as the reference) |
| `feature_names.json` | label order for the 8 features |
| `embeds_train.npy`, `embeds_test.npy` | frozen MiniLM sentence embeddings (384-d) |
| `data/train.jsonl`, `data/test.jsonl` | texts + multi-label targets |

Running the script writes `model_conditional.pt`, `model_crosslayer.pt`, and
the `weirder_*.png` figures into this folder.
