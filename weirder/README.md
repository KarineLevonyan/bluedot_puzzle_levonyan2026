# Weirder representations

The BlueDot puzzle hides **`country`** at hidden layer **`h2`** as an **XOR with `food`**:
a single axis carries the country signal with its sign flipped by the food label, so a linear
probe at `h2` sees nothing (~0.43) yet **2-feature slicing on `food`** fully unfolds it (~0.96).
(See `../bluedot-tais-puzzle/report.html`, Tasks 1–2, for the full characterization.)

This folder trains two MLP heads that hide `country` in deliberately **weirder** ways than the
puzzle — so that the puzzle's recovery trick (slice on one partner feature) no longer works —
and runs diagnostics comparing them against the puzzle model and a normally-trained one.

- **Model C — conditional alignment by sentiment.** At `h2`, the `country` direction is aligned
  with `food` when `sentiment=0` and with `person` when `sentiment=1`. This is a genuine **3-way
  interaction**: slicing on any single feature fails to recover `country`; only the
  `(sentiment × food)` or `(sentiment × person)` **double-slice** does.
- **Model X — cross-layer rotating alignment.** `country` is aligned with a different feature at
  each layer (`h1 ↔ food`, `h2 ↔ sentiment`, `h3 ↔ person`), so there is no clean `country`
  direction at any single hidden layer and the **inter-layer velocity** field (`v = h_{k+1} − h_k`)
  shows multiple divergence shocks across the trajectory.

## Architecture

Same 5-layer head as the puzzle, on top of frozen `all-MiniLM-L6-v2` (mean-pooled, 384-d):

```
384 → 64 → ReLU → 64 → ReLU → 64 → ReLU → 64 → ReLU → 8 (per-feature sigmoid)
            h0          h1          h2          h3        logits
```

The 8 binary features are `number, question, color, food, sentiment, country, person, body_part`.

## Run

```bash
python weirder.py
```

This trains a **normal** head, **Model C**, and **Model X**, loads the **puzzle** model
(`model.pt`), and for all four prints:

- test accuracy per feature;
- `country` linear-probe accuracy across layers (`embed, h0, h1, h2, h3`);
- inter-layer velocity divergence `‖v⁺ − v⁻‖` per transition (`country`);
- `country`-direction rotation `cos(angle)` between consecutive layers;
- 2-feature slicing accuracy for `country` at `h2`;
- 3-feature double-slicing for `country` at `h2`, conditioned on `(sentiment × food)` and
  `(sentiment × person)`.

It also writes five figures (`weirder_*.png`) and saves the two trained heads.

## Files

| file | role |
|------|------|
| `weirder.py` | training + diagnostics (entry point) |
| `model.pt` | the original puzzle model (used as the reference) |
| `feature_names.json` | label order for the 8 features |
| `embeds_train.npy`, `embeds_test.npy` | frozen MiniLM sentence embeddings (384-d) |
| `data/train.jsonl`, `data/test.jsonl` | texts + multi-label targets (7000 / 1500) |

Running the script writes into this folder:

| output | contents |
|--------|----------|
| `model_conditional.pt` | trained Model C |
| `model_crosslayer.pt` | trained Model X |
| `weirder_probe_heatmap.png` | per-feature × per-layer linear-probe accuracy (4 models) |
| `weirder_riemann.png` | signal magnitude, `country` linear separability, velocity divergence, direction rotation |
| `weirder_tsne.png` | t-SNE of `h2` colored by `country / food / sentiment / person` (3 models) |
| `weirder_3way.png` | `h2` PCA of `country` positives separated by `sentiment` (the 3-way interaction) |
| `weirder_slice_heatmap.png` | 2-feature slice-probe accuracy for `country` at `h2` |
