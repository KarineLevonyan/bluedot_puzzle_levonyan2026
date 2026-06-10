"""Two 'weirder' engineering variants:

  Model C — Conditional alignment by sentiment (extension #2)
            sentiment=0 batch: align d_country ↔ d_food at h2
            sentiment=1 batch: align d_country ↔ d_person at h2
            ⇒ 3-way interaction. 2-feature slicing should fail; only
              (sentiment × food) or (sentiment × person) recovers country.

  Model X — Cross-layer rotating alignment (extension #3)
            h1: align d_country ↔ d_food
            h2: align d_country ↔ d_sentiment
            h3: align d_country ↔ d_person
            ⇒ NO clean country direction at any hidden layer; multiple
              velocity-divergence shocks across the trajectory.

Diagnostics computed for puzzle, normal, model-C, model-X:
  - per-layer linear probe accuracy heatmap
  - per-layer velocity divergence
  - direction rotation across consecutive layers
  - 2-feature slicing accuracies for country
  - 3-feature slicing accuracies (conditioned on pairs of other features)
  - t-SNE of h2 colored by country, sentiment, partner features
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)

# ============================================================
# Data + base utilities
# ============================================================
def load_jsonl(p):
    t, l = [], []
    for line in open(p):
        d = json.loads(line); t.append(d["text"]); l.append(d["labels"])
    return t, np.array(l)

feature_names = json.load(open("feature_names.json"))
COUNTRY   = feature_names.index("country")
FOOD      = feature_names.index("food")
SENTIMENT = feature_names.index("sentiment")
PERSON    = feature_names.index("person")
COLOR     = feature_names.index("color")

train_texts, train_y = load_jsonl("data/train.jsonl")
test_texts,  test_y  = load_jsonl("data/test.jsonl")
emb_tr = np.load("embeds_train.npy").astype(np.float32)
emb_te = np.load("embeds_test.npy" ).astype(np.float32)
X_tr = torch.from_numpy(emb_tr); X_te = torch.from_numpy(emb_te)
Y_tr = torch.from_numpy(train_y.astype(np.float32))
Y_te = torch.from_numpy(test_y .astype(np.float32))

# ============================================================
# Model
# ============================================================
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),    # 0, 1  → h0
            nn.Linear(64, 64),  nn.ReLU(),    # 2, 3  → h1
            nn.Linear(64, 64),  nn.ReLU(),    # 4, 5  → h2
            nn.Linear(64, 64),  nn.ReLU(),    # 6, 7  → h3
            nn.Linear(64, 8),
        )
    def forward(self, x): return self.layers(x)
    def h_at(self, x, k):
        """Return post-ReLU output of layer k where k ∈ {0,1,2,3}."""
        return self.layers[:2*(k+1)](x)

# ============================================================
# Alignment loss helpers
# ============================================================
def class_mean_diff(h, labels, idx):
    pos = labels[:, idx] > 0.5
    neg = ~pos
    if pos.sum() < 2 or neg.sum() < 2: return None
    return h[pos].mean(0) - h[neg].mean(0)

def align_loss(h, y, target_idx, partner_idx):
    d_t = class_mean_diff(h, y, target_idx)
    d_p = class_mean_diff(h, y, partner_idx)
    if d_t is None or d_p is None: return torch.tensor(0.0, device=h.device)
    return ((d_t - d_p) ** 2).sum()

# ============================================================
# Train: normal
# ============================================================
def train_normal(seed=0, epochs=50):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Head(); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    BATCH = 256; N = len(X_tr)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            loss = F.binary_cross_entropy_with_logits(m(X_tr[idx]), Y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m

# ============================================================
# Train: Model C — conditional alignment
# ============================================================
def train_conditional(seed=0, epochs=80, lam=2.0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Head(); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    BATCH = 256; N = len(X_tr)
    print(f"Training Model C (conditional alignment by sentiment, λ={lam})")
    for ep in range(epochs):
        m.train(); perm = torch.randperm(N)
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            x, y = X_tr[idx], Y_tr[idx]
            logits = m(x)
            loss_bce = F.binary_cross_entropy_with_logits(logits, y)
            h2 = m.h_at(x, 2)
            sent0 = y[:, SENTIMENT] <= 0.5
            sent1 = y[:, SENTIMENT] >  0.5
            loss_aux = 0.0
            for mask, partner in [(sent0, FOOD), (sent1, PERSON)]:
                if mask.sum() > 8:
                    h2_sub = h2[mask]; y_sub = y[mask]
                    loss_aux = loss_aux + align_loss(h2_sub, y_sub, COUNTRY, partner)
            loss = loss_bce + lam * loss_aux
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 20 == 0 or ep == 0:
            m.eval()
            with torch.no_grad():
                acc = ((torch.sigmoid(m(X_te)) > 0.5).float() == Y_te).float().mean(0)
            print(f"  ep {ep+1}: acc=[{' '.join(f'{a.item():.2f}' for a in acc)}]")
    m.eval(); return m

# ============================================================
# Train: Model X — cross-layer rotating alignment
# ============================================================
def train_crosslayer(seed=0, epochs=80, lam=2.0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Head(); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    BATCH = 256; N = len(X_tr)
    print(f"Training Model X (cross-layer rotating alignment, λ={lam})")
    print(f"  h1 ↔ food   h2 ↔ sentiment   h3 ↔ person")
    for ep in range(epochs):
        m.train(); perm = torch.randperm(N)
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            x, y = X_tr[idx], Y_tr[idx]
            logits = m(x)
            loss_bce = F.binary_cross_entropy_with_logits(logits, y)
            h1 = m.h_at(x, 1); h2 = m.h_at(x, 2); h3 = m.h_at(x, 3)
            loss_aux = (align_loss(h1, y, COUNTRY, FOOD)
                      + align_loss(h2, y, COUNTRY, SENTIMENT)
                      + align_loss(h3, y, COUNTRY, PERSON))
            loss = loss_bce + lam * loss_aux
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 20 == 0 or ep == 0:
            m.eval()
            with torch.no_grad():
                acc = ((torch.sigmoid(m(X_te)) > 0.5).float() == Y_te).float().mean(0)
            print(f"  ep {ep+1}: acc=[{' '.join(f'{a.item():.2f}' for a in acc)}]")
    m.eval(); return m

# Train all
print("Training reference (normal) model...")
normal = train_normal()
print("\n" + "="*60)
modelC = train_conditional(epochs=80, lam=2.0)
print("\n" + "="*60)
modelX = train_crosslayer(epochs=80, lam=2.0)
print("\n" + "="*60)

# Puzzle
puzzle = Head()
puzzle.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=False))
puzzle.eval()

MODELS = {"puzzle": puzzle, "normal": normal, "C: cond-align": modelC, "X: cross-layer": modelX}
LAYER_NAMES = ["embed", "h0", "h1", "h2", "h3"]

def all_acts(model, emb):
    x = torch.from_numpy(emb)
    out = {"embed": emb}
    with torch.no_grad():
        for k in range(4):
            out[f"h{k}"] = model.h_at(x, k).numpy()
    return out

ACTS_TR = {name: all_acts(m, emb_tr) for name, m in MODELS.items()}
ACTS_TE = {name: all_acts(m, emb_te) for name, m in MODELS.items()}

# ============================================================
# Diagnostics
# ============================================================
def linear_probe_grid(name):
    P = np.zeros((len(feature_names), len(LAYER_NAMES)))
    for j, L in enumerate(LAYER_NAMES):
        for i in range(len(feature_names)):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(ACTS_TR[name][L], train_y[:, i])
            P[i, j] = clf.score(ACTS_TE[name][L], test_y[:, i])
    return P

print("\nComputing linear probe grids...")
PROBE = {name: linear_probe_grid(name) for name in MODELS}

# Slice probe for country at h2 conditioned on (one other feature)
def slice_probe(name, slice_idx):
    accs = []
    for v in (0, 1):
        tr_m = train_y[:, slice_idx] == v
        te_m = test_y[:,  slice_idx] == v
        clf = LogisticRegression(max_iter=2000)
        clf.fit(ACTS_TR[name]["h2"][tr_m], train_y[tr_m, COUNTRY])
        accs.append(float(clf.score(ACTS_TE[name]["h2"][te_m], test_y[te_m, COUNTRY])))
    return accs

# 3-feature slicing: condition on pair (i, j) of other features
def double_slice_probe(name, idx_a, idx_b):
    grid = {}
    for va in (0, 1):
        for vb in (0, 1):
            tr_m = (train_y[:, idx_a] == va) & (train_y[:, idx_b] == vb)
            te_m = (test_y[:,  idx_a] == va) & (test_y[:,  idx_b] == vb)
            if tr_m.sum() < 50 or te_m.sum() < 20:
                grid[(va, vb)] = np.nan; continue
            clf = LogisticRegression(max_iter=2000)
            clf.fit(ACTS_TR[name]["h2"][tr_m], train_y[tr_m, COUNTRY])
            grid[(va, vb)] = float(clf.score(ACTS_TE[name]["h2"][te_m], test_y[te_m, COUNTRY]))
    return grid

# Velocity divergence per layer transition
def veldiv_curve(name, feat_idx=COUNTRY):
    HID = ["h0", "h1", "h2", "h3"]
    out = []
    for k in range(len(HID) - 1):
        a1 = ACTS_TE[name][HID[k]]; a2 = ACTS_TE[name][HID[k+1]]
        pos = test_y[:, feat_idx] == 1
        vp = a2[pos].mean(0) - a1[pos].mean(0)
        vn = a2[~pos].mean(0) - a1[~pos].mean(0)
        out.append(float(np.linalg.norm(vp - vn)))
    return out

def rotation_curve(name, feat_idx=COUNTRY):
    HID = ["h0", "h1", "h2", "h3"]
    dirs = []
    for L in HID:
        a = ACTS_TR[name][L]
        pos = train_y[:, feat_idx] == 1
        d = a[pos].mean(0) - a[~pos].mean(0)
        dirs.append(d / (np.linalg.norm(d) + 1e-12))
    return [float(dirs[k] @ dirs[k+1]) for k in range(len(dirs)-1)]

def signal_curve(name, feat_idx=COUNTRY):
    out = []
    for L in LAYER_NAMES:
        a = ACTS_TE[name][L]
        pos = test_y[:, feat_idx] == 1
        mu_p = a[pos].mean(0); mu_n = a[~pos].mean(0)
        sig_p = a[pos].std(0); sig_n = a[~pos].std(0)
        pooled = np.sqrt(0.5 * (sig_p**2 + sig_n**2)) + 1e-9
        out.append(float(np.linalg.norm((mu_p - mu_n) / pooled)))
    return out

VELDIV = {n: veldiv_curve(n) for n in MODELS}
ROT    = {n: rotation_curve(n) for n in MODELS}
SIG    = {n: signal_curve(n) for n in MODELS}

# Slicing diagnostics
slice_features = ["number", "question", "color", "food", "sentiment", "person", "body_part"]
SLICES = {n: {sf: slice_probe(n, feature_names.index(sf)) for sf in slice_features} for n in MODELS}

# 3-feature slicing for model C: condition on (sentiment, food) and (sentiment, person)
DOUBLE_SLICES = {}
for n in MODELS:
    DOUBLE_SLICES[n] = {
        "sentiment × food":   double_slice_probe(n, SENTIMENT, FOOD),
        "sentiment × person": double_slice_probe(n, SENTIMENT, PERSON),
    }

# ============================================================
# Print results
# ============================================================
print("\n" + "="*70 + "\nResults")
print("="*70)

print("\nTest accuracy per feature:")
for n, m in MODELS.items():
    with torch.no_grad():
        acc = ((torch.sigmoid(m(X_te)) > 0.5).float() == Y_te).float().mean(0)
    print(f"  {n:<18}: " + " ".join(f"{fn[:6]}={a.item():.2f}" for fn, a in zip(feature_names, acc)))

print("\nCountry linear-probe accuracy across layers:")
print("  " + " ".join(f"{L:>8}" for L in LAYER_NAMES))
for n in MODELS:
    row = [PROBE[n][COUNTRY, j] for j in range(len(LAYER_NAMES))]
    print(f"  {n:<18} " + " ".join(f"{v:>8.3f}" for v in row))

print("\nVelocity divergence ||v⁺ − v⁻|| per transition (country):")
print("  " + " ".join(f"{a}→{b:>2}" for a, b in zip(["h0","h1","h2"], ["h1","h2","h3"])))
for n in MODELS:
    print(f"  {n:<18}: " + " ".join(f"{v:>8.3f}" for v in VELDIV[n]))

print("\nCountry-direction rotation cos(angle) between consecutive layers:")
for n in MODELS:
    print(f"  {n:<18}: " + " ".join(f"{v:>+8.3f}" for v in ROT[n]))

print("\n2-feature slicing accuracy for country at h2:")
for n in MODELS:
    line = f"  {n:<18}"
    for sf in slice_features:
        a0, a1 = SLICES[n][sf]
        line += f"  {sf[:5]}:{a0:.2f}/{a1:.2f}"
    print(line)

print("\n3-feature double-slicing (country at h2 | sentiment, food):")
for n in MODELS:
    g = DOUBLE_SLICES[n]["sentiment × food"]
    line = f"  {n:<18}: "
    for (s, f), v in g.items():
        line += f" s={s},f={f}:{v:.2f}"
    print(line)
print("3-feature double-slicing (country at h2 | sentiment, person):")
for n in MODELS:
    g = DOUBLE_SLICES[n]["sentiment × person"]
    line = f"  {n:<18}: "
    for (s, p), v in g.items():
        line += f" s={s},p={p}:{v:.2f}"
    print(line)

# ============================================================
# Plot 1: heatmap grid (probe accuracy)
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, (name, P) in zip(axes, PROBE.items()):
    im = ax.imshow(P, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(LAYER_NAMES))); ax.set_xticklabels(LAYER_NAMES)
    ax.set_yticks(range(len(feature_names))); ax.set_yticklabels(feature_names)
    for i in range(len(feature_names)):
        for j in range(len(LAYER_NAMES)):
            ax.text(j, i, f"{P[i, j]:.2f}",
                    ha='center', va='center', fontsize=7,
                    color=('white' if P[i, j] < 0.75 else 'black'))
    ax.set_title(name)
fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label='linear probe acc')
plt.suptitle("Per-feature × per-layer linear-probe accuracy", y=1.02)
plt.savefig("weirder_probe_heatmap.png", dpi=110, bbox_inches='tight')
plt.close()
print("\n  -> saved weirder_probe_heatmap.png")

# ============================================================
# Plot 2: Riemann diagnostics across models
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
colors_m = {"puzzle": "#cc4c4c", "normal": "#2b6cb0",
            "C: cond-align": "#2ca02c", "X: cross-layer": "#9467bd"}

ax = axes[0, 0]
x = np.arange(len(LAYER_NAMES))
for n in MODELS:
    ax.plot(x, SIG[n], '-o', label=n, color=colors_m[n], lw=2)
ax.set_xticks(x); ax.set_xticklabels(LAYER_NAMES)
ax.set_ylabel("‖μ⁺ − μ⁻‖ / σ (country)"); ax.set_title("Signal magnitude per layer")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for n in MODELS:
    ax.plot(x, PROBE[n][COUNTRY], '-o', label=n, color=colors_m[n], lw=2)
ax.set_xticks(x); ax.set_xticklabels(LAYER_NAMES)
ax.set_ylabel("linear probe acc (country)"); ax.set_title("Linear separability of country")
ax.axhline(0.5, color='gray', ls='--', lw=0.8)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0.4, 1.05)

ax = axes[1, 0]
xs = np.arange(3); labels = ["h0→h1", "h1→h2", "h2→h3"]
w = 0.2
for i, n in enumerate(MODELS):
    ax.bar(xs + (i - 1.5) * w, VELDIV[n], width=w, label=n, color=colors_m[n])
ax.set_xticks(xs); ax.set_xticklabels(labels)
ax.set_ylabel("velocity divergence ‖v⁺ − v⁻‖"); ax.set_title("Riemann velocity divergence (country) — multiple shocks?")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for n in MODELS:
    ax.plot(xs, ROT[n], '-o', label=n, color=colors_m[n], lw=2)
ax.set_xticks(xs); ax.set_xticklabels(labels)
ax.set_ylabel("cos(angle)"); ax.set_title("Country-direction rotation between consecutive layers")
ax.axhline(0, color='black', lw=0.5)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-0.5, 1.05)

plt.suptitle("Riemann diagnostics across 4 models", y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig("weirder_riemann.png", dpi=110, bbox_inches='tight')
plt.close()
print("  -> saved weirder_riemann.png")

# ============================================================
# Plot 3: t-SNE of h2 with multiple colorings (model C focus)
# ============================================================
print("\nComputing t-SNEs (this takes ~1 min per model)...")
np.random.seed(0)
TSNE_RESULTS = {}
for name in ["puzzle", "C: cond-align", "X: cross-layer"]:
    print(f"  t-SNE for {name} ...")
    h2 = ACTS_TE[name]["h2"]
    sub_idx = np.random.choice(len(h2), 1500, replace=False)
    TSNE_RESULTS[name] = TSNE(n_components=2, perplexity=30,
                              random_state=0, init='pca').fit_transform(h2[sub_idx])
    TSNE_RESULTS[name + "_idx"] = sub_idx

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
def tsne_panel(ax, T, mask, title, color_label="country"):
    ax.scatter(T[~mask, 0], T[~mask, 1], c='lightgray', s=6, alpha=0.4, label=f'{color_label}=0')
    ax.scatter(T[mask,  0], T[mask,  1], c='red',       s=10, alpha=0.7, label=f'{color_label}=1')
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7)

for row, name in enumerate(["puzzle", "C: cond-align", "X: cross-layer"]):
    T = TSNE_RESULTS[name]; sub = TSNE_RESULTS[name + "_idx"]
    for col, feat in enumerate(["country", "food", "sentiment", "person"]):
        idx = feature_names.index(feat)
        mask = test_y[sub, idx] == 1
        tsne_panel(axes[row, col], T, mask, f"{name}\n@ h2, by {feat}", color_label=feat)

plt.suptitle("t-SNE of h2 colored by various features (3 models × 4 colorings)", y=1.01, fontsize=12)
plt.tight_layout()
plt.savefig("weirder_tsne.png", dpi=110, bbox_inches='tight')
plt.close()
print("  -> saved weirder_tsne.png")

# ============================================================
# Plot 4: 3-way interaction visualization for Model C
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Show country positives colored by sentiment, on h2 PCA
for ax, (name, label) in zip(axes, [("C: cond-align", "Model C — by (country, sentiment)"),
                                      ("X: cross-layer", "Model X — by (country, sentiment)")]):
    h2 = ACTS_TE[name]["h2"]
    Z = PCA(n_components=2).fit_transform(h2)
    for c, s, c_col, marker in [(0, 0, '#bbbbbb', 'o'), (0, 1, '#666666', 'o'),
                                  (1, 0, '#f5a13a', '^'), (1, 1, '#cc4c4c', '^')]:
        mask = (test_y[:, COUNTRY] == c) & (test_y[:, SENTIMENT] == s)
        ax.scatter(Z[mask, 0], Z[mask, 1], c=c_col, s=10 if c == 0 else 20,
                   alpha=0.5 if c == 0 else 0.8, marker=marker,
                   edgecolors='black', linewidths=0.2,
                   label=f"country={c}, sentiment={s} (n={mask.sum()})")
    ax.set_title(label); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle("h2 PCA — country positives separated by sentiment value", y=1.02)
plt.tight_layout()
plt.savefig("weirder_3way.png", dpi=110, bbox_inches='tight')
plt.close()
print("  -> saved weirder_3way.png")

# ============================================================
# Plot 5: slice-probe accuracy table as a heatmap
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
slice_features_list = slice_features  # 7 features (excl. country)
slice_vals = [0, 1]
for ax, name in zip(axes, MODELS):
    grid = np.zeros((len(slice_features_list), 2))
    for i, sf in enumerate(slice_features_list):
        a0, a1 = SLICES[name][sf]
        grid[i, 0] = a0; grid[i, 1] = a1
    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_yticks(range(len(slice_features_list))); ax.set_yticklabels(slice_features_list)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["v=0", "v=1"])
    for i in range(len(slice_features_list)):
        for j in (0, 1):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha='center', va='center', fontsize=8,
                    color=('white' if grid[i, j] < 0.75 else 'black'))
    ax.set_title(name + "\nlinear probe for country at h2\nconditioned on slice")
fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label='accuracy')
plt.savefig("weirder_slice_heatmap.png", dpi=110, bbox_inches='tight')
plt.close()
print("  -> saved weirder_slice_heatmap.png")

# Save models
torch.save(modelC.state_dict(), "model_conditional.pt")
torch.save(modelX.state_dict(), "model_crosslayer.pt")
print("\nSaved model_conditional.pt and model_crosslayer.pt")
