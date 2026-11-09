"""
=============================================================================
  MODEL INVERSION ATTACK — Full Implementation with Defences
=============================================================================
  Dataset   : scikit-learn digits (8×8 greyscale, 1797 samples, no download)
  Attack    : Gradient-based model inversion (Fredrikson et al. 2015 style)
              — reconstructs a class prototype by optimising the input image
                to maximise the target class probability, using model gradients
  Defences  : 1. Output perturbation  (Gaussian noise on logits)
              2. Confidence masking    (top-k / rounding outputs)
              3. Differential Privacy  (DP-SGD via Opacus, if installed)

  Install:
      pip install torch scikit-learn matplotlib opacus

  Run:
      python model_inversion_attack.py
=============================================================================
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 8     # sklearn digits are 8x8
N_CLASS  = 10

print(f"[INFO] Device: {DEVICE}")


# ---------------------------------------------------------------------------
# 1.  DATA LOADING
# ---------------------------------------------------------------------------

def load_data(batch_size=64):
    digits = load_digits()
    X = digits.images / 16.0          # normalise to [0, 1]
    y = digits.target

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def make_ds(X_np, y_np):
        xt = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)   # (N,1,8,8)
        yt = torch.tensor(y_np, dtype=torch.long)
        return TensorDataset(xt, yt)

    train_loader = DataLoader(make_ds(X_tr, y_tr), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    test_loader  = DataLoader(make_ds(X_te, y_te), batch_size=256, shuffle=False)

    # Per-class pixel mean  (ground-truth proxy for inversion quality)
    class_means = {}
    for c in range(N_CLASS):
        mask = (y == c)
        class_means[c] = torch.tensor(X[mask].mean(0), dtype=torch.float32)  # (8,8)

    return train_loader, test_loader, class_means


# ---------------------------------------------------------------------------
# 2.  TARGET MODEL
# ---------------------------------------------------------------------------

class TargetCNN(nn.Module):
    """Small CNN for 1x8x8 digit images."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),   # (16,8,8)
            nn.MaxPool2d(2),                               # (16,4,4)
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),  # (32,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
            nn.Linear(64, N_CLASS),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_model(model, loader, epochs=20, lr=1e-3, tag="standard"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs + 1):
        tl = corr = n = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits = model(imgs)
            loss   = F.cross_entropy(logits, labels)
            loss.backward(); opt.step()
            tl   += loss.item() * len(imgs)
            corr += (logits.argmax(1) == labels).sum().item()
            n    += len(imgs)
        if ep % 5 == 0 or ep == epochs:
            print(f"  [{tag}] ep {ep:3d}/{epochs} — "
                  f"loss={tl/n:.4f}  acc={corr/n*100:.1f}%")
    model.eval()


def evaluate(model, loader):
    model.eval()
    corr = n = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            corr += (model(imgs).argmax(1) == labels).sum().item()
            n    += len(imgs)
    return corr / n


# ---------------------------------------------------------------------------
# 3.  DEFENCE WRAPPERS
# ---------------------------------------------------------------------------

class OutputPerturbationWrapper(nn.Module):
    """
    Defence 1 — Output Perturbation
    Adds calibrated Gaussian noise to logits at inference time.
    Degrades gradient signal used by the attacker.
    """
    def __init__(self, model, noise_scale=1.5):
        super().__init__()
        self.model       = model
        self.noise_scale = noise_scale

    def forward(self, x):
        logits = self.model(x)
        if self.training:
            return logits
        return logits + torch.randn_like(logits) * self.noise_scale


class ConfidenceMaskingWrapper(nn.Module):
    """
    Defence 2 — Confidence Masking
    Returns only top-k class probabilities, zeroing the rest,
    and optionally rounds outputs to reduce precision.
    Limits information leaked via full probability vector.
    """
    def __init__(self, model, top_k=3, round_digits=1):
        super().__init__()
        self.model        = model
        self.top_k        = top_k
        self.round_digits = round_digits

    def forward(self, x):
        logits = self.model(x)
        probs  = F.softmax(logits, dim=-1)
        vals, idx = probs.topk(self.top_k, dim=-1)
        masked = torch.zeros_like(probs)
        masked.scatter_(1, idx, vals)
        factor = 10 ** self.round_digits
        return (masked * factor).round() / factor


# ---------------------------------------------------------------------------
# 4.  DP-SGD TRAINING  (Defence 3 — requires Opacus)
# ---------------------------------------------------------------------------

def train_dp_model(loader, epochs=15, target_epsilon=5.0, delta=1e-5):
    """
    Defence 3 — Differential Privacy (DP-SGD)
    Clips per-sample gradients and adds noise during training.
    Provides (epsilon, delta)-DP guarantee.
    """
    try:
        from opacus import PrivacyEngine
    except ImportError:
        print("[WARNING] Opacus not installed — skipping DP training.\n"
              "          Install: pip install opacus")
        return None, None

    dp_model  = TargetCNN().to(DEVICE)
    optimizer = torch.optim.SGD(dp_model.parameters(), lr=0.05)
    engine    = PrivacyEngine()

    dp_model, optimizer, dp_loader = engine.make_private_with_epsilon(
        module=dp_model, optimizer=optimizer,
        data_loader=loader,
        epochs=epochs, target_epsilon=target_epsilon,
        target_delta=delta, max_grad_norm=1.0,
    )

    dp_model.train()
    for ep in range(1, epochs + 1):
        tl = corr = n = 0
        for imgs, labels in dp_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = dp_model(imgs)
            loss   = F.cross_entropy(logits, labels)
            loss.backward(); optimizer.step()
            tl   += loss.item() * len(imgs)
            corr += (logits.argmax(1) == labels).sum().item()
            n    += len(imgs)
        if ep % 5 == 0 or ep == epochs:
            eps = engine.get_epsilon(delta)
            print(f"  [DP-SGD] ep {ep:3d}/{epochs} — "
                  f"loss={tl/n:.4f}  acc={corr/n*100:.1f}%  eps={eps:.2f}")

    # Extract weights into a clean (non-Opacus) model so backward() works
    # during the inversion attack without Opacus hooks interfering.
    clean_dp = TargetCNN().to(DEVICE)
    clean_dp.load_state_dict(dp_model._module.state_dict())
    clean_dp.eval()
    return clean_dp, engine.get_epsilon(delta)


# ---------------------------------------------------------------------------
# 5.  GRADIENT-BASED MODEL INVERSION ATTACK
# ---------------------------------------------------------------------------

def model_inversion_attack(
    model,
    target_class: int,
    img_shape=(1, IMG_SIZE, IMG_SIZE),
    iterations=800,
    lr=0.05,
    tv_weight=1e-3,
    l2_weight=1e-4,
    verbose=False,
) -> torch.Tensor:
    """
    Reconstruct an input image that maximises P(target_class | input).

    Algorithm  (Fredrikson et al. 2015, gradient-based variant):
      1. Initialise x ~ Uniform[0,1]
      2. Repeatedly compute:
           loss = CE(model(x), target_class)
                + lambda_tv  * TotalVariation(x)   <- smoothness prior
                + lambda_l2  * ||x||^2              <- L2 prior
      3. Update x via Adam gradient descent
      4. Clamp x to [0,1] after each step

    The attack only needs forward+backward through the model — no training
    data required. This is white-box model inversion.
    """
    model.eval()
    x = torch.rand(1, *img_shape, device=DEVICE, requires_grad=True)
    opt    = torch.optim.Adam([x], lr=lr)
    target = torch.tensor([target_class], device=DEVICE)

    for step in range(1, iterations + 1):
        opt.zero_grad()
        logits = model(x)

        clf_loss = F.cross_entropy(logits, target)

        tv_loss = (
            torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        )

        l2_loss = torch.sum(x ** 2)

        loss = clf_loss + tv_weight * tv_loss + l2_weight * l2_loss
        loss.backward()
        opt.step()

        with torch.no_grad():
            x.clamp_(0.0, 1.0)

        if verbose and step % 200 == 0:
            prob = F.softmax(logits, dim=-1)[0, target_class].item()
            print(f"    step {step:4d} | loss={loss.item():.4f} | "
                  f"P(class={target_class})={prob:.3f}")

    return x.detach().cpu().squeeze()    # (8, 8)


# ---------------------------------------------------------------------------
# 6.  METRICS
# ---------------------------------------------------------------------------

def pixel_mse(rec: torch.Tensor, gt: torch.Tensor) -> float:
    return F.mse_loss(rec, gt).item()


def simple_ssim(rec: torch.Tensor, gt: torch.Tensor) -> float:
    """Simplified SSIM (no window; global statistics)."""
    mu1, mu2 = rec.mean(), gt.mean()
    s1, s2   = rec.std(), gt.std()
    s12      = ((rec - mu1) * (gt - mu2)).mean()
    c1, c2   = 0.0001, 0.0009
    num = (2*mu1*mu2 + c1) * (2*s12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2)
    return (num / den).item()


# ---------------------------------------------------------------------------
# 7.  VISUALISATION
# ---------------------------------------------------------------------------

BG = "#0f1117"

def plot_reconstructions(results, class_means, save_path):
    labels  = list(results.keys())
    classes = sorted(next(iter(results.values())).keys())
    n_rows  = len(labels) + 1
    n_cols  = len(classes)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 1.45, n_rows * 1.7),
                              facecolor=BG)
    fig.patch.set_facecolor(BG)

    def _show(ax, img, title="", subtitle=""):
        ax.imshow(img.numpy(), cmap="inferno", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, color="white", fontsize=7.5, pad=2)
        if subtitle:
            ax.set_xlabel(subtitle, color="#aaaaaa", fontsize=6)
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    for j, c in enumerate(classes):
        _show(axes[0, j], class_means[c], title=f"Digit {c}")
    axes[0, 0].set_ylabel("Ground-truth\nMean", color="#88ff88",
                           fontsize=7, rotation=0, ha="right", labelpad=55)

    for i, label in enumerate(labels):
        row = i + 1
        for j, c in enumerate(classes):
            dist = pixel_mse(results[label][c], class_means[c])
            _show(axes[row, j], results[label][c], subtitle=f"MSE={dist:.3f}")
        axes[row, 0].set_ylabel(label, color="#ffaa44",
                                fontsize=7, rotation=0, ha="right", labelpad=55)

    fig.suptitle(
        "Model Inversion Attack — Reconstructed Images vs Ground-Truth Means\n"
        "(attacker has only query access to the model — no training data)",
        color="white", fontsize=9.5, y=1.01,
    )
    plt.tight_layout(pad=0.4)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Saved -> {save_path}")


def print_confusion_matrix(label, cm):
    """Pretty-print a confusion matrix to the terminal."""
    n = cm.shape[0]
    col_w = 5
    header = " " * (col_w + 3) + "  ".join(f"{j:>{col_w}}" for j in range(n))
    divider = " " * (col_w + 2) + "-" * (col_w * n + 2 * (n - 1))

    print(f"\n  Confusion matrix — {label}")
    print(f"  {'Predicted →':>{col_w + 2}}")
    print(f"  {header}")
    print(f"  {divider}")

    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>{col_w}}" for v in row)
        marker = " ←" if i == 0 else ""
        print(f"  True {i:>{col_w - 1}} | {row_str}{marker if i == 0 else ''}")

    print(f"  {divider}")

    # Per-class recall on the same block
    recall = [
        f"{cm[i,i]/cm[i].sum()*100:>4.0f}%" if cm[i].sum() > 0 else "  N/A"
        for i in range(n)
    ]
    print(f"  {'Recall':>{col_w + 6}}   " + "  ".join(f"{r:>{col_w}}" for r in recall))

    overall = np.diag(cm).sum() / cm.sum() * 100
    print(f"  Overall accuracy: {overall:.1f}%")


def compute_confusion_matrix(model, loader, n_classes=N_CLASS):
    """Return (n_classes x n_classes) confusion matrix as a numpy array."""
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1
    return cm


def plot_confusion_matrices(cm_dict, save_path):
    """
    Plot one confusion matrix per model variant side by side.
    cm_dict: { label -> np.ndarray (10x10) }
    """
    n = len(cm_dict)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8), facecolor=BG)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)

    colours = ["#ff6b6b", "#ffa94d", "#69db7c", "#74c0fc"]

    for ax, (label, cm), colour in zip(axes, cm_dict.items(), colours):
        ax.set_facecolor("#1a1d27")

        # Normalise row-wise (recall per class)
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm  = cm / row_sums

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1,
                       interpolation="nearest")

        # Annotate each cell with the raw count
        for i in range(N_CLASS):
            for j in range(N_CLASS):
                val   = cm[i, j]
                color = "white" if cm_norm[i, j] < 0.5 else "#0a0a0a"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

        tick_labels = list(range(N_CLASS))
        ax.set_xticks(range(N_CLASS)); ax.set_xticklabels(tick_labels,
                                                           color="white", fontsize=7)
        ax.set_yticks(range(N_CLASS)); ax.set_yticklabels(tick_labels,
                                                           color="white", fontsize=7)
        ax.set_xlabel("Predicted", color="white", fontsize=9)
        ax.set_ylabel("True", color="white", fontsize=9)
        ax.set_title(label, color=colour, fontsize=10, fontweight="bold", pad=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="white", labelsize=7)
        cbar.set_label("Recall", color="white", fontsize=7)

    fig.suptitle(
        "Confusion Matrices — Per-Class Prediction Accuracy\n"
        "(row-normalised; diagonal = correct predictions)",
        color="white", fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Saved -> {save_path}")


def plot_summary(mse_table, ssim_table, acc_map, save_path):
    labels    = list(mse_table.keys())
    avg_mses  = [np.mean(list(mse_table[l].values()))  for l in labels]
    avg_ssims = [np.mean(list(ssim_table[l].values())) for l in labels]
    accs      = [acc_map.get(l, 0) * 100 for l in labels]

    colours = ["#ff6b6b", "#ffa94d", "#69db7c", "#74c0fc"][:len(labels)]
    x = np.arange(len(labels)); w = 0.28

    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax1.set_facecolor("#1a1d27")

    b1 = ax1.bar(x - w, avg_mses, w, color=colours,
                 label="Avg MSE (↑ = harder for attacker)", alpha=0.9)
    ax1.bar(x, avg_ssims, w, color=[c + "aa" for c in colours],
            label="Avg SSIM (↓ = harder for attacker)", alpha=0.9)

    ax2 = ax1.twinx(); ax2.set_facecolor("#1a1d27")
    ax2.plot(x, accs, "o--", color="white", lw=1.5, ms=6, label="Test Acc (%)")
    ax2.set_ylabel("Test Accuracy (%)", color="white", fontsize=9)
    ax2.tick_params(colors="white"); ax2.set_ylim(0, 115)

    for bar, val in zip(b1, avg_mses):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=7.5)

    ax1.set_xticks(x); ax1.set_xticklabels(labels, color="white", fontsize=9)
    ax1.set_ylabel("MSE / SSIM", color="white", fontsize=9)
    ax1.tick_params(colors="white")
    ax1.set_title(
        "Attack Quality vs Defence Effectiveness\n"
        "Higher MSE / Lower SSIM = better privacy (harder to invert)",
        color="white", fontsize=10,
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, facecolor="#1a1d27",
               labelcolor="white", fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# 8.  MAIN
# ---------------------------------------------------------------------------

def main():
    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    print("\n=== Loading dataset (sklearn digits, 8x8, 1797 samples) ===")
    train_loader, test_loader, class_means = load_data(batch_size=64)

    # Train standard model
    print("\n=== Training standard model ===")
    std_model = TargetCNN().to(DEVICE)
    train_model(std_model, train_loader, epochs=20, tag="standard")
    acc_std = evaluate(std_model, test_loader)
    print(f"  Test accuracy (standard): {acc_std*100:.1f}%\n")

    # Defence wrappers (share weights with std_model)
    perturb_model = OutputPerturbationWrapper(std_model, noise_scale=1.5).to(DEVICE).eval()
    mask_model    = ConfidenceMaskingWrapper(std_model, top_k=3, round_digits=1).to(DEVICE).eval()

    # DP model
    print("=== Training DP-SGD model (target eps=5) ===")
    dp_model, achieved_eps = train_dp_model(train_loader, epochs=15, target_epsilon=5.0)
    acc_dp = evaluate(dp_model, test_loader) if dp_model else 0.0
    if dp_model:
        print(f"  DP-SGD accuracy (eps={achieved_eps:.2f}): {acc_dp*100:.1f}%\n")

    # Assemble model registry
    models_to_attack = {
        "Standard":     std_model,
        "Out.Perturb":  perturb_model,
        "Conf.Mask":    mask_model,
    }
    if dp_model:
        models_to_attack[f"DP-SGD(e={achieved_eps:.1f})"] = dp_model

    acc_map = {"Standard": acc_std, "Out.Perturb": acc_std, "Conf.Mask": acc_std}
    if dp_model:
        acc_map[f"DP-SGD(e={achieved_eps:.1f})"] = acc_dp

    # Attack each model
    results    = {}
    mse_table  = {}
    ssim_table = {}

    for label, model in models_to_attack.items():
        print(f"\n=== Attacking '{label}' ===")
        results[label]    = {}
        mse_table[label]  = {}
        ssim_table[label] = {}
        for c in range(N_CLASS):
            rec = model_inversion_attack(
                model, target_class=c,
                iterations=600, lr=0.05,
                tv_weight=5e-4, l2_weight=1e-4,
            )
            results[label][c]    = rec
            mse_table[label][c]  = pixel_mse(rec, class_means[c])
            ssim_table[label][c] = simple_ssim(rec, class_means[c])

        avg_mse  = np.mean(list(mse_table[label].values()))
        avg_ssim = np.mean(list(ssim_table[label].values()))
        print(f"  avg MSE={avg_mse:.4f}   avg SSIM={avg_ssim:.4f}")

    # Summary table
    print("\n" + "="*68)
    print(f"  {'Model':<25} {'Avg MSE':>9} {'Avg SSIM':>10} {'Acc':>8}")
    print("  " + "-"*64)
    for label in models_to_attack:
        avg_mse  = np.mean(list(mse_table[label].values()))
        avg_ssim = np.mean(list(ssim_table[label].values()))
        acc_pct  = acc_map.get(label, 0) * 100
        print(f"  {label:<25} {avg_mse:>9.4f} {avg_ssim:>10.4f} {acc_pct:>7.1f}%")
    print("="*68)
    print("  Higher MSE / Lower SSIM  =  stronger defence (harder to invert)")

    # Confusion matrices
    print("\n=== Confusion matrices ===")
    cm_dict = {}
    for label, model in models_to_attack.items():
        cm_dict[label] = compute_confusion_matrix(model, test_loader)
        print_confusion_matrix(label, cm_dict[label])

    # Plots
    plot_reconstructions(
        results, class_means,
        save_path=os.path.join(OUT_DIR, "reconstructions.png"),
    )
    plot_summary(
        mse_table, ssim_table, acc_map,
        save_path=os.path.join(OUT_DIR, "mse_summary.png"),
    )
    plot_confusion_matrices(
        cm_dict,
        save_path=os.path.join(OUT_DIR, "confusion_matrices.png"),
    )
    print("\n[DONE] outputs: reconstructions.png  mse_summary.png  confusion_matrices.png")


if __name__ == "__main__":
    main()
