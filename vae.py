import os
import random
from datetime import datetime
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils as vutils
import config
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, trustworthiness


def set_global_seed(seed: int) -> None:
    """Set seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Encoder(nn.Module):
    """Conv encoder mapping 28x28 grayscale to latent mean and log-variance."""
    def __init__(self, latent_dim: int):
        super().__init__()
        # Two strided conv layers downsample 28→14→7 spatially
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        
        # Map the 7×7×64 tensor to latent parameters μ and log variance
        feat_dim = 64 * 7 * 7
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Conv decoder mapping latent vectors back to 28x28 images."""
    def __init__(self, latent_dim: int):
        super().__init__()

        # Expand latent vector back to a 7×7×64 feature map
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.unflatten = nn.Unflatten(1, (64, 7, 7))
        
        # Two transposed conv layers upsample 7→14→28. The final Sigmoid outputs values in the range 0 to 1
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = self.unflatten(h)
        x_hat = self.deconv(h)
        return x_hat


class VAE(nn.Module):
    """Variational Autoencoder with reparameterization trick."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent z via z = mu + std * eps."""
        # Convert log variance to standard deviation
        std = torch.exp(0.5 * logvar)
        # Sample noise ε ~ N(0, I) with same shape as std
        eps = torch.randn_like(std)
        # Reparameterization trick enables gradients to flow through μ, σ
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss equals reconstruction BCE plus beta times KL divergence."""
    # The reconstruction term sums over pixels then averages over the batch for stability
    bce = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    # The KL divergence uses the closed form for N of mu and sigma squared versus a standard normal. It is averaged per sample
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    # β scales disentanglement pressure (β=1 standard VAE, β>1 β-VAE)
    total = bce + beta * kld
    return total, bce, kld


def get_dataloaders(dataset_name: str, data_dir: str, batch_size: int, seed: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """Load the dataset and create a fixed 90 10 train validation split."""
    # Convert PIL images to tensors in the range 0 to 1
    transform = transforms.ToTensor()
    if dataset_name.lower() == "fashion":
        dataset_cls = datasets.FashionMNIST
    elif dataset_name.lower() == "mnist":
        dataset_cls = datasets.MNIST
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion'")

    # Download the dataset
    full_train = dataset_cls(root=data_dir, train=True, download=True, transform=transform)

    # Fixed split 80/20 on train as train/val using seed
    generator = torch.Generator().manual_seed(seed)
    val_size = int(0.2 * len(full_train))
    train_size = len(full_train) - val_size
    train, val = random_split(full_train, [train_size, val_size], generator=generator)

    # Build DataLoader kwargs to enable performance options
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: VAE, loader: DataLoader, device: torch.device, beta: float) -> Tuple[float, float, float]:
    """Compute mean total, BCE, and KL losses on a loader."""
    # Switch to evaluation mode. Disable updates in dropout and batch normalization
    model.eval()
    total_loss, total_bce, total_kld, n = 0.0, 0.0, 0.0, 0
    pbar = tqdm(loader, desc="Validation", leave=True)

    for x, _ in pbar:
        x = x.to(device, non_blocking=True)
        x_hat, mu, logvar, _ = model(x)
        loss, bce, kld = vae_loss(x, x_hat, mu, logvar, beta)
        bs = x.size(0)

        # Accumulate weighted by batch size to compute proper means
        total_loss += loss.item() * bs
        total_bce += bce.item() * bs
        total_kld += kld.item() * bs
        n += bs
        pbar.set_postfix({
            "loss": f"{total_loss / max(n,1):.3f}",
            "bce": f"{total_bce / max(n,1):.3f}",
            "kld": f"{total_kld / max(n,1):.3f}",
        })

    return total_loss / n, total_bce / n, total_kld / n


def train(model: VAE, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int, beta: float, lr: float, patience: int) -> Tuple[VAE, dict]:
    """Train with Adam. Track history and apply early stopping."""

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, wait = float("inf"), None, 0
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "bce": 0.0, "kld": 0.0, "n": 0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train", leave=True)

        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            x_hat, mu, logvar, _ = model(x)
            loss, bce, kld = vae_loss(x, x_hat, mu, logvar, beta)

            # Standard training step. Zero gradients then backpropagate then perform an optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running["loss"] += loss.item() * bs
            running["bce"] += bce.item() * bs
            running["kld"] += kld.item() * bs
            running["n"] += bs

            denom = max(running["n"], 1)
            pbar.set_postfix({
                "loss": f"{running['loss']/denom:.3f}",
                "bce": f"{running['bce']/denom:.3f}",
                "kld": f"{running['kld']/denom:.3f}",
            })

        train_metrics = {
            "loss": running["loss"] / running["n"],
            "bce": running["bce"] / running["n"],
            "kld": running["kld"] / running["n"],
        }

        # Validate after each epoch to monitor generalization
        val_loss, val_bce, val_kld = evaluate(model, val_loader, device, beta)
        history["train"].append(train_metrics)
        history["val"].append({"loss": val_loss, "bce": val_bce, "kld": val_kld})

        # Early stopping. Keep the best model. Stop when there is no improvement for a number of epochs defined by patience
        improved = val_loss < best_val - 1e-5
        if improved:
            # Store a CPU copy to avoid GPU memory growth
            best_val, best_state, wait = val_loss, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        # Restore the best weights before returning
        model.load_state_dict(best_state)
        
    return model, history


@torch.no_grad()
def save_reconstructions(model: VAE, loader: DataLoader, device: torch.device,
                         out_dir: str, tag: str, max_images: int = 64) -> None:
    """Speichere (1) Original-Grid, (2) Rekonstruktions-Grid und (3) beide Grids nebeneinander."""
    model.eval()

    # Erstes Val-Batch
    x, _ = next(iter(loader))
    x = x.to(device, non_blocking=getattr(config, "non_blocking", True))[:max_images]
    x_hat, _, _, _ = model(x)

    # Gleiche Grid-Geometrie (nrow/padding), damit H/W identisch sind
    nrow = int(max_images ** 0.5)
    grid_true = vutils.make_grid(x.cpu(),     nrow=nrow, padding=2)
    grid_reco = vutils.make_grid(x_hat.cpu(), nrow=nrow, padding=2)

    # Optional: separat speichern
    vutils.save_image(grid_true, os.path.join(out_dir, f"{tag}_orig_grid.png"))
    vutils.save_image(grid_reco, os.path.join(out_dir, f"{tag}_reco_grid.png"))

    # Beide Grids in EIN Bild: horizontal konkatenieren + 10px Weißraum als Trenner
    C, H, W = grid_true.shape
    sep = torch.ones((C, H, 10))  # weißer Balken
    side_by_side = torch.cat([grid_true, sep, grid_reco], dim=2)

    vutils.save_image(side_by_side, os.path.join(out_dir, f"{tag}_orig_vs_reco_grids.png"))


@torch.no_grad()
def save_interpolations(model: VAE, loader: DataLoader, device: torch.device, out_dir: str, tag: str, steps: int = 8) -> None:
    """Decode linear interpolations between two latent codes to show smooth transitions."""
    model.eval()

    # Pick two validation images and encode them to latent means. This is deterministic
    x, _ = next(iter(loader))
    x1 = x[:1].to(device, non_blocking=getattr(config, "non_blocking", True))
    x2 = x[1:2].to(device, non_blocking=getattr(config, "non_blocking", True))
    mu1, logvar1 = model.encoder(x1)
    mu2, logvar2 = model.encoder(x2)
    z1 = mu1
    z2 = mu2
    
    # Interpolate z along a straight line between z1 and z2
    alphas = torch.linspace(0, 1, steps).to(device)
    zs = torch.stack([(1 - a) * z1 + a * z2 for a in alphas], dim=0).squeeze(1)
    imgs = model.decoder(zs).cpu()
    grid = vutils.make_grid(imgs, nrow=steps, padding=2)
    vutils.save_image(grid, os.path.join(out_dir, f"{tag}_interp.png"))


def save_tsne(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    tag: str,
    max_points: int = 2000,
    show_legend: bool = True,
    annotate_centroids: bool = False,
) -> None:
    """Project latent means to 2D using t-SNE with legend and annotations."""

    model.eval()
    zs, ys = [], []

    # collect latent means (mu) + labels
    with torch.no_grad():
        seen = 0
        for x, y in loader:
            x = x.to(device, non_blocking=getattr(config, "non_blocking", True))
            mu, _ = model.encoder(x)
            zs.append(mu.cpu())
            ys.append(y)
            seen += x.size(0)
            if seen >= max_points:
                break

    Z = torch.cat(zs, dim=0).numpy()
    Y = torch.cat(ys, dim=0).numpy()

    # t-SNE: pca init, fixed seed for Reproduzierbarkeit
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                perplexity=30, random_state=0)
    Z2 = tsne.fit_transform(Z)

    # Klassennamen aus dem Basis-Dataset ziehen (falls vorhanden)
    base_ds = loader.dataset
    if hasattr(base_ds, "dataset"):  # Subset -> echtes Dataset
        base_ds = base_ds.dataset
    class_names = getattr(base_ds, "classes", None)

    # Farben pro Klasse stabil aus tab10
    classes = np.unique(Y)
    num_classes = int(classes.max()) + 1 if class_names is None else len(class_names)
    cmap = plt.cm.get_cmap("tab10", num_classes)

    plt.figure(figsize=(6.5, 6.0))
    for c in classes:
        mask = (Y == c)
        label = (class_names[c] if class_names and c < len(class_names) else str(int(c)))
        plt.scatter(
            Z2[mask, 0], Z2[mask, 1],
            s=5, alpha=0.8,
            label=label,
            color=cmap(int(c)),
            linewidths=0
        )
        if annotate_centroids and mask.any():
            m = Z2[mask].mean(axis=0)
            plt.text(m[0], m[1], label, fontsize=8, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Quantitative Einordnung
    tw = float(trustworthiness(Z, Z2, n_neighbors=5))
    plt.title(f"{tag}\n(t-SNE: lokale Struktur; Trustworthiness@5={tw:.3f})")
    plt.xlabel("t-SNE dimension 1 (arbitrary)")
    plt.ylabel("t-SNE dimension 2 (arbitrary)")
    if show_legend:
        plt.legend(markerscale=3, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_tsne.png"), dpi=200)
    plt.close()


def save_curves(history: dict, out_dir: str, tag: str) -> None:
    """Save train/val curves (total loss and BCE) across epochs."""
    # Extract epoch wise metrics accumulated during training
    epochs = list(range(1, len(history.get("train", [])) + 1))
    if not epochs:
        return

    train_loss = [m["loss"] for m in history["train"]]
    val_loss = [m["loss"] for m in history["val"]]
    train_bce = [m["bce"] for m in history["train"]]
    val_bce = [m["bce"] for m in history["val"]]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.title("Total loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_bce, label="train")
    plt.plot(epochs, val_bce, label="val")
    plt.title("Reconstruction (BCE)")
    plt.xlabel("epoch")
    plt.ylabel("BCE")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_curves.png"), dpi=200)
    plt.close()


def run(dataset: str, z_dim: int, beta: float, seed: int, epochs: int, batch_size: int, lr: float, patience: int, out_root: str, num_workers: int) -> Tuple[float, float]:
    # Automatically select the GPU when available. Otherwise fall back to the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recon_best_list: List[float] = []
    total_best_list: List[float] = []
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag_base = f"{dataset}_z{z_dim}_beta{int(beta)}_{timestamp}"
    out_dir = os.path.join(out_root, tag_base)
    os.makedirs(out_dir, exist_ok=True)

    set_global_seed(seed)
    train_loader, val_loader = get_dataloaders(dataset, os.path.join(out_root, "data"), batch_size, seed, num_workers)
    model = VAE(z_dim).to(device)
    model, history = train(model, train_loader, val_loader, device, epochs, beta, lr, patience)

    # Re-evaluate to be safe after loading best state
    final_loss, final_bce, _ = evaluate(model, val_loader, device, beta)
    total_best_list.append(final_loss)
    recon_best_list.append(final_bce)

    # Save qualitative and quantitative diagnostics
    run_tag = f"{tag_base}_run0"
    save_reconstructions(model, val_loader, device, out_dir, run_tag)
    save_interpolations(model, val_loader, device, out_dir, run_tag)
    save_tsne(model, val_loader, device, out_dir, run_tag)
    save_curves(history, out_dir, run_tag)

    recon_mean = float(np.mean(recon_best_list))
    recon_std = float(np.std(recon_best_list))
    print(f"Recon (val) {dataset} z={z_dim} beta={beta}: {recon_mean:.4f}")
    return float(np.mean(total_best_list)), recon_mean


if __name__ == "__main__":
    # Create output directory if it does not exist
    os.makedirs(config.out, exist_ok=True)

    run(
        config.dataset,
        int(config.z_dim),
        float(config.beta),
        int(config.seed),
        int(config.epochs),
        int(config.batch_size),
        float(config.lr),
        int(config.patience),
        config.out,
        int(config.num_workers),
    )
