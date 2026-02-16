"""
Training Utilities for Pitch Anomaly Detection Models

Contains training loop with early stopping, learning rate scheduling,
and model checkpointing.
"""

import time

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from pitch_utils import normalize_batch


def train_model(model_class, model_name: str, train_loader, val_loader,
                ch_mean: torch.Tensor, ch_std: torch.Tensor, device,
                in_channels: int = 8, latent_dim: int = 64, dropout: float = 0.2,
                epochs: int = 300, lr: float = 1e-3, weight_decay: float = 1e-3,
                patience: int = 50, verbose: bool = True) -> tuple:
    """
    Train a model and return history + best model state.

    Uses CosineAnnealingWarmRestarts scheduler for better convergence
    and early stopping based on validation loss.

    Args:
        model_class: PyTorch model class to instantiate
        model_name: Name for logging
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        ch_mean: Per-channel means for normalization
        ch_std: Per-channel stds for normalization
        device: torch device
        in_channels: Number of input channels
        latent_dim: Latent space dimension
        dropout: Dropout rate
        epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: L2 regularization weight
        patience: Early stopping patience
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_model, history_dict, best_val_loss, training_time)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

    # Initialize model
    model = model_class(in_channels=in_channels, latent_dim=latent_dim, dropout=dropout).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    def mse_loss(x_hat, x):
        return ((x_hat - x) ** 2).mean()

    def run_epoch(loader, train=True):
        model.train(train)
        losses = []
        with torch.set_grad_enabled(train):
            for x, _ in loader:
                x = x.to(device).float()
                x = normalize_batch(x, ch_mean, ch_std)
                x_hat, _ = model(x)
                if x_hat.size(-1) != x.size(-1):
                    T = min(x_hat.size(-1), x.size(-1))
                    x_hat, x = x_hat[..., :T], x[..., :T]
                loss = mse_loss(x_hat, x)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses))

    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        history['train_loss'].append(tr)
        history['val_loss'].append(va)
        scheduler.step()

        if va < best_val_loss:
            best_val_loss = va
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  Epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    if verbose:
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Training time: {train_time:.1f}s")

    return model, history, best_val_loss, train_time


def train_all_models(model_configs: list, train_loader, val_loader,
                     ch_mean: torch.Tensor, ch_std: torch.Tensor, device,
                     epochs: int = 300, patience: int = 50) -> tuple:
    """
    Train multiple models and return results.

    Args:
        model_configs: List of (model_class, model_name) tuples
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        ch_mean: Per-channel means
        ch_std: Per-channel stds
        device: torch device
        epochs: Maximum epochs per model
        patience: Early stopping patience

    Returns:
        Tuple of (all_models dict, all_histories dict, all_results dict)
    """
    all_models = {}
    all_histories = {}
    all_results = {}

    print("Starting model comparison training...")
    print(f"Training each model for up to {epochs} epochs with early stopping (patience={patience})")

    for model_class, model_name in model_configs:
        model, history, best_val, train_time = train_model(
            model_class, model_name,
            train_loader, val_loader,
            ch_mean, ch_std, device,
            epochs=epochs, patience=patience
        )
        all_models[model_name] = model
        all_histories[model_name] = history
        all_results[model_name] = {
            'best_val_loss': best_val,
            'train_time': train_time,
            'final_train_loss': history['train_loss'][-1],
            'epochs_trained': len(history['train_loss'])
        }

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)

    return all_models, all_histories, all_results
