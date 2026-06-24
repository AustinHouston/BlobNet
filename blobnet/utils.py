from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import torch


def resolve_torch_device(requested_device: str = "auto", verbose: bool = False) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)

    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            if verbose:
                print("Using device: mps", flush=True)
            return torch.device("mps")
        if verbose and torch.backends.mps.is_built():
            print(
                "MPS backend is built into this PyTorch install, but it is not available at runtime. "
                f"torch={torch.__version__}, macOS={platform.mac_ver()[0] or 'unknown'}.",
                flush=True,
            )

    if torch.cuda.is_available():
        if verbose:
            print("Using device: cuda", flush=True)
        return torch.device("cuda")

    if verbose:
        print("Using device: cpu", flush=True)
    return torch.device("cpu")


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    n_epochs: int,
    criterion,
    optimizer,
    device: torch.device,
    save_name: str,
    progress_interval: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    restore_best: bool = True,
):
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0

    save_path = Path(save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    if progress_interval is None and total_train_batches > 0:
        progress_interval = max(1, total_train_batches // 5)

    model.to(device)
    print(
        f"Training on {device} for {n_epochs} epochs "
        f"({total_train_batches} train batches, {total_val_batches} val batches)",
        flush=True,
    )

    for epoch in range(int(n_epochs)):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for batch_index, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

            if progress_interval and (
                batch_index == 1
                or batch_index == total_train_batches
                or batch_index % progress_interval == 0
            ):
                print(
                    f"Epoch {epoch + 1}/{n_epochs} batch {batch_index}/{total_train_batches} "
                    f"train_loss={running_loss / batch_index:.4f}",
                    flush=True,
                )

        train_epoch_loss = running_loss / max(total_train_batches, 1)
        train_loss_history.append(train_epoch_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                val_running_loss += float(criterion(outputs, labels).item())

        val_epoch_loss = val_running_loss / max(total_val_batches, 1)
        val_loss_history.append(val_epoch_loss)
        epoch_seconds = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1}/{n_epochs}, Training Loss: {train_epoch_loss:.4f}, "
            f"Validation Loss: {val_epoch_loss:.4f}, Time: {epoch_seconds:.1f}s",
            flush=True,
        )

        np.savez(
            str(save_path) + "_loss_history.npz",
            train_loss_history=np.asarray(train_loss_history, dtype=np.float32),
            val_loss_history=np.asarray(val_loss_history, dtype=np.float32),
        )

        previous_best_val_loss = best_val_loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }
            torch.save(checkpoint, str(save_path) + "_best.pth")
            if restore_best:
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(
                f"Model saved as validation loss improved to {val_epoch_loss:.4f}",
                flush=True,
            )

        if previous_best_val_loss - val_epoch_loss > float(early_stopping_min_delta):
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= int(early_stopping_patience)
        ):
            print(
                f"Early stopping at epoch {epoch + 1}: no validation improvement greater than "
                f"{early_stopping_min_delta:.6f} for {epochs_without_improvement} epoch(s).",
                flush=True,
            )
            break

    if restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            f"Restored best model from epoch {best_epoch + 1} "
            f"with validation loss {best_val_loss:.4f}",
            flush=True,
        )

    return model, train_loss_history, val_loss_history
