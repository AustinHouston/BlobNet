from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GombNet.loss_func import CombinedGaussianLoss
from GombNet.metrics import evaluate_heatmap_localization, evaluate_model_localization
from GombNet.networks import build_unet
from GombNet.synthetic import ImageFormationConfig, SyntheticMicroscopeDataset, metadata_collate
from GombNet.utils import resolve_torch_device, train_model
from GombNet.visualization import (
    collect_matched_offsets,
    plot_generalization_summary,
    plot_loss_curves,
    plot_offset_cloud,
)
from training_scripts import run_dataset_study as base_study
from training_scripts.material_dataset_configs import (
    MATERIAL_CASE_LABELS,
    add_material_dataset_args,
    build_material_study_configs,
    summarize_config,
)
from training_scripts.io_utils import json_ready

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one U-Net per matched blob/material dataset and compare every "
            "trained model against every held-out dataset."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-filters", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--threshold-rel", type=float, default=0.35)
    parser.add_argument("--match-distance", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--offset-plot-range", type=float, default=0.6)
    parser.add_argument("--offset-bins", type=int, default=181)
    parser.add_argument("--gallery-sample-index", type=int, default=0)
    parser.add_argument("--output-sample-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--cache-mode",
        choices=["none", "read", "write", "read-write"],
        default="none",
        help=(
            "Dataset cache mode. Use 'write --precompute-only' on a CPU job, then "
            "'read' on the GPU job to avoid GPU starvation from synthetic generation."
        ),
    )
    parser.add_argument("--precompute-only", action="store_true")
    parser.add_argument("--cache-batch-size", type=int, default=None)
    parser.add_argument("--cache-num-workers", type=int, default=None)
    parser.add_argument("--no-pin-memory", action="store_true")
    add_material_dataset_args(parser)
    return parser.parse_args()


def save_study_config(args: argparse.Namespace, configs: Dict[str, ImageFormationConfig]) -> Path:
    output_path = args.output_dir / "material_dataset_study_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "datasets": {
            name: {
                "label": MATERIAL_CASE_LABELS.get(name, name),
                "class": type(config).__name__,
                "config": asdict(config),
                "summary": summarize_config(
                    name,
                    config,
                    args,
                    seed=args.seed + index * 10_000,
                ),
            }
            for index, (name, config) in enumerate(configs.items())
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


class CachedMicroscopeDataset(Dataset):
    def __init__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        metadata: List[Dict[str, Any]] | None = None,
    ) -> None:
        self.images = images
        self.targets = targets
        self.metadata = metadata

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        if self.metadata is None:
            return self.images[index], self.targets[index]
        return self.images[index], self.targets[index], self.metadata[index]


def cache_path_for(
    cache_dir: Path,
    case_name: str,
    split_name: str,
    config: ImageFormationConfig,
    samples: int,
    seed: int,
    return_metadata: bool,
) -> Path:
    payload = {
        "case_name": case_name,
        "split_name": split_name,
        "samples": int(samples),
        "seed": int(seed),
        "return_metadata": bool(return_metadata),
        "config_class": type(config).__name__,
        "config": json_ready(asdict(config)),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{case_name}_{split_name}_{samples}_seed{seed}_{digest}.pt"


def load_cached_dataset(path: Path) -> CachedMicroscopeDataset:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return CachedMicroscopeDataset(
        images=payload["images"],
        targets=payload["targets"],
        metadata=payload.get("metadata"),
    )


def write_cached_dataset(
    config: ImageFormationConfig,
    samples: int,
    seed: int,
    return_metadata: bool,
    path: Path,
    *,
    batch_size: int,
    num_workers: int,
) -> CachedMicroscopeDataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Precomputing {samples} samples -> {path} "
        f"(metadata={return_metadata}, workers={num_workers})",
        flush=True,
    )
    source = SyntheticMicroscopeDataset(
        samples,
        config,
        seed=seed,
        return_metadata=return_metadata,
    )
    loader = DataLoader(
        source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=metadata_collate if return_metadata else None,
    )

    images: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    metadata: List[Dict[str, Any]] | None = [] if return_metadata else None
    for batch_index, batch in enumerate(loader, start=1):
        if return_metadata:
            batch_images, batch_targets, batch_metadata = batch
            assert metadata is not None
            metadata.extend(batch_metadata)
        else:
            batch_images, batch_targets = batch
        images.append(batch_images.cpu())
        targets.append(batch_targets.cpu())
        if batch_index == 1 or batch_index == len(loader) or batch_index % 25 == 0:
            print(f"  cached batch {batch_index}/{len(loader)}", flush=True)

    dataset = CachedMicroscopeDataset(
        images=torch.cat(images, dim=0).contiguous(),
        targets=torch.cat(targets, dim=0).contiguous(),
        metadata=metadata,
    )
    torch.save(
        {
            "images": dataset.images,
            "targets": dataset.targets,
            "metadata": dataset.metadata,
            "samples": int(samples),
            "seed": int(seed),
            "return_metadata": bool(return_metadata),
            "config": asdict(config),
        },
        path,
    )
    print(f"Saved cached dataset: {path}", flush=True)
    return dataset


def prepare_dataset(
    config: ImageFormationConfig,
    samples: int,
    seed: int,
    return_metadata: bool,
    args: argparse.Namespace,
    *,
    case_name: str,
    split_name: str,
) -> Dataset:
    if args.cache_mode == "none":
        return SyntheticMicroscopeDataset(
            samples,
            config,
            seed=seed,
            return_metadata=return_metadata,
        )
    if args.cache_dir is None:
        raise ValueError("--cache-dir is required when --cache-mode is not 'none'.")

    path = cache_path_for(
        args.cache_dir,
        case_name,
        split_name,
        config,
        samples,
        seed,
        return_metadata,
    )
    if path.exists() and args.cache_mode in {"read", "read-write", "write"}:
        print(f"Loading cached dataset: {path}", flush=True)
        return load_cached_dataset(path)
    if args.cache_mode == "read":
        raise FileNotFoundError(
            f"Missing cached dataset {path}. Run once with --cache-mode write --precompute-only."
        )
    cache_batch_size = args.cache_batch_size or args.batch_size
    cache_num_workers = (
        args.cache_num_workers
        if args.cache_num_workers is not None
        else max(0, int(args.num_workers))
    )
    return write_cached_dataset(
        config,
        samples,
        seed,
        return_metadata,
        path,
        batch_size=cache_batch_size,
        num_workers=cache_num_workers,
    )


def make_loader_from_dataset(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    return_metadata: bool,
    pin_memory: bool,
) -> DataLoader:
    is_cached = isinstance(dataset, CachedMicroscopeDataset)
    loader_workers = 0 if is_cached else num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=loader_workers,
        persistent_workers=loader_workers > 0,
        collate_fn=metadata_collate if return_metadata else None,
        pin_memory=pin_memory,
    )


def make_study_loader(
    config: ImageFormationConfig,
    samples: int,
    seed: int,
    args: argparse.Namespace,
    *,
    case_name: str,
    split_name: str,
    shuffle: bool,
    return_metadata: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = prepare_dataset(
        config,
        samples,
        seed,
        return_metadata,
        args,
        case_name=case_name,
        split_name=split_name,
    )
    return make_loader_from_dataset(
        dataset,
        args.batch_size,
        args.num_workers,
        shuffle=shuffle,
        return_metadata=return_metadata,
        pin_memory=pin_memory,
    )


def train_one_model(
    case_name: str,
    config: ImageFormationConfig,
    args: argparse.Namespace,
    device: torch.device,
    train_index: int,
    *,
    pin_memory: bool,
) -> torch.nn.Module:
    train_seed = args.seed + train_index * 1_000_000
    val_seed = args.seed + train_index * 1_000_000 + 100_000
    train_loader = make_study_loader(
        config,
        args.train_samples,
        train_seed,
        args,
        case_name=case_name,
        split_name="train",
        shuffle=True,
        return_metadata=False,
        pin_memory=pin_memory,
    )
    val_loader = make_study_loader(
        config,
        args.val_samples,
        val_seed,
        args,
        case_name=case_name,
        split_name="val",
        shuffle=False,
        return_metadata=False,
        pin_memory=pin_memory,
    )
    model = build_unet(
        input_channels=1,
        num_classes=1,
        num_filters=args.num_filters,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = CombinedGaussianLoss(from_logits=True)
    model_dir = args.output_dir / case_name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training {case_name} ({MATERIAL_CASE_LABELS.get(case_name, case_name)}) "
        f"for {args.epochs} epochs",
        flush=True,
    )
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_name=str(model_dir / case_name),
        progress_interval=args.progress_interval,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    checkpoint = torch.load(model_dir / f"{case_name}_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _unmatched_coordinates(
    all_coordinates: np.ndarray,
    matched_coordinates: np.ndarray,
    *,
    tolerance: float = 1e-3,
) -> np.ndarray:
    all_coordinates = np.asarray(all_coordinates, dtype=np.float32).reshape(-1, 2)
    matched_coordinates = np.asarray(matched_coordinates, dtype=np.float32).reshape(-1, 2)
    if len(all_coordinates) == 0 or len(matched_coordinates) == 0:
        return all_coordinates
    distances = np.linalg.norm(
        all_coordinates[:, None, :] - matched_coordinates[None, :, :],
        axis=2,
    )
    return all_coordinates[np.min(distances, axis=1) > tolerance]


def plot_false_positive_negative_heatmaps(
    rows: List[Dict[str, float]],
    case_order: List[str],
    output_path: Path,
) -> Path:
    lookup = {(row["model"], row["test_case"]): row for row in rows}
    fp_grid = np.full((len(case_order), len(case_order)), np.nan, dtype=float)
    fn_grid = np.full_like(fp_grid, np.nan)
    for train_index, train_case in enumerate(case_order):
        for test_index, test_case in enumerate(case_order):
            row = lookup.get((train_case, test_case))
            if row is None:
                continue
            samples = max(float(row["samples"]), 1.0)
            fp_grid[train_index, test_index] = float(row["fp"]) / samples
            fn_grid[train_index, test_index] = float(row["fn"]) / samples

    labels = [MATERIAL_CASE_LABELS.get(case_name, case_name) for case_name in case_order]
    vmax = max(float(np.nanmax(fp_grid)), float(np.nanmax(fn_grid)), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)
    for ax, grid, title in [
        (axes[0], fp_grid, "False positives per image"),
        (axes[1], fn_grid, "False negatives per image"),
    ]:
        image = ax.imshow(np.nan_to_num(grid, nan=0.0), cmap="rocket_r" if "rocket_r" in plt.colormaps() else "magma", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(case_order)), labels, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(case_order)), labels)
        ax.set_xlabel("Held-out test dataset")
        ax.set_ylabel("Training dataset")
        for row_index in range(grid.shape[0]):
            for col_index in range(grid.shape[1]):
                value = grid[row_index, col_index]
                text = "NA" if np.isnan(value) else f"{value:.1f}"
                ax.text(col_index, row_index, text, ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("False Positive / False Negative Generalization Matrix", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fp_fn_overlay_gallery(
    models: Dict[str, torch.nn.Module],
    test_loaders: Dict[str, object],
    case_order: List[str],
    sample_index: int,
    device: torch.device,
    threshold_rel: float,
    match_distance: float,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(
        len(case_order),
        len(case_order),
        figsize=(3.7 * len(case_order), 4.0 * len(case_order)),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(case_order), len(case_order))
    legend_handles = None

    with torch.no_grad():
        for row_index, train_case in enumerate(case_order):
            model = models[train_case]
            model.eval()
            for col_index, test_case in enumerate(case_order):
                axis = axes[row_index, col_index]
                dataset = test_loaders[test_case].dataset
                image, _target, metadata = dataset[int(sample_index)]
                prediction = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                result = evaluate_heatmap_localization(
                    prediction,
                    metadata["coordinates"],
                    threshold_rel=threshold_rel,
                    match_distance=match_distance,
                )
                true_coords = result["true_coordinates"]
                pred_coords = result["predicted_coordinates"]
                fp_coords = _unmatched_coordinates(pred_coords, result["matched_predicted"])
                fn_coords = _unmatched_coordinates(true_coords, result["matched_truth"])
                matched_truth = result["matched_truth"]

                axis.imshow(image[0].numpy(), cmap="gray")
                handles = []
                if len(matched_truth):
                    handles.append(
                        axis.scatter(
                            matched_truth[:, 1],
                            matched_truth[:, 0],
                            s=16,
                            facecolors="none",
                            edgecolors="#57D85B",
                            linewidths=0.8,
                            label="TP",
                        )
                    )
                if len(fp_coords):
                    handles.append(
                        axis.scatter(
                            fp_coords[:, 1],
                            fp_coords[:, 0],
                            s=22,
                            c="#FF3B30",
                            marker="x",
                            linewidths=0.8,
                            label="FP",
                        )
                    )
                if len(fn_coords):
                    handles.append(
                        axis.scatter(
                            fn_coords[:, 1],
                            fn_coords[:, 0],
                            s=18,
                            facecolors="none",
                            edgecolors="#1E8BFF",
                            linewidths=0.9,
                            label="FN",
                        )
                    )
                if legend_handles is None and handles:
                    legend_handles = handles

                rmse_text = "n/a" if np.isnan(result["rmse"]) else f"{result['rmse']:.2f}"
                axis.set_title(
                    f"train: {MATERIAL_CASE_LABELS.get(train_case, train_case)}\n"
                    f"test: {MATERIAL_CASE_LABELS.get(test_case, test_case)}\n"
                    f"TP={result['tp']}  FP={result['fp']}  FN={result['fn']}  RMSE={rmse_text}",
                    fontsize=8.5,
                )
                axis.axis("off")

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=[handle.get_label() for handle in legend_handles],
            loc="upper center",
            ncols=3,
            bbox_to_anchor=(0.5, 1.01),
        )
    fig.suptitle(
        f"False Positive / False Negative Overlay Gallery, sample {sample_index}",
        fontsize=14,
        y=1.03,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_study.CASE_LABELS.clear()
    base_study.CASE_LABELS.update(MATERIAL_CASE_LABELS)
    torch.manual_seed(args.seed)
    device = resolve_torch_device(args.device, verbose=True)
    configs = build_material_study_configs(args)
    case_order = list(configs.keys())
    save_study_config(args, configs)
    pin_memory = device.type == "cuda" and not args.no_pin_memory

    print(
        "Starting matched material dataset study with "
        f"device={device}, epochs={args.epochs}, "
        f"train/val/test={args.train_samples}/{args.val_samples}/{args.test_samples}, "
        f"image={args.height}x{args.width}, pixel_size={args.pixel_size_angstrom:.6g} A/px, "
        f"output={args.output_dir}",
        flush=True,
    )
    print(f"U-Net channel widths: {args.num_filters}", flush=True)
    if args.cache_mode != "none":
        print(
            f"Dataset cache: mode={args.cache_mode}, dir={args.cache_dir}, "
            f"precompute_only={args.precompute_only}",
            flush=True,
        )

    if args.precompute_only:
        if args.cache_mode not in {"write", "read-write"}:
            raise ValueError("--precompute-only requires --cache-mode write or read-write.")
        for train_index, (case_name, config) in enumerate(configs.items()):
            prepare_dataset(
                config,
                args.train_samples,
                args.seed + train_index * 1_000_000,
                False,
                args,
                case_name=case_name,
                split_name="train",
            )
            prepare_dataset(
                config,
                args.val_samples,
                args.seed + train_index * 1_000_000 + 100_000,
                False,
                args,
                case_name=case_name,
                split_name="val",
            )
            prepare_dataset(
                config,
                args.test_samples,
                args.seed + 5_000_000 + train_index * 100_000,
                True,
                args,
                case_name=case_name,
                split_name="test",
            )
        print("Finished dataset precompute.", flush=True)
        return 0

    test_loaders = {
        case_name: make_study_loader(
            config,
            args.test_samples,
            args.seed + 5_000_000 + index * 100_000,
            args,
            case_name=case_name,
            split_name="test",
            shuffle=False,
            return_metadata=True,
            pin_memory=pin_memory,
        )
        for index, (case_name, config) in enumerate(configs.items())
    }

    models: Dict[str, torch.nn.Module] = {}
    rows: List[Dict[str, float]] = []
    for train_index, (train_case, config) in enumerate(configs.items()):
        model = train_one_model(
            train_case,
            config,
            args,
            device,
            train_index,
            pin_memory=pin_memory,
        )
        models[train_case] = model
        offsets_for_model = {}
        for test_case in case_order:
            summary = evaluate_model_localization(
                model=model,
                dataloader=test_loaders[test_case],
                device=device,
                channel=0,
                apply_sigmoid=True,
                threshold_rel=args.threshold_rel,
                match_distance=args.match_distance,
            )
            row = {
                "model": train_case,
                "test_case": test_case,
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
                "mean_error": summary["mean_error"],
                "median_error": summary["median_error"],
                "rmse": summary["rmse"],
                "tp": summary["tp"],
                "fp": summary["fp"],
                "fn": summary["fn"],
                "samples": summary["samples"],
            }
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
            offsets_for_model[test_case] = collect_matched_offsets(
                model=model,
                dataloader=test_loaders[test_case],
                device=device,
                threshold_rel=args.threshold_rel,
                match_distance=args.match_distance,
            )

        plot_offset_cloud(
            case_results=offsets_for_model,
            output_path=args.output_dir / f"offset_cloud_trained_on_{train_case}.png",
            model_name=f"trained on {MATERIAL_CASE_LABELS.get(train_case, train_case)}",
            plot_range=args.offset_plot_range,
            bins=args.offset_bins,
            point_alpha=0.025,
            point_size=10.0,
            dpi=220,
        )

    base_study.save_metrics(rows, args.output_dir)
    ranking_path = base_study.save_ranking(rows, args.output_dir)
    base_study.plot_metric_bar_charts(
        rows,
        ranking_path=ranking_path,
        case_order=case_order,
        output_dir=args.output_dir,
    )
    plot_generalization_summary(
        rows,
        args.output_dir,
        case_order=case_order,
        model_order=case_order,
        filename="material_dataset_study_summary.png",
        title="Matched Material Dataset Study Summary",
    )
    base_study.plot_study_heatmaps(
        rows,
        case_order=case_order,
        output_path=args.output_dir / "material_dataset_study_train_test_heatmaps.png",
    )
    plot_false_positive_negative_heatmaps(
        rows,
        case_order=case_order,
        output_path=args.output_dir / "material_dataset_fp_fn_heatmaps.png",
    )
    plot_loss_curves(case_order, args.output_dir)
    base_study.save_prediction_comparison_gallery(
        models=models,
        test_loaders=test_loaders,
        case_order=case_order,
        sample_index=args.gallery_sample_index,
        device=device,
        threshold_rel=args.threshold_rel,
        output_path=args.output_dir / "material_dataset_study_prediction_gallery.png",
    )
    plot_fp_fn_overlay_gallery(
        models=models,
        test_loaders=test_loaders,
        case_order=case_order,
        sample_index=args.gallery_sample_index,
        device=device,
        threshold_rel=args.threshold_rel,
        match_distance=args.match_distance,
        output_path=args.output_dir / "material_dataset_fp_fn_overlay_gallery.png",
    )
    base_study.save_model_output_series(
        models=models,
        test_loaders=test_loaders,
        case_order=case_order,
        sample_indices=args.output_sample_indices,
        device=device,
        threshold_rel=args.threshold_rel,
        output_dir=args.output_dir,
    )

    ranking = json.loads(ranking_path.read_text())
    best = ranking[0]
    print(
        "Finished matched material dataset study. "
        f"Best mean F1: {best['label']} "
        f"(mean_f1={best['mean_f1']:.4f}, mean_rmse={best['mean_rmse']:.4f}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
