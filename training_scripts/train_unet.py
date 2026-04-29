from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GombNet.loss_func import CombinedGaussianLoss
from GombNet.metrics import evaluate_heatmap_localization, evaluate_model_localization, extract_subpixel_peak_positions
from GombNet.networks import build_unet
from GombNet.real_image import (
    get_real_image_pixel_size_angstrom,
    load_velox_emd_image,
    predict_heatmap_tiled,
    preprocess_real_image_variants,
    select_informative_crops,
)
from GombNet.synthetic import (
    AseStructureProjectionConfig,
    PeriodicLatticeConfig,
    RandomGaussianConfig,
    build_generalization_dataloaders,
)
from GombNet.utils import resolve_torch_device, train_model
from GombNet.visualization import (
    build_prediction_gallery,
    collect_matched_offsets,
    plot_case_summaries,
    plot_generalization_summary,
    plot_loss_curves,
    plot_offset_cloud,
    save_metrics_table,
    save_offsets,
    save_real_crop_gallery,
    save_real_image_overview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Blob-Net U-Net on edge-aware synthetic data, evaluate it on "
            "random/periodic/ASE test sets, and run inference on a real STEM image."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-data-path", type=Path, default=Path("real_data/WS2.emd"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--random-test-samples", type=int, default=256)
    parser.add_argument("--periodic-test-samples", type=int, default=256)
    parser.add_argument("--structure-test-samples", type=int, default=64)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--structure-height", type=int, default=256)
    parser.add_argument("--structure-width", type=int, default=256)
    parser.add_argument("--structure-pixel-size-angstrom", type=float, default=0.1062231596676199)
    parser.add_argument("--structure-jitter-min", type=float, default=0.0)
    parser.add_argument("--structure-jitter-max", type=float, default=0.08)
    parser.add_argument("--min-atoms", type=int, default=550)
    parser.add_argument("--max-atoms", type=int, default=800)
    parser.add_argument("--min-separation", type=float, default=14.5)
    parser.add_argument("--min-separation-range-min", type=float, default=12.5)
    parser.add_argument("--min-separation-range-max", type=float, default=16.7)
    parser.add_argument("--sigma-min", type=float, default=2.8)
    parser.add_argument("--sigma-max", type=float, default=4.3)
    parser.add_argument("--background-min", type=float, default=0.0)
    parser.add_argument("--background-max", type=float, default=0.30)
    parser.add_argument("--low-freq-noise-min", type=float, default=0.08)
    parser.add_argument("--low-freq-noise-max", type=float, default=0.30)
    parser.add_argument("--read-noise-min", type=float, default=0.03)
    parser.add_argument("--read-noise-max", type=float, default=0.14)
    parser.add_argument("--poisson-counts-min", type=float, default=50.0)
    parser.add_argument("--poisson-counts-max", type=float, default=25000.0)
    parser.add_argument("--blur-sigma-min", type=float, default=0.3)
    parser.add_argument("--blur-sigma-max", type=float, default=1.1)
    parser.add_argument("--edge-padding", type=int, default=24)
    parser.add_argument("--periodic-spacing-min", type=float, default=12.5)
    parser.add_argument("--periodic-spacing-max", type=float, default=16.7)
    parser.add_argument("--periodic-jitter-min", type=float, default=0.0)
    parser.add_argument("--periodic-jitter-max", type=float, default=0.25)
    parser.add_argument("--periodic-vacancy-min", type=float, default=0.0)
    parser.add_argument("--periodic-vacancy-max", type=float, default=0.03)
    parser.add_argument("--periodic-min-atoms", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-filters", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--threshold-rel", type=float, default=0.35)
    parser.add_argument("--real-threshold-rel", type=float, default=0.65)
    parser.add_argument("--match-distance", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--offset-plot-range", type=float, default=0.6)
    parser.add_argument("--offset-bins", type=int, default=181)
    parser.add_argument("--gallery-sample-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--real-crop-size", type=int, default=256)
    parser.add_argument("--real-num-crops", type=int, default=3)
    return parser.parse_args()


def make_random_config(args: argparse.Namespace) -> RandomGaussianConfig:
    return RandomGaussianConfig(
        image_shape=(args.height, args.width),
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        min_separation=args.min_separation,
        min_separation_range=(args.min_separation_range_min, args.min_separation_range_max),
        sigma_range=(args.sigma_min, args.sigma_max),
        background_range=(args.background_min, args.background_max),
        low_frequency_noise_range=(args.low_freq_noise_min, args.low_freq_noise_max),
        read_noise_std_range=(args.read_noise_min, args.read_noise_max),
        poisson_counts_range=(args.poisson_counts_min, args.poisson_counts_max),
        blur_sigma_range=(args.blur_sigma_min, args.blur_sigma_max),
        edge_padding=args.edge_padding,
    )


def make_periodic_config(
    args: argparse.Namespace,
    random_config: RandomGaussianConfig,
    lattice_type: str,
) -> PeriodicLatticeConfig:
    return PeriodicLatticeConfig(
        image_shape=(args.height, args.width),
        lattice_type=lattice_type,
        lattice_spacing_range=(args.periodic_spacing_min, args.periodic_spacing_max),
        jitter_std_range=(args.periodic_jitter_min, args.periodic_jitter_max),
        vacancy_fraction_range=(args.periodic_vacancy_min, args.periodic_vacancy_max),
        sigma_range=random_config.sigma_range,
        intensity_range=random_config.intensity_range,
        target_sigma=random_config.target_sigma,
        background_range=random_config.background_range,
        gradient_range=random_config.gradient_range,
        low_frequency_noise_range=random_config.low_frequency_noise_range,
        low_frequency_sigma_fraction_range=random_config.low_frequency_sigma_fraction_range,
        read_noise_std_range=random_config.read_noise_std_range,
        poisson_counts_range=random_config.poisson_counts_range,
        blur_sigma_range=random_config.blur_sigma_range,
        normalize_input=random_config.normalize_input,
        clamp_target=random_config.clamp_target,
        min_atoms=args.periodic_min_atoms,
        edge_padding=args.edge_padding,
    )


def make_structure_configs(
    args: argparse.Namespace,
    random_config: RandomGaussianConfig,
) -> Dict[str, AseStructureProjectionConfig]:
    common = dict(
        image_shape=(args.structure_height, args.structure_width),
        pixel_size_angstrom=args.structure_pixel_size_angstrom,
        rotation_range=(0.0, 180.0),
        position_jitter_std_range=(args.structure_jitter_min, args.structure_jitter_max),
        sigma_range=random_config.sigma_range,
        intensity_range=random_config.intensity_range,
        target_sigma=random_config.target_sigma,
        background_range=random_config.background_range,
        gradient_range=random_config.gradient_range,
        low_frequency_noise_range=random_config.low_frequency_noise_range,
        low_frequency_sigma_fraction_range=random_config.low_frequency_sigma_fraction_range,
        read_noise_std_range=random_config.read_noise_std_range,
        poisson_counts_range=random_config.poisson_counts_range,
        blur_sigma_range=random_config.blur_sigma_range,
        normalize_input=random_config.normalize_input,
        clamp_target=random_config.clamp_target,
        edge_padding=args.edge_padding,
    )
    return {
        "graphene": AseStructureProjectionConfig(structure_name="graphene", species_intensity_power=1.2, **common),
        "ws2": AseStructureProjectionConfig(structure_name="ws2", species_intensity_power=1.6, **common),
        "sto": AseStructureProjectionConfig(structure_name="sto", species_intensity_power=1.5, **common),
    }


def save_prediction_gallery(
    model: torch.nn.Module,
    test_loaders: Dict[str, object],
    case_names: Sequence[str],
    sample_indices: Sequence[int],
    device: torch.device,
    threshold_rel: float,
    output_path: Path,
) -> Path:
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for case_name in case_names:
            dataset = test_loaders[case_name].dataset
            for sample_index in sample_indices:
                image, target, metadata = dataset[int(sample_index)]
                prediction = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                result = evaluate_heatmap_localization(
                    prediction,
                    metadata["coordinates"],
                    threshold_rel=threshold_rel,
                )
                rmse_text = "n/a" if np.isnan(result["rmse"]) else f"{result['rmse']:.3f} px"
                rows.append(
                    {
                        "label": case_name.replace("_", " ").title(),
                        "metrics_text": (
                            f"sample={sample_index}  atoms={len(metadata['coordinates'])}  "
                            f"pred={len(result['predicted_coordinates'])}  "
                            f"F1={result['f1']:.3f}  RMSE={rmse_text}"
                        ),
                        "image": image[0].numpy(),
                        "target": target[0].numpy(),
                        "prediction": prediction,
                        "true_coords": metadata["coordinates"],
                        "predicted_coords": result["predicted_coordinates"],
                    }
                )
    return build_prediction_gallery(rows, output_path)


def train_and_evaluate(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.device]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = resolve_torch_device(args.device, verbose=True)

    random_config = make_random_config(args)
    cubic_config = make_periodic_config(args, random_config, "cubic")
    hexagonal_config = make_periodic_config(args, random_config, "hexagonal")
    structure_configs = make_structure_configs(args, random_config)
    periodic_cases = ["random", "cubic", "hexagonal"]
    structure_cases = ["graphene", "ws2", "sto"] if args.structure_test_samples > 0 else []
    all_cases = periodic_cases + structure_cases

    print(
        "Starting U-Net workflow with "
        f"device={device}, epochs={args.epochs}, "
        f"train/val/random_test/periodic_test/structure_test="
        f"{args.train_samples}/{args.val_samples}/{args.random_test_samples}/"
        f"{args.periodic_test_samples}/{args.structure_test_samples}, "
        f"image={args.height}x{args.width}, output={args.output_dir}",
        flush=True,
    )
    print(
        "Random training config: "
        f"min_atoms={random_config.min_atoms}, max_atoms={random_config.max_atoms}, "
        f"min_sep={random_config.min_separation}, min_sep_range={random_config.min_separation_range}, "
        f"sigma_range={random_config.sigma_range}, edge_padding={random_config.edge_padding}",
        flush=True,
    )
    print(f"U-Net channel widths: {args.num_filters}", flush=True)

    train_loader, val_loader, test_loaders = build_generalization_dataloaders(
        random_config=random_config,
        cubic_config=cubic_config,
        hexagonal_config=hexagonal_config,
        structure_configs=structure_configs if structure_cases else None,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        random_test_samples=args.random_test_samples,
        periodic_test_samples=args.periodic_test_samples,
        structure_test_samples=args.structure_test_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model_name = "unet"
    run_dir = args.output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_unet(
        input_channels=1,
        num_classes=1,
        num_filters=args.num_filters,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = CombinedGaussianLoss(from_logits=True)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_name=str(run_dir / model_name),
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    checkpoint_path = run_dir / f"{model_name}_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(
        f"Loaded best checkpoint from epoch {checkpoint.get('best_epoch', -1) + 1} "
        f"with validation loss {checkpoint.get('best_val_loss', float('nan')):.4f}",
        flush=True,
    )

    rows: List[Dict[str, float]] = []
    for case_name in all_cases:
        summary = evaluate_model_localization(
            model=model,
            dataloader=test_loaders[case_name],
            device=device,
            channel=0,
            apply_sigmoid=True,
            threshold_rel=args.threshold_rel,
            match_distance=args.match_distance,
        )
        row = {
            "model": model_name,
            "test_case": case_name,
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

    save_metrics_table(rows, args.output_dir)
    plot_case_summaries(rows, args.output_dir, case_order=all_cases)
    plot_generalization_summary(
        [row for row in rows if row["test_case"] in periodic_cases],
        args.output_dir,
        case_order=periodic_cases,
        model_order=[model_name],
        filename="generalization_summary.png",
        title="Blob-Net Generalization Summary",
    )
    if structure_cases:
        plot_generalization_summary(
            [row for row in rows if row["test_case"] in structure_cases],
            args.output_dir,
            case_order=structure_cases,
            model_order=[model_name],
            filename="generalization_summary_structures.png",
            title="Blob-Net Structure Generalization Summary",
        )
    plot_loss_curves([model_name], args.output_dir)

    save_prediction_gallery(
        model=model,
        test_loaders=test_loaders,
        case_names=periodic_cases,
        sample_indices=args.gallery_sample_indices,
        device=device,
        threshold_rel=args.threshold_rel,
        output_path=args.output_dir / "prediction_gallery_periodic.png",
    )
    if structure_cases:
        save_prediction_gallery(
            model=model,
            test_loaders=test_loaders,
            case_names=structure_cases,
            sample_indices=args.gallery_sample_indices,
            device=device,
            threshold_rel=args.threshold_rel,
            output_path=args.output_dir / "prediction_gallery_structures.png",
        )

    periodic_offsets = {
        case_name: collect_matched_offsets(
            model=model,
            dataloader=test_loaders[case_name],
            device=device,
            threshold_rel=args.threshold_rel,
            match_distance=args.match_distance,
        )
        for case_name in periodic_cases
    }
    plot_offset_cloud(
        case_results=periodic_offsets,
        output_path=args.output_dir / "unet_offset_cloud.png",
        model_name="unet",
        plot_range=args.offset_plot_range,
        bins=args.offset_bins,
        point_alpha=0.025,
        point_size=10.0,
        dpi=220,
    )
    save_offsets(periodic_offsets, args.output_dir / "unet_offset_cloud_offsets.npz")

    if structure_cases:
        structure_offsets = {
            case_name: collect_matched_offsets(
                model=model,
                dataloader=test_loaders[case_name],
                device=device,
                threshold_rel=args.threshold_rel,
                match_distance=args.match_distance,
            )
            for case_name in structure_cases
        }
        plot_offset_cloud(
            case_results=structure_offsets,
            output_path=args.output_dir / "unet_offset_cloud_structures.png",
            model_name="unet (structures)",
            plot_range=args.offset_plot_range,
            bins=args.offset_bins,
            point_alpha=0.025,
            point_size=10.0,
            dpi=220,
        )
        save_offsets(structure_offsets, args.output_dir / "unet_offset_cloud_structures_offsets.npz")

    return model, device


def run_real_image_inference(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> None:
    real_output_dir = args.output_dir / "real_image"
    real_output_dir.mkdir(parents=True, exist_ok=True)

    raw_image, metadata = load_velox_emd_image(args.real_data_path)
    pixel_size_angstrom = get_real_image_pixel_size_angstrom(metadata)
    image_variants = preprocess_real_image_variants(raw_image)
    heatmaps: Dict[str, np.ndarray] = {}
    coordinates: Dict[str, np.ndarray] = {}
    summary_rows = []

    for variant_name, image in image_variants.items():
        print(f"Predicting real image variant={variant_name}", flush=True)
        heatmap = predict_heatmap_tiled(
            model=model,
            image=image,
            device=device,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            batch_size=min(4, args.batch_size),
        )
        coords = extract_subpixel_peak_positions(heatmap, threshold_rel=args.real_threshold_rel)
        heatmaps[variant_name] = heatmap
        coordinates[variant_name] = coords
        summary_rows.append(
            {
                "variant": variant_name,
                "detections": int(len(coords)),
                "heatmap_mean": float(heatmap.mean()),
                "heatmap_max": float(heatmap.max()),
            }
        )

    overview_path = save_real_image_overview(
        output_path=real_output_dir / "real_image_overview.png",
        image_variants=image_variants,
        heatmaps=heatmaps,
        coordinates=coordinates,
        pixel_size_angstrom=pixel_size_angstrom,
        threshold_rel=args.real_threshold_rel,
    )
    preferred_variant = "flatfield_normalized" if "flatfield_normalized" in image_variants else next(iter(image_variants))
    crop_boxes = select_informative_crops(
        image_variants[preferred_variant],
        crop_size=args.real_crop_size,
        num_crops=args.real_num_crops,
    )
    crop_gallery_path = save_real_crop_gallery(
        output_path=real_output_dir / "real_image_crops.png",
        variant_name=preferred_variant,
        image=image_variants[preferred_variant],
        heatmap=heatmaps[preferred_variant],
        coordinates=coordinates[preferred_variant],
        crop_boxes=crop_boxes,
    )

    summary = {
        "real_data_path": str(args.real_data_path),
        "shape": list(raw_image.shape),
        "pixel_size_angstrom": pixel_size_angstrom,
        "real_threshold_rel": args.real_threshold_rel,
        "variants": summary_rows,
        "preferred_variant": preferred_variant,
        "crop_boxes_yxyx": crop_boxes,
    }
    summary_path = real_output_dir / "real_image_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved real image overview to {overview_path}", flush=True)
    print(f"Saved real image crop gallery to {crop_gallery_path}", flush=True)
    print(f"Saved real image summary to {summary_path}", flush=True)


def main() -> int:
    args = parse_args()
    model, device = train_and_evaluate(args)
    run_real_image_inference(args, model, device)
    print(f"Finished workflow in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
