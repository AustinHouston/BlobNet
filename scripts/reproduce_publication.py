"""Audit or regenerate the manuscript and SI from a clean repository checkout."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from blobnet.experimental import open_experimental_image

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts/manuscript_models"
DATA = ROOT / "experimental_data"
MANUSCRIPT = ROOT / "publication/manuscript"
MANUSCRIPT_OUTPUT = ROOT / "outputs/publication_manuscript_figures"
SI_OUTPUT = ROOT / "outputs/supplemental_information_20260913"
SI_DOCUMENT = SI_OUTPUT / "document"

EXPECTED_CHECKPOINTS = {
    "square": "7ca9e7beadff12a0b4bbafc4116029c257b101d950881e9fd5c6f14a2f32c3b5",
    "hexagonal": "9f03deaf71884a852aaf52250d4f3e0a29f90da2db9519687ab083facc67a3d0",
    "random": "2fa1cfd6caba4de9a3d5f1d5728d33266397c0b882d78d7d3dd600dadabfd041",
    "figure3_random": "7c4ef2d4e31445606cae11d5dfec3ddf57d68d8c1981953bba0137b8dc242ba1",
    "figure3_hexagonal": "70ede8a0a245c78bb98c9d1ea8672cc55f650e55d3a964357ff49be70e6e90e6",
}
EXPECTED_IMAGES = {
    "pristine_monolayer_MoS2.hf5",
    "Sigma3_coherent_twin_grain_boundary_FCC_Al.hf5",
    "high_angle_grain_boundary_monolayer_WS2.hf5",
    "Al72Ni11Co17_quasicrystal.hf5",
    "gold_implanted_in_TiO2.hf5",
}


def digest(path: Path) -> str:
    block = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            block.update(chunk)
    return block.hexdigest()


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def audit() -> None:
    errors = []
    for family, expected in EXPECTED_CHECKPOINTS.items():
        path = MODELS / family / "unet_best.pth"
        if not path.is_file():
            errors.append(f"missing checkpoint: {path.relative_to(ROOT)}")
        elif digest(path) != expected:
            errors.append(f"checkpoint hash mismatch: {path.relative_to(ROOT)}")
        elif path.stat().st_size >= 100_000_000:
            errors.append(f"checkpoint exceeds GitHub's 100 MB limit: {path.relative_to(ROOT)}")
    actual_images = {path.name for path in DATA.glob("*.hf5")}
    if actual_images != EXPECTED_IMAGES:
        errors.append(f"experimental NSID set differs: {sorted(actual_images)}")
    for path in DATA.glob("*.hf5"):
        if path.stat().st_size >= 100_000_000:
            errors.append(f"image exceeds GitHub's 100 MB limit: {path.relative_to(ROOT)}")
        try:
            image, metadata = open_experimental_image(path)
            if image.size == 0 or float(metadata["pixel_size_nm"]) <= 0:
                raise ValueError("empty image or invalid pixel size")
        except (KeyError, OSError, TypeError, ValueError) as error:
            errors.append(f"invalid publication NSID file: {path.relative_to(ROOT)} ({error})")
    for relative in ["manuscript.tex", "references.bib", "sections/manuscript.tex"]:
        if not (MANUSCRIPT / relative).is_file():
            errors.append(f"missing manuscript source: publication/manuscript/{relative}")
    if errors:
        raise RuntimeError("Publication audit failed:\n- " + "\n- ".join(errors))
    print("Publication audit passed: inputs, hashes, schemas, and GitHub file-size limits are valid.")


def reproduce_manuscript(device: str, compile_latex: bool) -> None:
    run(sys.executable, "-m", "scripts.measure_experimental_features",
        "--data-dir", str(DATA), "--output-dir", str(ROOT / "outputs/experimental_feature_measurements_local"))
    run(sys.executable, "-m", "scripts.make_manuscript_figures", "all",
        "--device", device, "--output-dir", str(MANUSCRIPT_OUTPUT),
        "--sweep-csv", str(MANUSCRIPT_OUTPUT / "pixel_size_metrics.csv"), "--regenerate-sweep")
    mappings = {
        "figure1_training_geometry_generalization.png": "fig-1.png",
        "figure2_edge_lattice_model_diagnostics.png": "fig-2.png",
        "figure3_experimental_haadf_outputs.png": "fig-3.png",
        "figure4_scale_spacing_robustness.png": "fig-4.png",
    }
    for generated, manuscript_name in mappings.items():
        shutil.copy2(MANUSCRIPT_OUTPUT / generated, MANUSCRIPT / "figures" / manuscript_name)
    if compile_latex:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "manuscript.tex"], cwd=MANUSCRIPT, check=True)


def reproduce_si(device: str, compile_latex: bool) -> None:
    run(sys.executable, "-m", "scripts.make_supplemental_figures", "--figure", "all",
        "--device", device, "--output-dir", str(SI_OUTPUT), "--document-dir", str(SI_DOCUMENT),
        "--model-dir", str(MODELS))
    if compile_latex:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "supplementary_information.tex"], cwd=SI_DOCUMENT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["audit", "manuscript", "si", "all"], default="audit")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--compile-latex", action="store_true")
    args = parser.parse_args()
    audit()
    if args.target in {"manuscript", "all"}:
        reproduce_manuscript(args.device, args.compile_latex)
    if args.target in {"si", "all"}:
        reproduce_si(args.device, args.compile_latex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
