"""Reproducible, independent SI experiments. Run one figure or all in S-number order.

All thresholds used for confirmatory comparisons are fixed or validation-selected.
Original checkpoints are read-only. New training, data and predictions are isolated.
Missing historical provenance is reported, never silently reconstructed as fact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml
import h5py
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_laplace, zoom
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset

from blobnet import synthetic as syn
from blobnet.metrics import extract_subpixel_peak_positions, match_coordinate_sets
from blobnet.networks import build_unet
from blobnet.loss_func import CombinedGaussianLoss
from scripts import make_manuscript_figures as mainfig
from scripts.train_unet import train_model

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('square', 'hexagonal', 'random')
LABELS = {'square': 'Square-Net', 'hexagonal': 'Hex-Net', 'random': 'Blob-Net'}
COLORS = {'square': '#3167a5', 'hexagonal': '#c77122', 'random': '#23835e'}
THRESHOLDS = np.array([.1, .2, .3, .35, .45, .55, .65, .73, .785, .85, .9])
TITLES = {
 1: 'Synthetic image formation and target construction',
 2: 'Dataset provenance and realized distributions',
 3: 'Training histories and checkpoint selection',
 4: 'Paired cross-geometry generalization',
 5: 'Detection and matching sensitivity',
 6: 'Synthetic edges and boundary-localized errors',
 7: 'Controlled positional disorder',
 8: 'Experimental preprocessing and disagreement atlas',
 9: 'Scale transfer under explicit evaluation protocols',
 10: 'Feature width, spacing, and receptive field',
 11: 'Noise robustness and a classical baseline',
 12: 'Training-distribution and random-seed controls',
 13: 'Tiled inference and Gaussian coordinate refinement',
}


def serial(value):
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    if isinstance(value, np.ndarray): return serial(value.tolist())
    if isinstance(value, np.generic): return serial(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(serial(value), sort_keys=True).encode()).hexdigest()[:20]


def evaluate(prediction, truth, threshold=.35, radius=3., separation=3, window=5, border=0):
    positions = extract_subpixel_peak_positions(prediction, threshold_rel=threshold,
                      min_distance=separation, window_size=window)
    truth = np.asarray(truth).reshape(-1, 2)
    if border:
        def inside(p):
            return np.all((p >= border) & (p < np.array(prediction.shape) - border), axis=1)
        positions, truth = positions[inside(positions)], truth[inside(truth)]
    result = match_coordinate_sets(positions, truth, max_distance=radius)
    result['positions'], result['truth'] = positions, truth
    tp, fp, fn = (result[k] for k in ('tp', 'fp', 'fn'))
    result.update(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                  f1=2 * tp / max(2 * tp + fp + fn, 1),
                  rmse=float(np.sqrt(np.mean(result['errors'] ** 2))) if tp else np.nan)
    return result


def compact(result):
    return {k: result[k] for k in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'rmse')}


def pooled(results):
    tp, fp, fn = [sum(r[k] for r in results) for k in ('tp', 'fp', 'fn')]
    errors = np.concatenate([r['errors'] for r in results])
    return dict(tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                rmse=float(np.sqrt(np.mean(errors ** 2))) if len(errors) else np.nan)


def bootstrap_paired(a, b, seed=19, draws=2000):
    d = np.asarray(a) - np.asarray(b)
    if len(d) == 0: raise ValueError('No paired images')
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    return [float(d.mean()), *np.quantile(means, [.025, .975]).tolist()]


def select_threshold(predictions, records):
    # Calibration callers supply only the independently generated validation set.
    scores = [pooled([evaluate(p, r['coordinates'], float(t)) for p, r in zip(predictions, records)])['f1']
              for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def show(ax, array, title='', points=None):
    ax.imshow(array, cmap='gray', interpolation='nearest')
    if points is not None and len(points):
        ax.scatter(points[:, 1], points[:, 0], s=8, facecolors='none', edgecolors='#e57527', linewidths=.5)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def basic_axes(ax, xlabel, ylabel):
    ax.set(xlabel=xlabel, ylabel=ylabel); ax.grid(alpha=.2)


def make_gold_tio2_figure(study):
    """Generate the four-panel Au-in-TiO2 figure used at the start of the SI."""
    image_path = ROOT / 'experimental_data/gold_implanted_in_TiO2.h5'
    with h5py.File(image_path, 'r') as handle:
        full_image = np.asarray(handle['image'], dtype=np.float32)
        pixel_size_nm = float(handle['image'].attrs['pixel_size_nm'])
    raw = full_image[256:768, 256:768]
    field = gaussian_filter(raw, 5, mode='reflect')
    field_median = float(np.median(field))
    corrected = raw / np.maximum(field, 0.05 * field_median) * field_median
    normalized = mainfig._normalize_image(corrected, low=1.0, high=99.8)
    processed = mainfig._normalize_image(
        gaussian_filter(normalized, 1, mode='reflect')
        - gaussian_filter(normalized, 10, mode='reflect'),
        low=1.0, high=99.8,
    )
    inference = mainfig._normalize_image(
        mainfig._interpolate_image(processed, (301, 301)), low=1.0, high=99.8
    )
    raw_display = mainfig._normalize_image(raw, low=1.0, high=99.8)

    models = [
        ('Blob-Net', study.args.model_dir / 'random/unet_best.pth', '#E69F00', 36),
        ('Hex-Net', study.args.model_dir / 'hexagonal/unet_best.pth', '#00BFFF', 18),
        ('Square-Net', study.args.model_dir / 'square/unet_best.pth', '#CC33CC', 6),
    ]
    positions = {}
    for name, checkpoint, _color, _size in models:
        model = mainfig._load_blobnet_model(checkpoint, study.device, [32, 64, 128, 256], 0.2)
        prediction = mainfig._predict_tiled(model, inference, study.device, 256, 64, 4)
        coordinates = extract_subpixel_peak_positions(
            prediction, threshold_rel=0.10, min_distance=3, window_size=5
        )
        positions[name] = np.asarray(coordinates, dtype=np.float32) * (511.0 / 300.0)

    windows = [(0, 512, 0, 512), (0, 512, 0, 512),
               (180, 340, 100, 260), (352, 512, 220, 380)]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.5), constrained_layout=True)
    for panel, (axis, window) in enumerate(zip(axes.flat, windows)):
        y0, y1, x0, x1 = window
        image = raw_display if panel == 0 else processed
        axis.imshow(image[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
        if panel:
            for name, _checkpoint, color, size in models:
                coordinates = positions[name]
                inside = ((coordinates[:, 0] >= y0) & (coordinates[:, 0] < y1)
                          & (coordinates[:, 1] >= x0) & (coordinates[:, 1] < x1))
                plotted = coordinates[inside] - np.array([y0, x0])
                scatter = axis.scatter(
                    plotted[:, 1], plotted[:, 0], s=size * (5 if panel > 1 else 1),
                    facecolors='none', edgecolors=color,
                    linewidths=1 if panel > 1 else 0.7,
                )
                scatter.set_path_effects([
                    patheffects.Stroke(
                        linewidth=1.7 if panel > 1 else 1.15,
                        foreground='black', alpha=0.65,
                    ),
                    patheffects.Normal(),
                ])
        axis.set(xticks=[], yticks=[], xlim=(-0.5, x1-x0-0.5), ylim=(y1-y0-0.5, -0.5))
        axis.set_title(chr(ord('a') + panel), loc='left', fontsize=20, fontweight='bold')
        mainfig._add_physical_scale_bar(
            axis, (y1-y0, x1-x0), pixel_size_nm,
            length_nm=1 if panel < 2 else 0.5, linewidth=3,
        )
    for label, window, color in [('c', windows[2], 'white'), ('d', windows[3], '#FF66CC')]:
        y0, y1, x0, x1 = window
        axes.flat[1].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                          color=color, linewidth=1.3)
        axes.flat[1].text(x0+4, y0+17, label, color=color, fontsize=13, fontweight='bold')
    fig.legend(
        handles=[Line2D([], [], linestyle='none', marker='o', markerfacecolor='none',
                        markeredgecolor=color, markersize=np.sqrt(size)+2, label=name)
                 for name, _checkpoint, color, size in models],
        loc='outside upper center', ncol=3, frameon=False, fontsize=12,
    )

    stem = 'fig-S01-au-tio2'
    for directory in (study.out / 'figures', study.doc / 'figures'):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f'{stem}.png', dpi=300)
    plt.close(fig)
    provenance = {
        'figure': 'S1',
        'image': str(image_path.relative_to(ROOT)),
        'image_sha256': digest(image_path),
        'image_crop_yx': [256, 256, 512, 512],
        'field_sigma_source_px': 5,
        'dog_sigmas_source_px': [1, 10],
        'normalization_percentiles': [1, 99.8],
        'inference_shape': [301, 301],
        'threshold_rel': 0.10,
        'model_sha256': {name: digest(checkpoint) for name, checkpoint, _color, _size in models},
        'prediction_counts': {name: len(value) for name, value in positions.items()},
        'layout': '2x2',
    }
    arrays = {
        'background': processed,
        **{name.replace('-', '_')+'_coordinates_yx': value
           for name, value in positions.items()},
    }
    output_data = study.out / 'data'
    output_data.mkdir(parents=True, exist_ok=True)
    dump(output_data / 'gold-au-tio2.json', provenance)
    np.savez_compressed(output_data / 'gold-au-tio2.npz', **arrays)
    document_data = study.doc / 'source_data'
    document_data.mkdir(parents=True, exist_ok=True)
    dump(document_data / f'{stem}.json', provenance)
    np.savez_compressed(document_data / f'{stem}.npz', **arrays)
    section = ROOT / 'supplemental/BlobNet_SI/sections/utkarsh_gold_tio2.tex'
    if section.is_file():
        shutil.copy2(section, study.doc / 'sections/utkarsh_gold_tio2.tex')
    return study.doc / 'figures' / f'{stem}.pdf'


class Study:
    def __init__(self, args):
        self.args = args
        self.out = args.output_dir.resolve(); self.doc = args.document_dir.resolve()
        for p in (self.out / 'figures', self.out / 'data', self.out / 'cache', self.doc / 'figures', self.doc / 'sections'):
            p.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('mps' if args.device == 'auto' and torch.backends.mps.is_available()
                                   else 'cpu' if args.device == 'auto' else args.device)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.models = {}; self.configs = {}; self.results = {}
        self.checkpoints = {}
        for family in FAMILIES:
            cp = args.model_dir / family / 'unet_best.pth'
            self.checkpoints[family] = {'path': str(cp.resolve()), 'sha256': digest(cp)}
            cfg = yaml.safe_load((ROOT / f'configs/dataset_configs/{family}.yaml').read_text())
            cls = syn.RandomAtomImageConfig if family == 'random' else syn.PeriodicLatticeConfig
            self.configs[family] = cls(**cfg['parameters'])
        source_paths = [Path(__file__), Path(syn.__file__), Path(mainfig.__file__), ROOT / 'blobnet/metrics.py', ROOT / 'blobnet/networks.py']
        self.source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
        # Raw predictions are keyed by input bytes + checkpoint bytes, so unrelated
        # plotting changes do not invalidate expensive inference caches.
        manifest = dict(args=vars(args), device=str(self.device), checkpoints=self.checkpoints,
                        source_sha256=self.source_hashes, torch_version=torch.__version__,
                        numpy_version=np.__version__, package_versions={name: importlib.metadata.version(name) for name in ('numpy','scipy','matplotlib','torch','ase','h5py')}, configs={k: asdict(v) for k, v in self.configs.items()},
                        evaluation_data='New independent draws from explicitly saved current configurations; not recovered original training data.',
                        calibration_seed_start=7_000_000, test_seed_start=8_000_000)
        if args.figure != 'document' or not (self.out / 'run_manifest.json').exists():
            dump(self.out / 'run_manifest.json', manifest)
        dump(self.out / 'invocations' / f'{fingerprint(manifest)}.json', manifest)
        for source_path in source_paths:
            source_digest = digest(source_path)
            destination = self.out / 'source_snapshots' / source_digest / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
        plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                             'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'pdf.fonttype': 42})

    def model(self, family):
        if family not in self.models:
            self.models[family] = mainfig._load_blobnet_model(Path(self.checkpoints[family]['path']), self.device, [32,64,128,256], .2)
        return self.models[family]

    def predict(self, family, image):
        image = np.asarray(image, np.float32)
        key = fingerprint([self.checkpoints[family]['sha256'], hashlib.sha256(image.tobytes()).hexdigest(), image.shape])
        path = self.out / 'cache' / f'pred_{key}.npy'
        if path.exists(): return np.load(path)
        h, w = image.shape
        padded = np.pad(image, ((0, (-h) % 8), (0, (-w) % 8)), mode='reflect')
        pred = mainfig._predict_array(self.model(family), padded, self.device)[:h, :w]
        np.save(path, pred)
        return pred

    def record(self, family, split, index, config=None):
        config = config or self.configs[family]
        seed = {'validation': 7_000_000, 'test': 8_000_000, 'training': 6_000_000}[split] + index
        key = fingerprint([asdict(config), seed, self.source_hashes['blobnet/synthetic.py']])
        path = self.out / 'cache' / f'record_{key}.npz'
        if path.exists():
            with np.load(path) as f: return {k: f[k] for k in f.files}
        rec = syn.generate_atom_image(config, np.random.default_rng(seed))
        rec = {k: rec[k] for k in ('image','target','coordinates','sigmas','intensities','count_scale','total_counts')}
        np.savez_compressed(path, **rec)
        return rec

    def corpus(self, family, split, n=None):
        n = n or (self.args.validation_samples if split == 'validation' else self.args.samples)
        return [self.record(family, split, i) for i in range(n)]

    def save(self, number, fig, data, caption, suffix=''):
        stem = f'fig-S{number:02d}{suffix}'
        fig.savefig(self.out / 'figures' / f'{stem}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
        payload = dict(title=TITLES[number], caption=caption, data=data, source_sha256=self.source_hashes)
        dump(self.out / 'data' / f'{stem}.json', payload)
        self.results[stem] = payload
        shutil.copy2(self.out / 'figures' / f'{stem}.png', self.doc / 'figures')
        print(f'Saved {stem}: {caption[:100]}', flush=True)

    def s1(self):
        data = {}
        for row, family in enumerate(FAMILIES):
            fig, axes = plt.subplots(3, 3, figsize=(8.5,7.5), constrained_layout=True)
            config = replace(self.configs[family], image_shape=(128,128))
            if family == 'random': config = replace(config, min_atoms=65,max_atoms=85)
            else: config = replace(config, min_atoms=20)
            rng = np.random.default_rng(101 + row)
            cloud = syn.point_cloud_from_config(config, rng)
            rec = syn.render_atom_image(cloud.coordinates, config, rng, target_coordinates=cloud.target_coordinates,
                                        return_stages=True)
            for col, (name, array) in enumerate(rec['stages'].items()):
                show(axes.flat[col], array, name.replace('_',' '))
            show(axes.flat[8], rec['target'], 'target: max Gaussians', rec['coordinates'])
            fig.suptitle(family.capitalize(), fontsize=12)
            data[family] = dict(config=asdict(config), count_scale=rec['count_scale'], total_counts=rec['total_counts'],
                                visible=len(rec['coordinates']), rendered=len(rec['rendered_coordinates']))
            self.save(1, fig, data[family], f'{family.capitalize()} geometry: sequential snapshots from the production renderer with unchanged random draws. Each panel uses its own grayscale range to expose weak components. Inputs sum atomic Gaussians; targets take their pixelwise maximum. Off-frame support is rendered before cropping; only in-frame centers receive labels. Poisson scale is a peak-intensity scale, not image-total electron dose. Examples are 128-pixel explanatory crops with declared atom counts.', ['', '-b', '-c'][row])
        # A dedicated border/target profile panel makes the label convention testable.
        cfg = replace(self.configs['random'], image_shape=(48,48), target_sigma=2., edge_padding=16)
        coords = np.array([[24.,-1.],[24.,8.],[24.,12.]], np.float32)
        rec = syn.render_atom_image(coords,cfg,np.random.default_rng(51),target_coordinates=coords[1:],
                                   intensities=np.ones(3),sigmas=np.full(3,2.),return_stages=True)
        fig, ax = plt.subplots(1,3,figsize=(10,3),constrained_layout=True)
        show(ax[0], rec['stages']['atomic_contrast'], 'Off-frame atomic support', coords[1:])
        show(ax[1], rec['target'], 'Only in-frame targets', coords[1:])
        ax[2].plot(rec['stages']['atomic_contrast'][24],label='summed input')
        ax[2].plot(rec['target'][24],label='maximum target'); ax[2].legend()
        basic_axes(ax[2], 'x (px)', 'Intensity / target')
        self.save(1,fig,{'coordinates':coords},'Border and overlap conventions. The atom centered at x=-1 contributes image intensity but no target. Overlapping target peaks use a maximum rather than an intensity sum.', '-d')

    def s2(self):
        audit = {}; rows = []; occupancy = {}; examples = {}
        for family in FAMILIES:
            folder = ROOT / f'outputs/datasets/{family}_2026_06_24a'
            manifest_path = folder / 'dataset_manifest.yaml'
            manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else None
            audit[family] = dict(path=folder, manifest=manifest, actual_counts={}, sha256={}, exact_training_provenance='unverified')
            occupancy[family] = np.zeros((16,16))
            for split in ('train','val','test'):
                paths = sorted((folder / split).glob('*.npz'))
                audit[family]['actual_counts'][split] = len(paths)
                for j,p in enumerate(paths):
                    audit[family]['sha256'][str(p.relative_to(folder))] = digest(p)
                    with np.load(p) as z:
                        xy=z['coordinates']; im=z['image']; target=z['target']
                        spacing=cKDTree(xy).query(xy,k=2)[0][:,1] if len(xy)>1 else np.array([np.nan])
                        rows.append(dict(family=family,split=split,index=j,count=len(xy),spacing=float(np.median(spacing)),
                          sigma=float(np.mean(z['sigmas'])), intensity=float(np.mean(z['intensities'])),
                          foreground=float(np.mean(target>.1)), image_std=float(im.std()),
                          background_std=float(im[target<.01].std()) if np.any(target<.01) else np.nan,
                          counts=int(z['total_counts']) if 'total_counts' in z else None))
                        occupancy[family]+=np.histogram2d(xy[:,0],xy[:,1],bins=16,range=[[0,im.shape[0]],[0,im.shape[1]]])[0]
                        examples.setdefault(family,(im.copy(),xy.copy()))
        # Report exact duplicates across splits using image/coordinate payloads,
        # not compressed-file hashes (metadata compression can differ).
        hashes={}; duplicates=[]
        for family in FAMILIES:
            folder=ROOT/f'outputs/datasets/{family}_2026_06_24a'
            for split in ('train','val','test'):
                for p in sorted((folder/split).glob('*.npz')):
                    with np.load(p) as z:
                        h=hashlib.sha256(z['image'].tobytes()+z['coordinates'].tobytes()).hexdigest()
                    if h in hashes and hashes[h][1]!=split: duplicates.append([hashes[h], [family,split,str(p)]])
                    hashes[h]=[family,split,str(p)]
        fig,axes=plt.subplots(2,4,figsize=(13,6),constrained_layout=True)
        fields=['count','spacing','sigma','intensity','foreground','image_std','background_std','counts']
        for ax,field in zip(axes.flat,fields):
            positions=[]; values=[]; names=[]
            for fi,family in enumerate(FAMILIES):
                for si,split in enumerate(('train','val','test')):
                    vals=[r[field] for r in rows if r['family']==family and r['split']==split and r[field] is not None]
                    if vals:
                        bp=ax.boxplot(vals,positions=[fi*4+si],widths=.65,patch_artist=True,showfliers=False)
                        bp['boxes'][0].set_facecolor(COLORS[family]); positions.append(fi*4+si); names.append(split)
            ax.set_xticks([1,5,9],['Square','Hex','Random']); ax.set_title(field.replace('_',' ')); ax.grid(axis='y',alpha=.2)
        self.save(2,fig,dict(audit=audit,rows=rows,cross_split_exact_duplicates=duplicates),
          'Realized distributions of every locally available saved sample; within each family, boxes are train/validation/test from left to right. Original data generation and training took place on another computer. These files are a local snapshot, not the original manuscript training datasets. Split counts and file hashes are supplied in the source data. Absence of exact duplicates does not prove independence of generative processes.')
        fig,axes=plt.subplots(3,3,figsize=(10,9),constrained_layout=True)
        for row,family in enumerate(FAMILIES):
            im,xy=examples[family]
            show(axes[row,0],im,f'{family}: example')
            axes[row,1].imshow(occupancy[family]/max(occupancy[family].sum(),1),cmap='viridis'); axes[row,1].set_title('Normalized spatial occupancy')
            axes[row,1].set_xticks([]); axes[row,1].set_yticks([])
            field=np.histogram2d(xy[:,0],xy[:,1],bins=128,range=[[0,im.shape[0]],[0,im.shape[1]]])[0]
            field-=field.mean(); spec=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(field)))**2)
            axes[row,2].imshow(spec,cmap='magma'); axes[row,2].set_title('Coordinate power spectrum'); axes[row,2].set_xticks([]); axes[row,2].set_yticks([])
        self.save(2,fig,{'occupancy':occupancy},'Spatial coverage and coordinate spectra reveal geometry and sampling correlations. Spectra are illustrative individual realizations; occupancy aggregates the audited local files.', '-b')

    def s3(self):
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True); records={}
        for ax,family in zip(axes,FAMILIES):
            p=self.args.model_dir/family
            rows=list(csv.DictReader((p/'loss_history.csv').open()))
            metrics=json.loads((p/'training_metrics.json').read_text())
            config=yaml.safe_load((p/'resolved_config.yaml').read_text())
            epochs=[int(r['epoch']) for r in rows]
            ax.plot(epochs,[float(r['train_loss']) for r in rows],label='Training',color=COLORS[family])
            ax.plot(epochs,[float(r['val_loss']) for r in rows],'--',label='Validation',color='#393939')
            ax.axvline(metrics['best_epoch'],color='#999999',ls=':'); ax.set_title(LABELS[family])
            basic_axes(ax,'Epoch','Heatmap loss'); ax.legend(fontsize=8)
            records[family]=dict(metrics=metrics,config=config,checkpoint=self.checkpoints[family],history=rows)
        self.save(3,fig,records,'Recovered original manuscript-model histories. Vertical lines mark the saved best-validation epochs. All models have 1,927,841 trainable parameters. Absolute losses across geometries are not an accuracy ranking. Stopping patience counts epochs without an improvement exceeding 0.0005, whereas a checkpoint is saved on any new validation minimum. New repeated-seed controls are reported in S12.')

    def s4(self):
        fig,axes=plt.subplots(3,4,figsize=(13,9),constrained_layout=True); output={}
        for row,family in enumerate(FAMILIES):
            records=self.corpus(family,'test'); results={}
            for model in FAMILIES:
                results[model]=[evaluate(self.predict(model,r['image']),r['coordinates']) for r in records]
            output[family]={m:dict(pooled=pooled(rs),per_image=[compact(r) for r in rs]) for m,rs in results.items()}
            for col,metric in enumerate(('precision','recall','f1','rmse')):
                vals=[[r[metric] for r in results[m]] for m in FAMILIES]
                axes[row,col].boxplot(vals,tick_labels=['Square','Hex','Blob'],showfliers=False)
                axes[row,col].set_title(f'{family}: {metric}'); axes[row,col].grid(axis='y',alpha=.2)
            output[family]['paired_F1_difference']={m:bootstrap_paired([r['f1'] for r in results['random']],[r['f1'] for r in results[m]]) for m in ('square','hexagonal')}
            print('S4',family, {m: output[family][m]['pooled']['f1'] for m in FAMILIES},flush=True)
        self.save(4,fig,output,f'Paired evaluation of all three archived models on the same {self.args.samples} independently generated 512-pixel images per geometry. Threshold 0.35, 3-pixel matching radius, 3-pixel peak separation, 5-pixel centroid window. Boxes show per-image distributions; pooled metrics and 2,000-resample paired image-bootstrap intervals are saved in JSON. Test seeds begin at 8,000,000 and were not used for calibration. Current configurations define this new experiment; it does not retroactively identify the original training datasets.')

    def edge(self,family,index,validation=False):
        funcs={'mos2':mainfig._make_mos2_edge_record,'sto':mainfig._make_sto_edge_record,'graphene':mainfig._make_graphene_rattled_edge_record}
        return funcs[family]((512,512),(9_000_000 if validation else 10_000_000)+index,(1.15,2.65),total_counts_range=(64.,64.),quiet_background=True)

    def s5(self):
        fig,axes=plt.subplots(3,3,figsize=(12,9),constrained_layout=True); data={}
        for row,family in enumerate(('mos2','sto','graphene')):
            records=[self.edge(family,i) for i in range(self.args.replicates)]
            validation=[self.edge(family,i,True) for i in range(self.args.validation_samples)]
            data[family]={}
            for model in FAMILIES:
                preds=[self.predict(model,r['image']) for r in records]
                vp=[self.predict(model,r['image']) for r in validation]
                vscore=[pooled([evaluate(p,r['coordinates'],float(t),border=10) for p,r in zip(vp,validation)])['f1'] for t in THRESHOLDS]
                selected=float(THRESHOLDS[np.argmax(vscore)])
                metrics=[pooled([evaluate(p,r['coordinates'],float(t),border=10) for p,r in zip(preds,records)]) for t in THRESHOLDS]
                data[family][model]=dict(thresholds=THRESHOLDS,metrics=metrics,validation_threshold=selected,validation_f1=vscore)
                axes[row,0].plot(THRESHOLDS,[m['f1'] for m in metrics],label=LABELS[model],color=COLORS[model])
                axes[row,0].scatter([selected],[metrics[list(THRESHOLDS).index(selected)]['f1']],color=COLORS[model],s=35)
                axes[row,1].plot([m['recall'] for m in metrics],[m['precision'] for m in metrics],color=COLORS[model])
                radii=[1,2,3,4,5]
                radial=[pooled([evaluate(p,r['coordinates'],selected,radius=d,border=10) for p,r in zip(preds,records)]) for d in radii]
                axes[row,2].plot(radii,[m['f1'] for m in radial],color=COLORS[model]); data[family][model]['radius_sensitivity']=radial
            axes[row,0].axvline(mainfig.FIGURE2_TUNED_THRESHOLDS[{'mos2':'mos2_edge','sto':'srtio3_edge','graphene':'graphene_rattled_edge'}[family]],color='black',ls=':',label='Main Fig. 2 selected')
            for col,(x,y) in enumerate([('Relative threshold','F1'),('Recall','Precision'),('Match radius (px)','F1')]): basic_axes(axes[row,col],x,y)
            axes[row,0].set_title(family); axes[row,0].legend(fontsize=6)
        self.save(5,fig,data,'Edge threshold sensitivity on independent test realizations. Colored dots mark each model\'s independently validation-selected threshold; dotted lines identify the main-figure thresholds selected to favor Blob-Net on the displayed examples. The latter selection is exploratory. All curves exclude a 10-pixel image border; precision-recall points are threshold operating points, not calibrated posterior probabilities.')
        fig,axes=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True); out={}
        records=self.corpus('random','test',self.args.replicates)
        for ax,model in zip(axes,FAMILIES):
            grid=np.zeros((4,4)); preds=[self.predict(model,r['image']) for r in records]
            for i,sep in enumerate([1,2,3,5]):
                for j,window in enumerate([3,5,7,9]):
                    grid[i,j]=pooled([evaluate(p,r['coordinates'],separation=sep,window=window) for p,r in zip(preds,records)])['f1']
            im=ax.imshow(grid,vmin=0,vmax=1,cmap='viridis'); ax.set_xticks(range(4),[3,5,7,9]); ax.set_yticks(range(4),[1,2,3,5]); ax.set(xlabel='Centroid window (px)',ylabel='Peak separation (px)',title=LABELS[model]); out[model]=grid
        fig.colorbar(im,ax=list(axes),label='F1',shrink=.8)
        self.save(5,fig,out,'Random-image postprocessing sensitivity at fixed threshold 0.35 and matching radius 3 pixels. Each grid cell uses identical images and cached predictions. This separates heatmap quality from peak-extraction choices.', '-b')

    def s6(self):
        fig,axes=plt.subplots(3,5,figsize=(15,9),constrained_layout=True); output={}
        for row,family in enumerate(('mos2','sto','graphene')):
            mask=edge_mask(family,(512,512)); signed=signed_distance(mask)
            rec=self.edge(family,0)
            show(axes[row,0],mask,f'{family}: material mask')
            show(axes[row,1],rec['image'],'Rendered image',rec['coordinates'])
            # Render all projected sites before vacancies/rattle using documented
            # geometry, with the same material mask shown separately.
            before=pristine_edge_coordinates(family,(512,512))
            axes[row,2].scatter(before[:,1],before[:,0],s=1,color='#777777')
            axes[row,2].scatter(rec['coordinates'][:,1],rec['coordinates'][:,0],s=1,color='#c77122')
            axes[row,2].invert_yaxis(); axes[row,2].set_aspect('equal'); axes[row,2].set(xlim=(0,512),ylim=(512,0),title='Projected sites / retained atoms')
            bins=np.array([-100,-50,-25,-10,0,10,25,50,100,250]); centers=(bins[:-1]+bins[1:])/2
            output[family]={'bins_px':bins,'models':{},'construction':edge_construction(family)}
            for model in FAMILIES:
                fp_hist=np.zeros(len(bins)-1); true_hist=np.zeros_like(fp_hist); tp_hist=np.zeros_like(fp_hist); error_sums=np.zeros_like(fp_hist)
                per_image=[]
                for i in range(self.args.replicates):
                    r=self.edge(family,i); result=evaluate(self.predict(model,r['image']),r['coordinates'],border=10)
                    unmatched=unmatched_points(result['positions'],result['matched_predicted'])
                    fp_hist+=np.histogram(sample_distance(signed,unmatched),bins)[0]
                    true_hist+=np.histogram(sample_distance(signed,result['truth']),bins)[0]
                    d=sample_distance(signed,result['matched_truth'])
                    tp_hist+=np.histogram(d,bins)[0]; error_sums+=np.histogram(d,bins,weights=result['errors']**2)[0]
                    per_image.append(compact(result))
                valid=np.zeros_like(mask); valid[10:-10,10:-10]=True
                areas=np.histogram(signed[valid],bins)[0]
                rates=fp_hist/np.maximum(areas*self.args.replicates,1)*10000
                recall=np.divide(tp_hist,true_hist,out=np.full_like(tp_hist,np.nan),where=true_hist>0)
                rms=np.sqrt(np.divide(error_sums,tp_hist,out=np.full_like(tp_hist,np.nan),where=tp_hist>0))
                axes[row,3].plot(centers,rates,color=COLORS[model],label=LABELS[model])
                axes[row,4].plot(centers,recall,color=COLORS[model])
                output[family]['models'][model]=dict(fp=fp_hist,truth=true_hist,tp=tp_hist,fp_per_10000px=rates,recall=recall,rmse=rms,per_image=per_image)
            basic_axes(axes[row,3],'Signed boundary distance (px)','FP / 10,000 px'); basic_axes(axes[row,4],'Signed boundary distance (px)','Recall')
            axes[row,3].axvline(0,color='k',ls=':'); axes[row,4].axvline(0,color='k',ls=':'); axes[row,3].legend(fontsize=6)
        self.save(6,fig,output,f'Construction and boundary-distance diagnostics for {self.args.replicates} independent noise/vacancy/displacement realizations per material, with fixed boundary geometries. Gray sites precede masking, vacancies and displacement; orange sites are retained ground truth. Signed distance is positive inside the ideal material mask; displaced atoms can cross it. FP rates are area-normalized, recall is atom-normalized. Fixed threshold 0.35; 10-pixel image-border exclusion. This tests stochastic reproducibility, not the diversity of all possible edge shapes. Species intensities, widths, masks and displacement rules are recorded in source data.')

    def s7(self):
        levels=[0.,.1,.5,1.,2.5,4.]; fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True); output=[]
        y,x=np.mgrid[18:240:15,18:240:15]; xy=np.c_[y.ravel(),x.ravel()].astype(np.float32)
        xy=xy[xy[:,0]>100+.2*xy[:,1]]; cfg=controlled_config((256,256))
        for std in levels:
            collected={m:[] for m in FAMILIES}; regional={m:{'edge':[],'interior':[]} for m in FAMILIES}; common={m:[] for m in FAMILIES}
            for i in range(self.args.replicates):
                rng=np.random.default_rng(11_000_000+i); shift=rng.normal(size=xy.shape)
                moved=xy+std*shift; rec=syn.render_atom_image(moved,cfg,np.random.default_rng(12_000_000+i),intensities=np.full(len(xy),.8),sigmas=np.full(len(xy),2.))
                results={m:evaluate(self.predict(m,rec['image']),moved,border=10) for m in FAMILIES}
                ids={m:cKDTree(moved).query(r['matched_truth'])[1] if r['tp'] else np.array([],int) for m,r in results.items()}
                common_ids=set.intersection(*(set(ids[m].tolist()) for m in FAMILIES))
                for m,r in results.items():
                    collected[m].append(r)
                    common[m].extend([float(e) for k,e in zip(ids[m],r['errors']) if k in common_ids])
                    for zone in ('edge','interior'):
                        def zone_mask(p):
                            d=(p[:,0]-100-.2*p[:,1])/np.sqrt(1.04)
                            return d<30 if zone=='edge' else d>=30
                        t=r['truth'][zone_mask(r['truth'])]; p=r['positions'][zone_mask(r['positions'])]
                        rr=match_coordinate_sets(p,t,max_distance=3.); regional[m][zone].append(rr)
                if i==0 and std in (0.,2.5): show(axes[0,0 if std==0 else 1],rec['image'],f'Displacement SD {std:g} px',moved)
            for m in FAMILIES:
                output.append(dict(std=std,model=m,metrics=pooled(collected[m]),
                  edge=pooled(regional[m]['edge']),interior=pooled(regional[m]['interior']),
                  common_rmse=np.sqrt(np.mean(np.square(common[m]))) if common[m] else np.nan,common_matches=len(common[m])))
        for m in FAMILIES:
            r=[r for r in output if r['model']==m]
            for ax,key in [(axes[0,2],'f1'),(axes[1,0],'rmse')]: ax.plot(levels,[v['metrics'][key] for v in r],label=LABELS[m],color=COLORS[m])
            axes[1,1].plot(levels,[v['common_rmse'] for v in r],color=COLORS[m]); axes[1,2].plot(levels,[v['edge']['recall'] for v in r],color=COLORS[m],label=LABELS[m]); axes[1,2].plot(levels,[v['interior']['recall'] for v in r],color=COLORS[m],ls='--')
        for ax,ylabel in [(axes[0,2],'F1'),(axes[1,0],'Matched RMSE (px)'),(axes[1,1],'Common-subset RMSE (px)'),(axes[1,2],'Recall: edge / interior (dashed)')]: basic_axes(ax,'Displacement SD (px)',ylabel)
        axes[0,2].legend(fontsize=7)
        self.save(7,fig,output,'Controlled square-edge displacement sweep. Brightness, Gaussian width, vacancies (none), material geometry and noise seeds are fixed within each replicate; only the amplitude of the same displacement field changes. Solid/dashed regional recall separates a 30-pixel edge band from the interior. Common-subset RMSE uses the same ground-truth atoms successfully matched by every model. This intervention isolates displacement but does not reproduce every material-specific difference in main Figure 2.')

    def experimental(self):
        files=['pristine_monolayer_MoS2.h5','Sigma3_coherent_twin_grain_boundary_FCC_Al.h5','high_angle_grain_boundary_monolayer_WS2.h5','Al72Ni11Co17_quasicrystal.h5']
        result=[]
        for name in files:
            path=ROOT/'experimental_data'/name
            raw=mainfig._load_experimental_image(path); pixel=mainfig._read_channel_pixel_size_nm(path); selection=None
            # This is an explicit SI protocol, equal for both networks; it is
            # deliberately not claimed to be an exact replay of all main panels.
            display,native,transform=mainfig._make_fixed_fov_resolution_view(raw,.8,8.,512,max(64,int(round(512*pixel/.025))))
            result.append(dict(name=name,path=path,source=raw,display=display,image=native,
                               pixel_nm=pixel,actual_inference_pixel_nm=512*pixel/native.shape[0],
                               transform=transform,selection=selection,sha256=digest(path)))
        return result

    def s8(self):
        images=self.experimental(); fig,axes=plt.subplots(4,4,figsize=(12,12),constrained_layout=True); output=[]
        atlas,axs=plt.subplots(4,3,figsize=(10,12),constrained_layout=True)
        sensitivity,sax=plt.subplots(1,4,figsize=(13,3.5),constrained_layout=True)
        for row,record in enumerate(images):
            image=record['image']; blob=self.predict('random',image); hexx=self.predict('hexagonal',image)
            pb=extract_subpixel_peak_positions(blob); ph=extract_subpixel_peak_positions(hexx)
            rad=.06/record['actual_inference_pixel_nm']; matched=match_coordinate_sets(pb,ph,rad)
            only_b=unmatched_points(pb,matched['matched_predicted']); only_h=unmatched_points(ph,matched['matched_truth'])
            show(axes[row,0],mainfig._center_crop_or_pad(record['source'],512),record['name'].split(' - ')[0])
            show(axes[row,1],image,'Equal processed input'); show(axes[row,2],blob,'Blob-Net heatmap'); show(axes[row,3],hexx,'Hex-Net heatmap')
            add_si_scale_bar(axes[row,0],record['pixel_nm'],1.)
            add_si_scale_bar(axes[row,1],record['actual_inference_pixel_nm'],1.)
            candidates=np.concatenate([only_b,only_h]); center=candidates[len(candidates)//2] if len(candidates) else np.array(image.shape)/2
            y,x=np.round(center).astype(int); y0=max(0,y-35); y1=min(image.shape[0],y+35); x0=max(0,x-35); x1=min(image.shape[1],x+35)
            for col,array in enumerate((image,blob,hexx)): show(axs[row,col],array[y0:y1,x0:x1],['Input disagreement crop','Blob-Net','Hex-Net'][col])
            add_si_scale_bar(axs[row,0],record['actual_inference_pixel_nm'],.25)
            axs[row,0].set_ylabel(record['name'].split(' - ')[0],fontsize=8)
            for points,color in ((only_b,COLORS['random']),(only_h,COLORS['hexagonal'])):
                inside=(points[:,0]>=y0)&(points[:,0]<y1)&(points[:,1]>=x0)&(points[:,1]<x1)
                for ax in axs[row]: ax.scatter(points[inside,1]-x0,points[inside,0]-y0,s=28,facecolors='none',edgecolors=color,linewidths=.8)
            scale_results=[]
            for scale in (.8,1.,1.2):
                scaled=zoom(image,scale,order=1,prefilter=False)
                preds={m:self.predict(m,scaled) for m in ('random','hexagonal')}
                for threshold in (.25,.35,.45):
                    positions={m:extract_subpixel_peak_positions(p,threshold_rel=threshold) for m,p in preds.items()}
                    res=match_coordinate_sets(positions['random'],positions['hexagonal'],rad*scale)
                    scale_results.append(dict(scale=scale,threshold=threshold,blob_count=len(positions['random']),hex_count=len(positions['hexagonal']),shared=res['tp'],blob_only=res['fp'],hex_only=res['fn']))
            for m,key in [('random','blob_count'),('hexagonal','hex_count')]:
                for scale,style in zip((.8,1.,1.2),(':','-','--')):
                    rr=[v for v in scale_results if v['scale']==scale]; sax[row].plot([v['threshold'] for v in rr],[v[key] for v in rr],style,color=COLORS[m],label=f'{m}, {scale:g}x')
            basic_axes(sax[row],'Relative threshold','Detected columns'); sax[row].set_title(record['name'].split(' - ')[0],fontsize=8)
            radius_sensitivity={r:match_coordinate_sets(pb,ph,r/record['actual_inference_pixel_nm'])['tp'] for r in (.04,.06,.08)}
            output.append({k:v for k,v in record.items() if k not in ('source','display','image')} | dict(blob_positions=pb,hex_positions=ph,blob_only=only_b,hex_only=only_h,radius_sensitivity=radius_sensitivity,scale_threshold_sensitivity=scale_results,crop_yxyx=[y0,x0,y1,x1]))
        sax[-1].legend(fontsize=5)
        caption='Source crops, explicitly processed inputs and model heatmaps for four experimental files. Both archived models receive identical DoG processing (sigma 0.8/8 source pixels), percentile normalization and approximately 0.025 nm inference sampling. Actual sampling, source hashes and channel selection are recorded. The first file is labeled WS2 in its source; its conflict with the manuscript MoS2 designation remains unresolved. File-based labels do not independently establish composition. No experimental ground truth or accuracy claim is inferred.'
        self.save(8,fig,output,caption)
        self.save(8,atlas,output,'Automatically selected disagreement crops (middle coordinate in the ordered disagreement list), with Blob-Net-only and Hex-Net-only positions outlined in green and orange. Selection is reproducible, not a claim that one model is correct. Original coordinates and the 0.04/0.06/0.08 nm agreement-radius sensitivity are supplied for independent annotation.', '-b')
        self.save(8,sensitivity,output,'Experimental detection counts under threshold and scale perturbations with identical processing for both networks. Solid curves are the nominal scale; dotted/dashed curves are 0.8/1.2 times the nominal pixel-array size. Detection count and inter-model agreement are not accuracy measurements.', '-c')

    def s9(self):
        factors=[.5,.67,1.,1.5,2.,3.,4.]; output=[]
        cfg=replace(self.configs['random'],image_shape=(256,256),min_atoms=300,max_atoms=350)
        # Paired scenes, same physical FOV, pixel-center aligned interpolation.
        for factor in factors:
            for protocol in ('resampled','regenerated'):
                records={}
                for split,n in [('validation',self.args.validation_samples),('test',self.args.replicates)]:
                    records[split]=[]
                    for i in range(n):
                        base=self.record('random',split,i,cfg)
                        if protocol=='resampled':
                            size=int(round(256/factor)); image=mainfig._interpolate_image(base['image'],(size,size))
                            scale=(size-1)/255; coords=base['coordinates']*scale; actual=1/scale
                        else:
                            scaled=replace(cfg,sigma_range=tuple(v/factor for v in cfg.sigma_range),
                              min_separation_range=tuple(v/factor for v in cfg.min_separation_range),target_sigma=cfg.target_sigma/factor)
                            r=self.record('random',split,i,scaled); image=r['image']; coords=r['coordinates']; actual=factor
                        records[split].append(dict(image=image,coordinates=coords,actual_factor=actual))
                for model in FAMILIES:
                    preds={split:[self.predict(model,r['image']) for r in rr] for split,rr in records.items()}
                    threshold=select_threshold(preds['validation'],records['validation'])
                    for mode,t,radius in [('fixed',.35,3.),('validation',threshold,3.),('physical_tolerance',.35,3./records['test'][0]['actual_factor'])]:
                        rr=[evaluate(p,r['coordinates'],t,radius=radius) for p,r in zip(preds['test'],records['test'])]
                        met=pooled(rr)
                        met['rmse_angstrom']=met['rmse']*records['test'][0]['actual_factor']*.1062231596676199
                        output.append(dict(factor=factor,actual_factor=records['test'][0]['actual_factor'],protocol=protocol,model=model,mode=mode,threshold=t,radius_px=radius,metrics=met,truth_count=sum(len(r['coordinates']) for r in records['test'])))
            print('S9 factor',factor,flush=True)
        fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
        for row,protocol in enumerate(('resampled','regenerated')):
            for m in FAMILIES:
                for mode,style in [('fixed','-'),('validation','--'),('physical_tolerance',':')]:
                    rr=[r for r in output if r['protocol']==protocol and r['model']==m and r['mode']==mode]
                    for col,metric in enumerate(('f1','recall','rmse','rmse_angstrom')):
                        axes[row,col].plot(factors,[r['metrics'][metric] for r in rr],style,color=COLORS[m],label=f'{LABELS[m]}: {mode}')
            for col,metric in enumerate(('F1','Recall','RMSE (px)','RMSE (angstrom)')):
                basic_axes(axes[row,col],'Physical pixel-size factor',metric); axes[row,col].set_title(protocol); axes[row,col].axvline(1,color='#999999',ls=':')
        axes[0,0].legend(fontsize=5)
        self.save(9,fig,output,'Three-model scale transfer. Top: interpolated copies of identical noisy scenes with fixed physical field of view and atom identities (endpoint-aligned coordinate transforms). Bottom: freshly rendered scenes with scaled feature widths/minimum separations, fixed 256-pixel array and requested count range; actual counts are recorded because packing can saturate. Solid: threshold 0.35 and 3-pixel tolerance. Dashed: independently validation-selected threshold, same tolerance. Dotted: threshold 0.35 and physical-distance-preserving tolerance. The 0.106223 angstrom reference is an explicit scale convention for synthetic data, not a calibrated dose experiment. RMSE is conditional on matches; TP counts and precision are supplied in source data.')

    def s10(self):
        widths=[.8,1.4,2.,3.,4.5,6.]; spacings=[7.,11.,15.,24.,48.,80.]; output={m:np.zeros((6,6)) for m in FAMILIES}
        for wi,width in enumerate(widths):
            for si,spacing in enumerate(spacings):
                results={m:[] for m in FAMILIES}
                for i in range(max(2,self.args.replicates//4)):
                    rng=np.random.default_rng(13_000_000+i)
                    # A uniform periodic test grid provides exact spacing; independent
                    # phase and intensity vary across replicates. No Z/brightness confound.
                    yy,xx=np.meshgrid(np.arange(16,240,spacing),np.arange(16,240,spacing),indexing='ij')
                    coords=np.c_[yy.ravel(),xx.ravel()]+rng.uniform(-2,2,(1,2))
                    rec=syn.render_atom_image(coords,controlled_config((256,256)),np.random.default_rng(14_000_000+i),intensities=np.full(len(coords),.8),sigmas=np.full(len(coords),width))
                    for m in FAMILIES: results[m].append(evaluate(self.predict(m,rec['image']),coords))
                for m in FAMILIES: output[m][wi,si]=pooled(results[m])['f1']
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for ax,m in zip(axes,FAMILIES):
            im=ax.imshow(output[m],vmin=0,vmax=1,cmap='viridis',origin='lower',aspect='auto')
            ax.set_xticks(range(6),spacings); ax.set_yticks(range(6),widths); ax.set(xlabel='Spacing (px)',ylabel='Gaussian sigma (px)',title=LABELS[m])
            # Bounds of the current configs, clearly distinguished from verified training provenance.
            ax.add_patch(plt.Rectangle((.95,.95),1.05,2.05,fill=False,edgecolor='white',ls='--',lw=1.5))
        fig.colorbar(im,ax=list(axes),label='F1',shrink=.8)
        self.save(10,fig,dict(widths=widths,spacings=spacings,f1=output),'Independent feature-width/spacing map on controlled square arrays with constant amplitude and noise. Each cell uses identical images for all models. White dashed regions approximately indicate current configuration ranges, not recovered checkpoint training bounds. Gaussian sigma is reported directly; FWHM is 2.355 times sigma. This grid separates width from brightness, unlike a species-dependent intensity illustration.')
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
        stages=['input','enc1','pool1','enc2','pool2','enc3','pool3','bottleneck']
        rf=[1,5,6,14,16,32,36,68]; axes[0].plot(range(8),rf,'o-',color='#444444'); axes[0].set_xticks(range(8),stages,rotation=50,ha='right'); axes[0].set_ylabel('Theoretical receptive field (px)')
        axes[0].set_title('Bottleneck RF = 68 px')
        contexts={}; yy,xx=np.mgrid[0:128,0:128]; radius=np.hypot(yy-64,xx-64); central=np.exp(-((yy-64)**2+(xx-64)**2)/8).astype(np.float32)
        for name,family in [('square','square'),('hexagonal','hexagonal'),('random','random')]:
            cfg=replace(self.configs[family],image_shape=(128,128))
            if family=='random': cfg=replace(cfg,min_atoms=60,max_atoms=70)
            else: cfg=replace(cfg,min_atoms=20)
            ctx=self.record(family,'test',77,cfg)['image'].copy(); ctx[radius<18]=central[radius<18]
            contexts[name]=ctx
        context_data={}
        for m in FAMILIES:
            vals=[float(self.predict(m,im)[64,64]) for im in contexts.values()]
            axes[1].plot(list(contexts),vals,'o-',label=LABELS[m],color=COLORS[m]); context_data[m]=vals
        axes[1].set(ylabel='Central heatmap value',title='Same central 18-px-radius patch'); axes[1].legend(fontsize=6)
        model=self.model('random'); inp=torch.from_numpy(contexts['random']).to(self.device)[None,None].requires_grad_(True)
        val=torch.sigmoid(model(inp))[0,0,64,64]; grad=torch.autograd.grad(val,inp)[0][0,0].detach().cpu().numpy()
        axes[2].imshow(np.log10(np.abs(grad)+1e-12),cmap='magma'); axes[2].set_title('Blob-Net input-gradient magnitude (log)'); axes[2].set_xticks([]); axes[2].set_yticks([])
        self.save(10,fig,dict(stages=stages,rf=rf,context_values=context_data,output_dependency='At output phase (64,64), decoder/upsampling yields 92 input pixels of support in each axis for this architecture; validate with structural propagation test. Different phases shift endpoints.',gradient=grad),
          'Architecture and context intervention. The 68-pixel bottleneck RF is not the output RF: decoder convolutions extend output support (92 pixels for the tested phase). Evaluation-mode BatchNorm uses stored statistics. Central intensity pixels within radius 18 are held exactly fixed while the external context changes; no post-intervention normalization is applied. Context-response differences and a single input-gradient map are diagnostic examples, not proof of complete locality or a population-level effective RF.', '-b')

    def s11(self):
        experiments={'Poisson scale':[2.,8.,32.,128.,512.], 'Read noise SD':[0.,.03,.08,.15,.3], 'Background amplitude':[0.,.05,.15,.3,.6]}
        fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True); data=[]
        base=replace(self.configs['random'],image_shape=(128,128),min_atoms=60,max_atoms=80,
                     total_counts_range=(64.,64.),read_noise_std_range=(.03,.03),inhomogeneous_background_range=(.05,.05))
        for col,(name,levels) in enumerate(experiments.items()):
            for value in levels:
                cfg=replace(base,**({'total_counts_range':(value,value)} if name=='Poisson scale' else
                                   {'read_noise_std_range':(value,value)} if name=='Read noise SD' else
                                   {'inhomogeneous_background_range':(value,value)}))
                records={sp:[self.record('random',sp,i+400,cfg) for i in range(n)] for sp,n in [('validation',self.args.validation_samples),('test',self.args.replicates)]}
                # Tune LoG sigma and threshold ONLY on separate validation images.
                best=(-1,None,None)
                for sigma in (1.,2.,3.,4.):
                    vp=[log_response(r['image'],sigma) for r in records['validation']]
                    t=select_threshold(vp,records['validation'])
                    score=pooled([evaluate(p,r['coordinates'],t) for p,r in zip(vp,records['validation'])])['f1']
                    if score>best[0]: best=(score,sigma,t)
                for method in ('random','log'):
                    if method=='random':
                        vp=[self.predict('random',r['image']) for r in records['validation']]
                        threshold=select_threshold(vp,records['validation']); preds=[self.predict('random',r['image']) for r in records['test']]
                    else:
                        threshold=best[2]; preds=[log_response(r['image'],best[1]) for r in records['test']]
                    met=pooled([evaluate(p,r['coordinates'],threshold) for p,r in zip(preds,records['test'])])
                    data.append(dict(experiment=name,value=value,method=method,threshold=threshold,log_sigma=best[1] if method=='log' else None,metrics=met))
            for method,color in [('random',COLORS['random']),('log','#8559a8')]:
                rr=[r for r in data if r['experiment']==name and r['method']==method]
                axes[0,col].plot(levels,[r['metrics']['f1'] for r in rr],'o-',label='Blob-Net' if method=='random' else 'LoG',color=color)
                axes[1,col].plot(levels,[r['metrics']['rmse'] for r in rr],'o-',color=color)
            basic_axes(axes[0,col],name,'F1'); basic_axes(axes[1,col],name,'Matched RMSE (px)'); axes[0,col].legend()
        self.save(11,fig,data,'Synthetic noise/background robustness with a scale-normalized Laplacian-of-Gaussian baseline. Each method receives identical images. Thresholds, and LoG sigma, are selected independently for each condition on validation images only, then frozen for test evaluation. One nuisance parameter changes at a time; identical seeds retain paired latent scenes where the generator permits. Poisson scale is not a calibrated microscope dose. Source data include precision, recall, TP/FP/FN and calibration settings.')

    def s12(self):
        """Independent repeat-training experiment with declared compact crops.

        Training data are fixed across initialization seeds within a regime;
        validation and test use distinct million-offset seeds. These are new
        SI runs, never relabeled as original manuscript training histories.
        """
        regimes=['square','hexagonal','random','wide_random','dense_random']; data=[]; histories={}
        for regime in regimes:
            family=regime if regime in FAMILIES else 'random'
            cfg=replace(self.configs[family],image_shape=(self.args.training_size,self.args.training_size))
            area=(self.args.training_size/512)**2
            if family=='random': cfg=replace(cfg,min_atoms=max(16,int(1300*area)),max_atoms=max(20,int(1500*area)))
            else: cfg=replace(cfg,min_atoms=max(12,int(200*area)))
            if regime=='wide_random': cfg=replace(cfg,sigma_range=(1.,4.5),min_separation_range=(8.,20.))
            if regime=='dense_random': cfg=replace(cfg,sampling_mode='relaxed_dense',min_atoms=max(20,int(1800*area)),max_atoms=max(24,int(2000*area)),relaxation_iterations=12)
            # Materialize and fingerprint the precise new training set.
            training=[self.record(family,'training',i,cfg) for i in range(self.args.training_samples)]
            validation=[self.record(family,'validation',i+1000,cfg) for i in range(self.args.training_validation_samples)]
            tx=torch.from_numpy(np.stack([r['image'] for r in training]))[:,None]
            ty=torch.from_numpy(np.stack([r['target'] for r in training]))[:,None]
            vx=torch.from_numpy(np.stack([r['image'] for r in validation]))[:,None]
            vy=torch.from_numpy(np.stack([r['target'] for r in validation]))[:,None]
            for seed in self.args.training_seeds:
                spec=dict(regime=regime,config=asdict(cfg),seed=seed,epochs=self.args.training_epochs,
                   samples=self.args.training_samples,val_samples=self.args.training_validation_samples,batch_size=16,
                   training_payload_sha256=hashlib.sha256(tx.numpy().tobytes()+ty.numpy().tobytes()).hexdigest(),
                   learning_rate=.001,filters=[32,64,128,256],dropout=.2,
                   loss=dict(mse_weight=.5,peak_weight=.5,threshold=.1,peak_boost=5.,from_logits=True))
                folder=self.out/'training'/f'{regime}_seed{seed}_{fingerprint(spec)}'; folder.mkdir(parents=True,exist_ok=True)
                cp=folder/'unet_best.pth'; history_path=folder/'summary.json'
                torch.manual_seed(seed); np.random.seed(seed)
                model=build_unet(num_filters=[32,64,128,256],dropout=.2).to(self.device)
                if cp.exists() and history_path.exists():
                    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model_state_dict']); hist=json.loads(history_path.read_text())
                else:
                    generator=torch.Generator().manual_seed(seed)
                    train_loader=DataLoader(TensorDataset(tx,ty),batch_size=16,shuffle=True,generator=generator)
                    val_loader=DataLoader(TensorDataset(vx,vy),batch_size=16)
                    started=time.time()
                    model,tr,va=train_model(model,train_loader,val_loader,self.args.training_epochs,
                      CombinedGaussianLoss(),torch.optim.Adam(model.parameters(),lr=.001),self.device,str(folder/'unet'),
                      progress_interval=99999,early_stopping_patience=6,early_stopping_min_delta=.0005)
                    hist=dict(spec=spec,train=tr,validation=va,best_epoch=int(np.argmin(va))+1,seconds=time.time()-started,device=str(self.device))
                    dump(history_path,hist)
                model.eval(); histories[f'{regime}/{seed}']=hist
                for test_family in FAMILIES:
                    tc=replace(self.configs[test_family],image_shape=(self.args.training_size,self.args.training_size))
                    if test_family=='random': tc=replace(tc,min_atoms=max(16,int(1300*area)),max_atoms=max(20,int(1500*area)))
                    else: tc=replace(tc,min_atoms=max(12,int(200*area)))
                    records=[self.record(test_family,'test',i+2000,tc) for i in range(self.args.training_test_samples)]
                    results=[]
                    for record in records:
                        p=mainfig._predict_array(model,record['image'],self.device); results.append(evaluate(p,record['coordinates']))
                    data.append(dict(regime=regime,seed=seed,test_family=test_family,metrics=pooled(results),per_image=[compact(r) for r in results],checkpoint=cp,sha256=digest(cp)))
                del model
                if self.device.type=='mps': torch.mps.empty_cache()
                print('S12 completed',regime,seed,flush=True)
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for ax,family in zip(axes,FAMILIES):
            for j,regime in enumerate(regimes):
                values=[r['metrics']['f1'] for r in data if r['regime']==regime and r['test_family']==family]
                ax.scatter(np.full(len(values),j)+np.linspace(-.1,.1,len(values)),values,s=25,color=COLORS.get(regime,'#8559a8'))
                ax.plot([j-.25,j+.25],[np.mean(values)]*2,color='black')
            ax.set_xticks(range(5),['Square','Hex','Random','Wide random','Dense random'],rotation=35,ha='right'); ax.set(title=f'{family} test',ylabel='F1'); ax.grid(axis='y',alpha=.2)
        self.save(12,fig,dict(results=data,histories=histories),f'New controlled repeat-training experiment: {self.args.training_samples} training and {self.args.training_validation_samples} validation images of {self.args.training_size} by {self.args.training_size} pixels per regime; {len(self.args.training_seeds)} initialization seeds; up to {self.args.training_epochs} epochs. Dots are individual training runs, horizontal lines their means. All models share architecture, optimizer, stopping rule and fixed test threshold 0.35. Test images are paired across runs. Wide/dense random regimes change width/spacing or packing distributions. These compact-image SI runs assess seed and distribution effects; they are not retraining replicas of the 512-pixel manuscript experiments, and density remains a measured covariate.')
        fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
        for ax,regime in zip(axes.flat,regimes):
            for color_index,seed in enumerate(self.args.training_seeds):
                color=plt.get_cmap('tab10')(color_index)
                hist=histories[f'{regime}/{seed}']; ax.plot(range(1,len(hist['train'])+1),hist['train'],alpha=.65,color=color)
                ax.plot(range(1,len(hist['validation'])+1),hist['validation'],'--',label=f'seed {seed}',color=color)
            ax.set_title(regime); basic_axes(ax,'Epoch','Loss'); ax.legend(fontsize=7)
        axes[1,2].axis('off'); axes[1,2].text(0,.9,'Solid: training\nDashed: validation\n\nExact configurations, histories,\nseeds and checkpoint hashes\nare recorded in source data.',va='top')
        self.save(12,fig,histories,'All new supplemental training histories. Solid curves show training and dashed curves validation. Seed labels denote initialization and shuffle seeds; generated datasets are held fixed within each regime. Selected checkpoints and timing are recorded per run.', '-b')

    def s13(self):
        record=self.experimental()[-1]; image=record['image']
        image=mainfig._interpolate_image(image,(768,768))
        start=time.perf_counter(); whole=self.predict('random',image); whole_seconds=time.perf_counter()-start
        tiled=mainfig._predict_tiled(self.model('random'),image,self.device,256,96,1)
        diff=np.abs(whole-tiled); pred_whole=extract_subpixel_peak_positions(whole); pred_tiled=extract_subpixel_peak_positions(tiled)
        match=match_coordinate_sets(pred_tiled,pred_whole,3.)
        starts=mainfig._tile_starts(768,256,160)
        seam=np.zeros_like(image,bool)
        for k in starts[1:]: seam[max(0,k-3):k+4,:]=True; seam[:,max(0,k-3):k+4]=True
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        show(axes[0],image,'Large-field input'); show(axes[1],tiled,'Overlapping tiled prediction')
        im=axes[2].imshow(diff,cmap='magma'); axes[2].set_title('Absolute tiled - whole difference'); axes[2].set_xticks([]); axes[2].set_yticks([]); fig.colorbar(im,ax=axes[2],shrink=.7)
        data=dict(image_shape=image.shape,tile_size=256,overlap=96,source=record['path'],mean_abs=float(diff.mean()),max_abs=float(diff.max()),seam_mean_abs=float(diff[seam].mean()),nonseam_mean_abs=float(diff[~seam].mean()),coordinate_agreement=compact_match(match),whole_inference_or_cache_seconds=whole_seconds)
        self.save(13,fig,data,'Whole-image versus Hann-weighted overlapping tiled inference on the same 768-pixel experimental view. Tile size 256, overlap 96; coordinate agreement uses a 3-pixel radius and does not constitute experimental accuracy. Error is evaluated both near tile starts and elsewhere; outer-image padding can also contribute. Timing includes possible prediction-cache lookup and is not presented as a speed benchmark.')
        data=[]; fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
        for i in range(self.args.replicates):
            cfg=replace(self.configs['random'],image_shape=(128,128),min_atoms=50,max_atoms=65,total_counts_range=(256.,256.),read_noise_std_range=(.02,.02),sigma_range=(1.7,2.3))
            rec=self.record('random','test',i+4000,cfg); heat=self.predict('random',rec['image'])
            initial=extract_subpixel_peak_positions(heat); refined,success=refine_gaussians(rec['image'],initial)
            before=match_coordinate_sets(initial,rec['coordinates'],3.); after=match_coordinate_sets(refined,rec['coordinates'],3.)
            # Comparison holds the initial accepted correspondence fixed, rather
            # than silently discarding fits that move outside the match gate.
            ids=cKDTree(initial).query(before['matched_predicted'])[1] if before['tp'] else np.array([],int)
            paired_errors=np.linalg.norm(refined[ids]-before['matched_truth'],axis=1)
            data.append(dict(before_errors=before['errors'],after_errors_same_initial_matches=paired_errors,fit_successes=int(success.sum()),fit_attempts=len(success),before=compact_match(before),after=compact_match(after)))
        initial_errors=np.concatenate([r['before_errors'] for r in data]); final_errors=np.concatenate([r['after_errors_same_initial_matches'] for r in data])
        axes[0].hist(initial_errors,bins=30,alpha=.6,label='Heatmap centroid'); axes[0].hist(final_errors,bins=30,alpha=.6,label='Gaussian refinement'); axes[0].legend(fontsize=7); basic_axes(axes[0],'Error (px)','Initial matched atoms')
        axes[1].scatter(initial_errors,final_errors,s=3,alpha=.3); limit=max(initial_errors.max(initial=1),final_errors.max(initial=1)); axes[1].plot([0,limit],[0,limit],'k--'); basic_axes(axes[1],'Initial error (px)','Refined error (px)')
        axes[2].axis('off'); axes[2].text(0,.9,f'Initial RMSE: {np.sqrt(np.mean(initial_errors**2)):.3f} px\nRefined RMSE: {np.sqrt(np.mean(final_errors**2)):.3f} px\n\nSame initial matched subset\nGaussian + planar background\n11 x 11 pixel fitting window\nFailed fits retain initial position',va='top')
        self.save(13,fig,data,'Gaussian refinement on independent synthetic images with known centers. A bounded isotropic Gaussian plus planar background is fitted in an 11-pixel window initialized from each Blob-Net heatmap centroid. Paired errors retain the original matched atom identities, including unsuccessful fits (which retain their initial position). Post-fit detection metrics and fit success counts are supplied. This is a favorable Gaussian-model test, not a claim of experimental picometer precision.', '-b')


def add_si_scale_bar(ax,pixel_nm,length_nm):
    xmin,xmax=ax.get_xlim();ymax,ymin=ax.get_ylim()
    length=length_nm/pixel_nm
    if length>.7*(xmax-xmin):
        length_nm=length_nm/2;length=length_nm/pixel_nm
    left=xmin+.07*(xmax-xmin);bottom=ymax-.08*(ymax-ymin)
    ax.plot([left,left+length],[bottom,bottom],color='white',lw=2.5)
    ax.text(left,bottom-.04*(ymax-ymin),f'{length_nm:g} nm',color='white',fontsize=7)


def log_response(image,sigma):
    return np.maximum(-gaussian_laplace(image,sigma)*sigma**2,0).astype(np.float32)


def controlled_config(shape):
    return syn.ImageFormationConfig(image_shape=shape,sigma_range=(2.,2.),intensity_range=(.8,.8),
       target_sigma=2.,background_range=(.05,.05),gradient_range=(0.,0.),
       inhomogeneous_background_range=(0.,0.),low_frequency_noise_range=(0.,0.),
       read_noise_std_range=(.03,.03),total_counts_range=(64.,64.),blur_sigma_range=(.5,.5),edge_padding=16)


def compact_match(result):
    return {k:result[k] for k in ('tp','fp','fn')} | {'rmse':float(np.sqrt(np.mean(result['errors']**2))) if result['tp'] else np.nan}


def unmatched_points(points, matched):
    if len(points)==0: return np.empty((0,2))
    if len(matched)==0: return points.copy()
    return points[cKDTree(matched).query(points)[0]>1e-4]


def signed_distance(mask):
    return distance_transform_edt(mask)-distance_transform_edt(~mask)


def sample_distance(field,points):
    points=np.asarray(points).reshape(-1,2)
    if not len(points): return np.array([])
    idx=np.clip(np.round(points).astype(int),0,np.array(field.shape)-1)
    return field[idx[:,0],idx[:,1]]


def edge_mask(family,shape):
    y,x=np.mgrid[:shape[0],:shape[1]]
    if family=='mos2': return ((y>150+.26*x+8*np.sin(x/42))&((x-330)**2+(y-238)**2>52**2))|(((x-168)**2/80**2+(y-340)**2/45**2)<1)
    if family=='sto': return (y>135+.22*x+7.5*np.sin(x/48))&((x-340)**2+(y-258)**2>42**2)
    return (y>145+.18*x+9*np.sin(x/34))&((x-315)**2+(y-248)**2>46**2)


def edge_construction(family):
    return {
     'mos2':dict(projection='ASE MoS2, 44x44; merge projected columns',rotation_deg=17,pixel_angstrom=.12,intensity='Z^1.45 / max',sigma_px=[1.15,2.65],edge_bulk_displacement_sd=[.32,.11],edge_bulk_vacancy_probability=[.18,.025]),
     'sto':dict(projection='ASE STO, 44x44; merge tolerance 0.45 angstrom',rotation_deg=-8,pixel_angstrom=.13,intensity='normalized species weights, low 0.22 high 1',sigma_px=[1.15,2.65],edge_bulk_displacement_sd=[.38,.10],edge_bulk_vacancy_probability=[.13,.02]),
     'graphene':dict(projection='Honeycomb basis, nearest neighbor 15 px',rotation_deg=10,intensity=.78,sigma_px=1.9,edge_bulk_displacement_sd=[2.45,.18],edge_bulk_vacancy_probability=[.10,.01])}[family]


def pristine_edge_coordinates(family,shape):
    if family in ('mos2','sto'):
        unit=syn.build_ase_structure_unit_cell('mos2' if family=='mos2' else 'sto').repeat((44,44,1))
        xy=np.asarray(unit.get_positions(),np.float32)[:,:2]; xy-=xy.mean(axis=0,keepdims=True)
        xy,_=mainfig._merge_projected_columns_for_rendering(xy,np.asarray(unit.get_atomic_numbers()),**({'tolerance':.45} if family=='sto' else {}))
        angle=17 if family=='mos2' else -8; pixel=.12 if family=='mos2' else .13
        theta=np.deg2rad(angle); rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        xy=xy@rot.T/pixel; xy+=np.array([shape[1]*(.49 if family=='mos2' else .50),shape[0]*(.54 if family=='mos2' else .55)])
    else:
        a1=np.array([np.sqrt(3)*15,0]); a2=np.array([np.sqrt(3)*7.5,22.5])
        xy=np.array([i*a1+j*a2+b for i in range(-28,29) for j in range(-28,29) for b in (np.array([0,0]),np.array([0,15]))]); xy-=xy.mean(axis=0)
        theta=np.deg2rad(10); xy=xy@np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T; xy+=[shape[1]*.48,shape[0]*.56]
    keep=(xy[:,0]>=0)&(xy[:,0]<shape[1])&(xy[:,1]>=0)&(xy[:,1]<shape[0]); return xy[keep,::-1]


def refine_gaussians(image,positions):
    refined=np.asarray(positions).copy(); success=np.zeros(len(positions),bool)
    for i,(y,x) in enumerate(positions):
        iy,ix=int(round(y)),int(round(x)); y0=max(0,iy-5); y1=min(image.shape[0],iy+6); x0=max(0,ix-5); x1=min(image.shape[1],ix+6)
        patch=image[y0:y1,x0:x1]; yy,xx=np.mgrid[y0:y1,x0:x1]
        if min(patch.shape)<7: continue
        def residual(p):
            a,cy,cx,s,b,gy,gx=p
            return (a*np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*s*s))+b+gy*(yy-y)+gx*(xx-x)-patch).ravel()
        initial=[max(float(patch.max()-patch.min()),.01),y,x,2.,float(patch.min()),0.,0.]
        fit=least_squares(residual,initial,bounds=([0,y-2,x-2,.5,-1,-.2,-.2],[2,y+2,x+2,5,2,.2,.2]),max_nfev=80)
        if fit.success and np.all(np.isfinite(fit.x)): refined[i]=fit.x[1:3]; success[i]=True
    return refined,success


def latex_escape(text):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(c,c) for c in str(text))


def figure_result_note(number, data):
    """Short measured-result prose, with no assumptions about favorable outcomes."""
    if number == 2:
        return 'The local audit found ' + '; '.join(f"{k}: {v['actual_counts']['train']} training, {v['actual_counts']['val']} validation and {v['actual_counts']['test']} test images" for k,v in data['audit'].items()) + '. No complete provenance to all archived training runs is asserted.'
    if number == 3:
        return 'Selected checkpoint epochs were ' + ', '.join(f"{LABELS[k]} {v['metrics']['best_epoch']}" for k,v in data.items()) + '.'
    if number == 4:
        return 'Pooled F1 for Blob-Net in this new evaluation was ' + ', '.join(f"{data[k]['random']['pooled']['f1']:.3f} on {k}" for k in FAMILIES) + '. Paired confidence intervals are reported in the source data; these results use current saved SI configurations, not an authenticated reconstruction of the original evaluation.'
    if number == 5:
        return 'Validation-selected thresholds (Square-Net, Hex-Net, Blob-Net) were ' + '; '.join(k+': '+', '.join(f"{v[m]['validation_threshold']:.3f}" for m in FAMILIES) for k,v in data.items()) + '. The test curves should be used to assess the range of settings over which any ranking persists.'
    if number == 7:
        rows=[r for r in data if r['model']=='random' and r['std'] in (0.,2.5)]
        return 'In the controlled displacement experiment, Blob-Net matched-subset RMSE was ' + ', '.join(f"{r['metrics']['rmse']:.3f} px at displacement SD {r['std']:g} px" for r in rows) + '. This experiment does not reproduce the improved Blob-Net localization seen in the cross-material graphene comparison; displacement alone is not an established explanation for that improvement.'
    if number == 8:
        return 'The nominal SI protocol produced ' + '; '.join(f"{r['name'].split(' - ')[0]}: {len(r['blob_only'])} Blob-Net-only and {len(r['hex_only'])} Hex-Net-only detections" for r in data) + '. These are disagreements, not verified true or false detections.'
    if number == 12:
        return f"The controlled training experiment contains {len(data['histories'])} completed runs. It uses compact crops and is reported separately from the original manuscript training histories."
    if number == 13:
        return f"Mean absolute tiled/whole heatmap difference was {data['mean_abs']:.6f}; maximum difference was {data['max_abs']:.6f}. The matching analysis found {data['coordinate_agreement']['tp']} shared coordinate detections."
    return ''


def write_tables(study):
    rows=[]
    # Complete current configuration table, including all geometry-specific fields.
    parameters={k:asdict(v) for k,v in study.configs.items()}
    keys=sorted(set().union(*(p.keys() for p in parameters.values())))
    for key in keys:
        vals=[]
        for family in FAMILIES:
            value=parameters[family].get(key,'--')
            if value is None: value='not used'
            elif isinstance(value,(tuple,list)): value=', '.join(str(x) for x in value)
            vals.append(latex_escape(value))
        rows.append(latex_escape(key.replace('_',' '))+' & '+' & '.join(vals)+r'\\')
    text=r'''\clearpage
\section{Supplementary parameter and provenance tables}
\small
\begin{longtable}{p{2.0in}p{1.15in}p{1.15in}p{1.15in}}
\caption{Complete current configuration parameters used for the new SI cross-geometry evaluation. These are distinct from a verified reconstruction of original training files. Figure-specific overrides are stated in captions and machine-readable source data.}\\
\hline Parameter & Square & Hexagonal & Random\\\hline\endfirsthead
\hline Parameter & Square & Hexagonal & Random\\\hline\endhead
'''+ '\n'.join(rows)+r'''
\hline\end{longtable}
\normalsize
\begin{table}[!htbp]\centering\small
\caption{Archived model provenance. Identifiers are the first 12 hexadecimal characters of the SHA-256 checkpoint digest; full digests and absolute source paths are provided in run\_manifest.json.}
\begin{tabular}{lrrl}\hline
Model & Epochs & Training images & Checkpoint identifier\\\hline
'''
    for family in FAMILIES:
        metrics=json.loads((study.args.model_dir/family/'training_metrics.json').read_text())
        text+=f"{LABELS[family]} & {metrics['epochs_completed']} & {metrics['train_samples']} & "+r'\texttt{'+study.checkpoints[family]['sha256'][:12]+r'}\\'+'\n'
    text+=r'''\hline\end{tabular}\end{table}
\begin{table}[!htbp]\centering\small
\caption{Default supplemental evaluation and preprocessing settings. Per-figure changes are explicit in captions and source data.}
\begin{tabular}{p{2.6in}p{3.0in}}\hline
Setting & Value\\\hline
Relative detection threshold & 0.35; validation-selected where indicated\\
Minimum peak separation & 3 pixels\\
Centroid window & 5 pixels\\
Hungarian matching tolerance & 3 pixels\\
Edge image-border exclusion & 10 pixels, distinct from material boundary\\
Experimental DoG widths & 0.8 and 8 source pixels\\
Experimental target sampling & Approximately 0.025 nm per inference pixel\\
Experimental normalization & Production percentile normalization (1, 99.8 percentiles)\\
Experimental agreement radii & 0.04, 0.06, 0.08 nm\\
Synthetic scale reference & 0.106223 angstrom per pixel, explicit convention\\
Tiled inference & 256-pixel tiles, 96-pixel overlap, Hann weights\\
Gaussian refinement & 11-pixel windows; isotropic Gaussian plus planar background\\
\hline\end{tabular}\end{table}
'''
    (study.doc/'sections/tables.tex').write_text(text)


def additional_panels(study):
    path=study.out/'data/fig-S04.json'
    if path.exists():
        data=json.loads(path.read_text())['data']
        fig,axes=plt.subplots(1,3,figsize=(10,3.5),constrained_layout=True)
        for ax,family in zip(axes,FAMILIES):
            rr=data[family]['paired_F1_difference']
            mean=np.array([rr[m][0] for m in ('square','hexagonal')]); lo=np.array([rr[m][1] for m in ('square','hexagonal')]); hi=np.array([rr[m][2] for m in ('square','hexagonal')])
            ax.errorbar(range(2),mean,yerr=[mean-lo,hi-mean],fmt='o',capsize=5,color=COLORS['random'])
            ax.axhline(0,color='black',lw=.7); ax.set_xticks(range(2),['vs Square','vs Hex']); ax.set_title(family); ax.set_ylabel('Paired F1 difference: Blob-Net - comparator'); ax.grid(axis='y',alpha=.2)
        study.save(4,fig,{k:v['paired_F1_difference'] for k,v in data.items()},'Mean paired per-image F1 differences with 95% percentile bootstrap intervals (2,000 resamples of images, preserving model pairing). Positive differences favor Blob-Net. These quantify image sampling uncertainty for the fixed archived checkpoints, not variability across training runs.', '-b')
    path=study.out/'data/fig-S06.json'
    if path.exists():
        data=json.loads(path.read_text())['data'];fig,axes=plt.subplots(1,3,figsize=(10,3.5),constrained_layout=True)
        for ax,(family,values) in zip(axes,data.items()):
            bins=np.array(values['bins_px']);centers=(bins[:-1]+bins[1:])/2
            for m in FAMILIES:
                rms=[np.nan if v is None else v for v in values['models'][m]['rmse']]
                ax.plot(centers,rms,'o-',color=COLORS[m],label=LABELS[m])
            ax.axvline(0,color='black',ls=':');ax.set_title(family);basic_axes(ax,'Signed boundary distance (px)','Matched RMSE (px)')
        axes[0].legend(fontsize=7)
        study.save(6,fig,data,'Localization error versus signed material-boundary distance for the same edge experiment as S6. Bins without matched atoms are undefined and are left blank. RMSE is conditional on detection; it should be read together with the recall and false-positive curves.', '-b')
    path=study.out/'data/fig-S09.json'
    if path.exists():
        data=json.loads(path.read_text())['data'];fig,axes=plt.subplots(2,2,figsize=(9,7),constrained_layout=True)
        for row,protocol in enumerate(('resampled','regenerated')):
            for m in FAMILIES:
                rr=[r for r in data if r['model']==m and r['protocol']==protocol and r['mode']=='fixed']
                axes[row,0].plot([r['factor'] for r in rr],[r['metrics']['precision'] for r in rr],'o-',color=COLORS[m],label=LABELS[m])
                axes[row,1].plot([r['factor'] for r in rr],[r['metrics']['tp'] for r in rr],'o-',color=COLORS[m])
            for col,metric in enumerate(('Precision','Matched atoms (pooled TP)')):
                basic_axes(axes[row,col],'Physical pixel-size factor',metric);axes[row,col].set_title(protocol)
        axes[0,0].legend(fontsize=7)
        study.save(9,fig,data,'Precision and numbers of successful matches under the fixed-threshold, fixed-pixel-tolerance protocol from S9. These panels expose survivor selection behind a low matched-subset RMSE at poor recall. The regenerated protocol may change actual atom counts through packing constraints.', '-b')



    path=study.out/'data/fig-S12.json'
    if path.exists():
        histories=json.loads(path.read_text())['data']['histories']
        regimes=['square','hexagonal','random','wide_random','dense_random']
        fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
        for ax,regime in zip(axes.flat,regimes):
            runs=sorted((k,v) for k,v in histories.items() if k.startswith(regime+'/'))
            for color_index,(key,hist) in enumerate(runs):
                color=plt.get_cmap('tab10')(color_index)
                ax.plot(range(1,len(hist['train'])+1),hist['train'],color=color,alpha=.65)
                ax.plot(range(1,len(hist['validation'])+1),hist['validation'],'--',color=color,label=key.split('/')[-1])
            ax.set_title(regime);basic_axes(ax,'Epoch','Loss');ax.legend(title='Seed',fontsize=7,title_fontsize=7)
        axes[1,2].axis('off');axes[1,2].text(0,.9,'Solid: training\nDashed: validation\nMatching colors identify the same seed.\n\nExact settings, histories and checkpoint\nhashes accompany every run.',va='top')
        study.save(12,fig,histories,'All new supplemental training histories. Solid curves show training and dashed curves validation; each initialization seed has the same color in both. Generated datasets are held fixed within each regime. Selected checkpoints and timing are recorded per run.', '-b')


def write_document(study):
    doc=study.doc; completed=[]; sections=[]; writeup=[]
    for number in range(1,14):
        if number == 2: continue  # Omitted from the SI at author request.
        paths=sorted((study.out/'data').glob(f'fig-S{number:02d}*.json'), key=lambda p: (len(p.stem),p.stem))
        if number in (4, 5, 8): paths = paths[:1]
        if not paths: continue
        completed.append(number)
        primary=json.loads(paths[0].read_text())
        result_note=figure_result_note(number,primary['data'])
        writeup.append(f'## Figure S{number}: {TITLES[number]}\n')
        if result_note:
            writeup.append(result_note+'\n')
        for j,path in enumerate(paths):
            payload=json.loads(path.read_text()); caption=payload['caption']; stem=path.stem
            source_figure=study.out/'figures'/f'{stem}.png'
            target_figure=doc/'figures'/f'{stem}.png'
            if source_figure.resolve()!=target_figure.resolve(): shutil.copy2(source_figure,target_figure)
            # Each non-floating figure is followed by its caption and notes;
            # the next sheet starts on a new page.
            sections.append(r'\clearpage'+'\n')
            if j==0:
                sections.append(r'\setcounter{section}{'+str(number-1)+'}\n')
                sections.append(r'\section{'+latex_escape(TITLES[number])+'}\n')
            sections.append(r'\noindent\makebox[\textwidth]{\includegraphics[width=\textwidth,height=0.62\textheight,keepaspectratio]{'+stem+'.png}}\n')
            sections.append(r'\begingroup\captionsetup{type=figure}'+'\n')
            if j==0:
                sections.append(r'\setcounter{figure}{'+str(number-1)+'}\n'+r'\caption{'+latex_escape(caption)+'}\n')
            else:
                sections.append(r'\caption*{\textbf{Figure S'+str(number)+', continued.} '+latex_escape(caption)+'}\n')
            sections.append(r'\endgroup'+'\n')
            if j==0 and result_note: sections.append(latex_escape(result_note)+'\n')
            if j == 0: writeup.append(caption.split('. ')[0]+'.\n')
    (doc/'sections/supplementary_figures.tex').write_text('\n'.join(sections))
    table_rows=[]
    for family in FAMILIES:
        metrics=json.loads((study.args.model_dir/family/'training_metrics.json').read_text())
        table_rows.append(f"{LABELS[family]} & {metrics['epochs_completed']} & {metrics['best_epoch']} & {metrics['best_validation_loss']:.5f} & {metrics['trainable_parameters']:,} \\\\")
    methods=r'''\section*{Scope and reproducibility}
This supplement accompanies \emph{Geometry-Agnostic Atom Localization from Aperiodic Training Data with Blob-Net}. It documents recovered training records and new controlled analyses. All supplemental figures are generated by a separate script, with outputs isolated from the main manuscript. Source data include exact settings, image-level metrics, checkpoint hashes and source-code hashes.

The original manuscript-model checkpoints are read from the archived Square-Net, Hex-Net and Blob-Net runs. The authors report that original data generation and training were performed on a different computer. Transferred checkpoints, configurations and histories document those runs; the small local dataset copies must not be interpreted as their training sets. Locally available dataset manifests do not establish a complete chain of provenance to those checkpoints. Accordingly, the dataset audit reports the available files as a local snapshot, while new synthetic evaluations explicitly identify their configurations and independent seeds. New compact-image training runs are supplementary controls, not replicas of the original 512-pixel training experiments.

\section*{Image formation and evaluation}
Synthetic inputs sum isotropic Gaussian atomic features; targets use a pixelwise maximum of fixed-width Gaussians. Background, blur, Poisson sampling and additive read noise follow the production renderer. The Poisson parameter controls peak-intensity count scale rather than total electron dose. No multislice scattering or detector calibration is asserted.

Unless otherwise stated, localization uses a threshold of 0.35 relative to the heatmap maximum, a 3-pixel minimum peak separation, a 5-pixel centroid window and a 3-pixel Hungarian matching tolerance. Localization RMSE is conditional on true-positive matches. Missing matches therefore cannot be interpreted as zero localization error. Paired comparisons use the same images for each model. Confidence intervals resample images rather than treating atomic columns as independent experiments.

Validation seeds begin at 7,000,000 and test seeds at 8,000,000. Edge validation and test realizations use separate 9,000,000 and 10,000,000 ranges. Further controlled experiments use the explicit seed offsets in their source data. Threshold selection uses validation data; test-set curves are descriptive sensitivity analyses. The main Figure 2 threshold-selection procedure is disclosed in S5, including its use of a Blob-Net advantage criterion on the displayed examples.

\section*{Recovered model histories}
\begin{table}[!htbp]\centering\small
\caption{Archived manuscript training records. The selected epoch is the minimum validation-loss checkpoint.}
\begin{tabular}{lrrrr}\hline
Model & Epochs & Best epoch & Validation loss & Parameters\\\hline
'''+ '\n'.join(table_rows)+r'''
\hline\end{tabular}\end{table}

All three archived models use filter widths [32, 64, 128, 256], bottleneck dropout 0.2, Adam learning rate 0.001 and batch size 32. The maximum is 20 epochs, with patience six and minimum improvement 0.0005 for the stopping counter. A checkpoint is saved on any new validation minimum. The loss combines equal weights of global MSE and peak-weighted MSE (target threshold 0.1; peak weight multiplier 5). Their combined effect is a weight of three on peak pixels relative to one elsewhere, averaged over all pixels.

\section*{Experimental interpretation and limits}
Experimental comparisons report detector agreement and sensitivity, not independently measured accuracy. No manual annotations were supplied or invented. Neither absence of visible missed columns nor inter-model agreement establishes perfect detection.

The bottleneck receptive field is 68 pixels. Decoder operations enlarge output support; the bottleneck value alone does not bound all output dependence. Context interventions and synthetic scale sweeps characterize specified tests and do not establish that the model is free from all dataset priors.
'''
    (doc/'sections/methods.tex').write_text(methods)
    write_tables(study)
    header=r'''\documentclass[11pt]{article}
\usepackage[letterpaper,margin=1in]{geometry}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{graphicx}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{xspace}
\usepackage{longtable}
\usepackage[hidelinks]{hyperref}
\graphicspath{{figures/}}
\setlength{\parindent}{0.5in}
\setlength{\parskip}{0pt}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thesection}{S\arabic{section}}
\title{Supplementary Information\\[0.5em]\large Geometry-Agnostic Atom Localization from Aperiodic Training Data with Blob-Net}
\author{Austin C. Houston \and Darel Pates \and Elizabeth Heon \and Kai Xiao \and Gerd Duscher}
\date{}
\begin{document}
\maketitle
\IfFileExists{sections/utkarsh_gold_tio2.tex}{\input{sections/utkarsh_gold_tio2}}{}
\input{sections/methods}
\input{sections/supplementary_figures}
\input{sections/tables}
\end{document}
'''
    (doc/'supplementary_information.tex').write_text(header)
    missing=[n for n in range(1,14) if n not in completed]
    (doc/'SI_writeup.md').write_text('# Blob-Net supplemental information\n\n'+
      'This is a working SI with measured results, not a claim that every main-text assertion has been confirmed. '+
      (f'All {len(completed)} requested figure groups have been generated.\n\n' if not missing else f'Completed figure groups: {completed}. Remaining: {missing}.\n\n')+
      '\n'.join(writeup)+'\n## Outstanding scientific provenance\n\nOriginal data generation and training took place on another computer, as confirmed by the author. The complete original NPZ datasets are not available in this local workspace. Experimental material identity and independent annotations require author input. New compact training controls do not replace full-scale repeats. The supplement must be read alongside these limitations.\n')
    shutil.copy2(study.out/'run_manifest.json',doc/'run_manifest.json')
    for filename in ('author_source_context.json','test_results.txt'):
        if (study.out/filename).exists(): shutil.copy2(study.out/filename,doc/filename)
    for folder_name in ('source_snapshots','invocations'):
        if (study.out/folder_name).exists(): shutil.copytree(study.out/folder_name,doc/folder_name,dirs_exist_ok=True)
    if (study.out/'initial_run_manifest.json').exists(): shutil.copy2(study.out/'initial_run_manifest.json',doc/'initial_run_manifest.json')
    shutil.copy2(Path(__file__),doc/'make_supplemental_figures.py')
    source=doc/'source_data'; source.mkdir(exist_ok=True)
    for p in (study.out/'data').glob('*.json'): shutil.copy2(p,source/p.name)
    (doc/'README.md').write_text('# Blob-Net SI\n\nStandalone LaTeX companion matching the main manuscript typography and authors.\n\nFrom the BlobNet code repository, run `uv run blobnet-reproduce-publication --target si --device auto --compile-latex`. Generated numerical data, figures, source snapshots, and the assembled document remain under `outputs/supplemental_information_20260913`.\n')
    print('Document updated:',doc,flush=True)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-at',type=int,default=1,choices=range(1,14),help='First figure when running all; earlier cached results remain available.')
    parser.add_argument('--figure',default='all',choices=['all','document','gold']+[str(i) for i in range(1,14)])
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/supplemental_information_20260913')
    parser.add_argument('--document-dir',type=Path,default=ROOT/'outputs/supplemental_information_20260913/document')
    parser.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/manuscript_models')
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--samples',type=int,default=256)
    parser.add_argument('--validation-samples',type=int,default=8)
    parser.add_argument('--replicates',type=int,default=16)
    parser.add_argument('--training-size',type=int,default=128)
    parser.add_argument('--training-samples',type=int,default=512)
    parser.add_argument('--training-validation-samples',type=int,default=128)
    parser.add_argument('--training-test-samples',type=int,default=128)
    parser.add_argument('--training-epochs',type=int,default=20)
    parser.add_argument('--training-seeds',type=int,nargs='+',default=[0,1,2])
    args=parser.parse_args()
    for name in ('samples','validation_samples','replicates','training_samples','training_validation_samples','training_test_samples','training_epochs'):
        if getattr(args,name)<1: parser.error(f'{name} must be positive')
    if args.training_size<64 or args.training_size%8: parser.error('training-size must be >=64 and divisible by 8')
    return args


def main():
    args=parse_args(); study=Study(args)
    numbers=([number for number in range(args.start_at,14) if number != 2]
             if args.figure=='all' else [] if args.figure in {'document','gold'} else [int(args.figure)])
    if args.figure in {'all', 'gold'}:
        print('Starting gold/TiO2 supplemental figure', flush=True)
        make_gold_tio2_figure(study)
    for number in numbers:
        print(f'Starting S{number}: {TITLES[number]}',flush=True)
        getattr(study,f's{number}')()
        write_document(study)
    additional_panels(study)
    write_document(study)
    return 0


if __name__=='__main__': raise SystemExit(main())
