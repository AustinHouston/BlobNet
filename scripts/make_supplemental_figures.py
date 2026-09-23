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
from scipy.spatial import cKDTree

from blobnet import synthetic as syn
from blobnet.metrics import extract_subpixel_peak_positions, match_coordinate_sets
from blobnet.networks import build_unet
from scripts import make_manuscript_figures as mainfig

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('square', 'hexagonal', 'random')
LABELS = {'square': 'Square-Net', 'hexagonal': 'Hex-Net', 'random': 'Blob-Net'}
COLORS = {'square': '#3167a5', 'hexagonal': '#c77122', 'random': '#23835e'}
THRESHOLDS = np.array([.1, .2, .3, .35, .45, .55, .65, .73, .785, .85, .9])
TITLES = {
 1: 'Gold implanted in titanium dioxide',
 2: 'Square image formation and target construction',
 3: 'Hexagonal image formation and target construction',
 4: 'Random image formation and target construction',
 5: 'Training histories and checkpoint selection',
 6: 'Paired cross-geometry generalization',
 7: 'Synthetic edges and boundary-localized errors',
 8: 'Experimental preprocessing and disagreement atlas',
 9: 'Image sampling and scale sensitivity',
 10: 'Theoretical receptive field',
 11: 'Noise robustness and a classical baseline',
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

    stem = 'fig-S01'
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
        return make_gold_tio2_figure(self)

    def _image_formation(self, family, number):
        config = replace(self.configs[family], image_shape=(128,128))
        if family == 'random':
            config = replace(config, min_atoms=65, max_atoms=85)
        else:
            config = replace(config, min_atoms=20)
        rng = np.random.default_rng(99 + number)
        cloud = syn.point_cloud_from_config(config, rng)
        rec = syn.render_atom_image(
            cloud.coordinates, config, rng,
            target_coordinates=cloud.target_coordinates, return_stages=True,
        )
        fig, axes = plt.subplots(3, 3, figsize=(8.5,7.5), constrained_layout=True)
        for axis, (name, array) in zip(axes.flat, rec['stages'].items()):
            show(axis, array, name.replace('_',' '))
        show(axes.flat[8], rec['target'], 'target: max Gaussians', rec['coordinates'])
        fig.suptitle(family.capitalize(), fontsize=12)
        data = dict(
            config=asdict(config), count_scale=rec['count_scale'],
            total_counts=rec['total_counts'], visible=len(rec['coordinates']),
            rendered=len(rec['rendered_coordinates']),
        )
        self.save(number, fig, data, f'{family.capitalize()} geometry: sequential snapshots from the production renderer with unchanged random draws. Each panel uses its own grayscale range to expose weak components. Inputs sum atomic Gaussians; targets take their pixelwise maximum. Examples are 128-pixel explanatory crops.')

    def s2(self):
        return self._image_formation('square', 2)

    def s3(self):
        return self._image_formation('hexagonal', 3)

    def s4(self):
        return self._image_formation('random', 4)

    def s5(self):
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
        self.save(5,fig,records,'Recovered original manuscript-model histories. Vertical lines mark the saved best-validation epochs. All models have 1,927,841 trainable parameters. Absolute losses across geometries are not an accuracy ranking. Stopping patience counts epochs without an improvement exceeding 0.0005, whereas a checkpoint is saved on any new validation minimum. ')

    def s6(self):
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
            print('S6',family, {m: output[family][m]['pooled']['f1'] for m in FAMILIES},flush=True)
        self.save(6,fig,output,f'Paired evaluation of all three archived models on the same {self.args.samples} independently generated 512-pixel images per geometry. Threshold 0.35, 3-pixel matching radius, 3-pixel peak separation, 5-pixel centroid window. Boxes show per-image distributions; pooled metrics and 2,000-resample paired image-bootstrap intervals are saved in JSON. Test seeds begin at 8,000,000 and were not used for calibration. Current configurations define this new experiment; it does not retroactively identify the original training datasets.')

    def edge(self,family,index,validation=False):
        funcs={'mos2':mainfig._make_mos2_edge_record,'sto':mainfig._make_sto_edge_record,'graphene':mainfig._make_graphene_rattled_edge_record}
        return funcs[family]((512,512),(9_000_000 if validation else 10_000_000)+index,(1.15,2.65),total_counts_range=(64.,64.),quiet_background=True)

    def s7(self):
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
        self.save(7,fig,output,f'Construction and boundary-distance diagnostics for {self.args.replicates} independent noise/vacancy/displacement realizations per material, with fixed boundary geometries. Gray sites precede masking, vacancies and displacement; orange sites are retained ground truth. Signed distance is positive inside the ideal material mask; displaced atoms can cross it. FP rates are area-normalized, recall is atom-normalized. Fixed threshold 0.35; 10-pixel image-border exclusion. This tests stochastic reproducibility, not the diversity of all possible edge shapes. Species intensities, widths, masks and displacement rules are recorded in source data.')

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
        stages=['input','enc1','pool1','enc2','pool2','enc3','pool3','bottleneck']
        rf=[1,5,6,14,16,32,36,68]
        fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
        ax.plot(range(len(stages)), rf, 'o-', color='#444444')
        ax.set_xticks(range(len(stages)), stages, rotation=50, ha='right')
        ax.set_ylabel('Theoretical receptive field (px)')
        ax.set_title('Bottleneck RF = 68 px')
        self.save(10, fig, dict(stages=stages, rf=rf),
                  'Theoretical receptive field through the encoder and bottleneck.')

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


def latex_escape(text):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(c,c) for c in str(text))


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


def write_document(study):
    doc=study.doc; completed=[1] if (doc/'figures/fig-S01.png').is_file() else []; sections=[]; writeup=[]
    for number in range(2,12):
        paths=list((study.out/'data').glob(f'fig-S{number:02d}.json'))
        if not paths: continue
        completed.append(number)
        primary=json.loads(paths[0].read_text())
        writeup.append(f'## Figure S{number}: {TITLES[number]}\n')
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
            if j == 0: writeup.append(caption.split('. ')[0]+'.\n')
    (doc/'sections/supplementary_figures.tex').write_text('\n'.join(sections))
    table_rows=[]
    for family in FAMILIES:
        metrics=json.loads((study.args.model_dir/family/'training_metrics.json').read_text())
        table_rows.append(f"{LABELS[family]} & {metrics['epochs_completed']} & {metrics['best_epoch']} & {metrics['best_validation_loss']:.5f} & {metrics['trainable_parameters']:,} \\\\")
    methods=r'''\section*{Scope and reproducibility}
This supplement accompanies \emph{Geometry-Agnostic Atom Localization from Aperiodic Training Data with Blob-Net}. It documents recovered training records and new controlled analyses. All supplemental figures are generated by a separate script, with outputs isolated from the main manuscript. Source data include exact settings, image-level metrics, checkpoint hashes and source-code hashes.

The original manuscript-model checkpoints are read from the archived Square-Net, Hex-Net and Blob-Net runs. The authors report that original data generation and training were performed on a different computer. Transferred checkpoints, configurations and histories document those runs; the small local dataset copies must not be interpreted as their training sets. New synthetic evaluations explicitly identify their configurations and independent seeds.

\section*{Image formation and evaluation}
Synthetic inputs sum isotropic Gaussian atomic features; targets use a pixelwise maximum of fixed-width Gaussians. Background, blur, Poisson sampling and additive read noise follow the production renderer. The Poisson parameter controls peak-intensity count scale rather than total electron dose. No multislice scattering or detector calibration is asserted.

Unless otherwise stated, localization uses a threshold of 0.35 relative to the heatmap maximum, a 3-pixel minimum peak separation, a 5-pixel centroid window and a 3-pixel Hungarian matching tolerance. Localization RMSE is conditional on true-positive matches. Missing matches therefore cannot be interpreted as zero localization error. Paired comparisons use the same images for each model. Confidence intervals resample images rather than treating atomic columns as independent experiments.

Validation seeds begin at 7,000,000 and test seeds at 8,000,000. Edge validation and test realizations use separate 9,000,000 and 10,000,000 ranges. Further controlled experiments use the explicit seed offsets in their source data. Threshold selection uses validation data; test-set curves are descriptive sensitivity analyses.

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
    missing=[n for n in range(1,12) if n not in completed]
    (doc/'SI_writeup.md').write_text('# Blob-Net supplemental information\n\n'+
      'This is a working SI with measured results, not a claim that every main-text assertion has been confirmed. '+
      (f'All {len(completed)} requested figure groups have been generated.\n\n' if not missing else f'Completed figure groups: {completed}. Remaining: {missing}.\n\n')+
      '\n'.join(writeup)+'\n## Outstanding scientific provenance\n\nOriginal data generation and training took place on another computer, as confirmed by the author. The complete original NPZ datasets are not available in this local workspace. Experimental material identity and independent annotations require author input. The supplement must be read alongside these limitations.\n')
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
    parser.add_argument('--start-at',type=int,default=1,choices=range(1,12),help='First figure when running all; earlier cached results remain available.')
    parser.add_argument('--figure',default='all',choices=['all','document','gold']+[str(i) for i in range(1,12)])
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/supplemental_information_20260913')
    parser.add_argument('--document-dir',type=Path,default=ROOT/'outputs/supplemental_information_20260913/document')
    parser.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/manuscript_models')
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--samples',type=int,default=256)
    parser.add_argument('--validation-samples',type=int,default=8)
    parser.add_argument('--replicates',type=int,default=16)
    args=parser.parse_args()
    for name in ('samples','validation_samples','replicates'):
        if getattr(args,name)<1: parser.error(f'{name} must be positive')
    return args


def main():
    args=parse_args(); study=Study(args)
    numbers=(list(range(args.start_at,12)) if args.figure=='all'
             else [] if args.figure=='document' else [1] if args.figure=='gold'
             else [int(args.figure)])
    for number in numbers:
        print(f'Starting S{number}: {TITLES[number]}',flush=True)
        getattr(study,f's{number}')()
        write_document(study)
    write_document(study)
    return 0


if __name__=='__main__': raise SystemExit(main())
