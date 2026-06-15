from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import quote
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from GombNet.networks import build_unet


TEM_IMAGENET_REPO = 'https://github.com/xinhuolin/TEM-ImageNet-v1.3'
ATOMAI_PRETRAINED_URLS = {
    'G_MD': 'https://github.com/ziatdinovmax/atomai/blob/master/pretrained/G_MD.tar?raw=true',
    'BFO': 'https://github.com/ziatdinovmax/atomai/blob/master/pretrained/bfo.tar?raw=true',
}
ATOMSEGNET_WEIGHT_FILES = {
    'circularMask': 'circularMask.pth',
    'circularMask_mse_beta': 'circularMask_mse_beta.pth',
    'circularMask_chi10_beta': 'circularMask_chi10_beta.pth',
    'circularMask_chi100_beta': 'circularMask_chi100_beta.pth',
    'guassianMask': 'guassianMask.pth',
    'gaussianMask+': 'gaussianMask+.pth',
    'denoise': 'denoise.pth',
    'denoise&bgremoval': 'denoise&bgremoval.pth',
    'denoise&bgremoval&superres': 'denoise&bgremoval&superres.pth',
    'denoise&airysuperrez_beta': 'denoise&airysuperrez_beta.pth',
    'Gen1-noNoiseNoBackgroundSuperresolution': 'Gen1-noNoiseNoBackgroundSuperresolution.pth',
    'Gen1-circularMask': 'Gen1-circularMask.pth',
    'Gen1-gaussianMask': 'Gen1-gaussianMask.pth',
    'Gen1-noBackgroundNonoise': 'Gen1-noBackgroundNonoise.pth',
    'Gen1-noNoise': 'Gen1-noNoise.pth',
}
ATOMSEGNET_DEFAULT_LOCALIZERS = [
    'gaussianMask+',
    'guassianMask',
    'circularMask',
    'Gen1-gaussianMask',
    'Gen1-circularMask',
]


class ModelCandidate:
    name: str
    family: str

    def setup(self) -> None:
        raise NotImplementedError

    def predict_heatmap(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def download_file(url: str, path: Path, allow_download: bool) -> Path:
    if path.exists():
        return path
    if not allow_download:
        raise FileNotFoundError(f'{path} is missing. Re-run with --download-models to fetch it.')
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url} -> {path}', flush=True)
    urlretrieve(url, path)
    return path


def postprocess_heatmap(heatmap: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    heatmap = np.nan_to_num(np.asarray(heatmap, dtype=np.float32).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
    if heatmap.ndim != 2:
        raise ValueError(f'Expected a 2D heatmap after squeezing, got {heatmap.shape}')
    if heatmap.shape != tuple(output_shape):
        pil = Image.fromarray(heatmap.astype(np.float32), mode='F')
        heatmap = np.asarray(pil.resize((output_shape[1], output_shape[0]), resample=Image.BILINEAR), dtype=np.float32)
    heatmap = heatmap - min(float(heatmap.min()), 0.0)
    peak = float(heatmap.max())
    return (heatmap / peak if peak > 0 else heatmap).astype(np.float32)


class BlobNetCandidate(ModelCandidate):
    def __init__(self, checkpoint_path: Path, device: torch.device, num_filters: Sequence[int], dropout: float) -> None:
        self.name = 'blobnet_unet'
        self.family = 'blobnet'
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_filters = list(num_filters)
        self.dropout = float(dropout)
        self.model: Optional[nn.Module] = None

    def setup(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f'BlobNet checkpoint not found: {self.checkpoint_path}')
        self.model = build_unet(input_channels=1, num_classes=1, num_filters=self.num_filters, dropout=self.dropout)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        self.model.to(self.device).eval()

    def predict_heatmap(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('BlobNetCandidate.setup() must be called before prediction.')
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None].to(self.device)
        with torch.no_grad():
            output = torch.sigmoid(self.model(tensor))[0, 0].detach().cpu().numpy()
        return postprocess_heatmap(output, image.shape)


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class NestedUNet(nn.Module):
    def __init__(self, in_channels: int = 1, filters: Sequence[int] = (32, 64, 128, 256, 512)) -> None:
        super().__init__()
        f = list(filters)
        self.blocks = nn.ModuleDict({
            '0_0': VGGBlock(in_channels, f[0], f[0]), '1_0': VGGBlock(f[0], f[1], f[1]),
            '2_0': VGGBlock(f[1], f[2], f[2]), '3_0': VGGBlock(f[2], f[3], f[3]),
            '4_0': VGGBlock(f[3], f[4], f[4]), '0_1': VGGBlock(f[0] + f[1], f[0], f[0]),
            '1_1': VGGBlock(f[1] + f[2], f[1], f[1]), '2_1': VGGBlock(f[2] + f[3], f[2], f[2]),
            '3_1': VGGBlock(f[3] + f[4], f[3], f[3]), '0_2': VGGBlock(f[0] * 2 + f[1], f[0], f[0]),
            '1_2': VGGBlock(f[1] * 2 + f[2], f[1], f[1]), '2_2': VGGBlock(f[2] * 2 + f[3], f[2], f[2]),
            '0_3': VGGBlock(f[0] * 3 + f[1], f[0], f[0]), '1_3': VGGBlock(f[1] * 3 + f[2], f[1], f[1]),
            '0_4': VGGBlock(f[0] * 4 + f[1], f[0], f[0]),
        })
        self.final = nn.Conv2d(f[0], 1, kernel_size=1)

    @staticmethod
    def pool(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, 2, 2)

    @staticmethod
    def up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.blocks
        x0_0 = b['0_0'](x); x1_0 = b['1_0'](self.pool(x0_0)); x0_1 = b['0_1'](torch.cat([x0_0, self.up(x1_0)], 1))
        x2_0 = b['2_0'](self.pool(x1_0)); x1_1 = b['1_1'](torch.cat([x1_0, self.up(x2_0)], 1)); x0_2 = b['0_2'](torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x3_0 = b['3_0'](self.pool(x2_0)); x2_1 = b['2_1'](torch.cat([x2_0, self.up(x3_0)], 1)); x1_2 = b['1_2'](torch.cat([x1_0, x1_1, self.up(x2_1)], 1)); x0_3 = b['0_3'](torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x4_0 = b['4_0'](self.pool(x3_0)); x3_1 = b['3_1'](torch.cat([x3_0, self.up(x4_0)], 1)); x2_2 = b['2_2'](torch.cat([x2_0, x2_1, self.up(x3_1)], 1)); x1_3 = b['1_3'](torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        return torch.tanh(self.final(b['0_4'](torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))))


class ClassicUNet(nn.Module):
    def __init__(self, colordim: int = 1) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(colordim, 64, 3, padding=1); self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64); self.bn1_2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1); self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128); self.bn2_2 = nn.BatchNorm2d(128)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1); self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.upconv4 = nn.Conv2d(256, 128, 1); self.bn4 = nn.BatchNorm2d(128); self.bn4_1 = nn.BatchNorm2d(256); self.bn4_2 = nn.BatchNorm2d(256); self.bn4_out = nn.BatchNorm2d(256)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=1); self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv7 = nn.Conv2d(128, 64, 1); self.bn7 = nn.BatchNorm2d(64); self.bn7_1 = nn.BatchNorm2d(128); self.bn7_2 = nn.BatchNorm2d(128); self.bn7_out = nn.BatchNorm2d(128)
        self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1); self.conv9_2 = nn.Conv2d(64, 64, 3, padding=1); self.conv9_3 = nn.Conv2d(64, colordim, 1)
        self.bn9_1 = nn.BatchNorm2d(64); self.bn9_2 = nn.BatchNorm2d(64); self.bn9_3 = nn.BatchNorm2d(colordim); self.bn9 = nn.BatchNorm2d(colordim)
        self.maxpool = nn.MaxPool2d(2, 2); self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.bn1_2(self.conv1_2(F.relu(self.bn1_1(self.conv1_1(x))))))
        x2 = F.relu(self.bn2_2(self.conv2_2(F.relu(self.bn2_1(self.conv2_1(self.maxpool(x1)))))))
        xup = F.relu(self.bn4_2(self.conv4_2(F.relu(self.bn4_1(self.conv4_1(self.maxpool(x2)))))))
        xup = self.bn4_out(torch.cat((x2, self.bn4(self.upconv4(self.upsample(xup)))), 1))
        xup = F.relu(self.bn7_2(self.conv7_2(F.relu(self.bn7_1(self.conv7_1(xup))))))
        xup = self.bn7_out(torch.cat((x1, self.bn7(self.upconv7(self.upsample(xup)))), 1))
        return torch.sigmoid(self.bn9(F.relu(self.conv9_3(F.relu(self.bn9_2(self.conv9_2(F.relu(self.bn9_1(self.conv9_1(xup))))))))))


def strip_dataparallel_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace('module.', '', 1): value for key, value in state_dict.items()}


def translate_nested_unet_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    layer_map = {'conv1': 'layers.0', 'bn1': 'layers.1', 'conv2': 'layers.3', 'bn2': 'layers.4'}
    translated = {}
    for key, value in strip_dataparallel_prefix(state_dict).items():
        if not key.startswith('conv'):
            translated[key] = value
            continue
        block_name, layer_name, suffix = key.split('.', 2)
        translated[f'blocks.{block_name.removeprefix("conv")}.{layer_map[layer_name]}.{suffix}'] = value
    return translated


def pad_to(tensor: torch.Tensor, size: int | None = None, multiple: int | None = None) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    if size is None:
        pad_h = (int(multiple or 1) - height % int(multiple or 1)) % int(multiple or 1)
        pad_w = (int(multiple or 1) - width % int(multiple or 1)) % int(multiple or 1)
        size_h, size_w = height + pad_h, width + pad_w
    else:
        size_h = size_w = max(size, height, width)
        if size_h > size:
            size_h = size_w = int(2 ** math.ceil(math.log2(size_h)))
    top = (size_h - height) // 2; bottom = size_h - height - top; left = (size_w - width) // 2; right = size_w - width - left
    return (F.pad(tensor, (left, right, top, bottom)) if top or bottom or left or right else tensor), (top, bottom, left, right)


def unpad(tensor: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    top, bottom, left, right = padding
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    return tensor[..., top:height - bottom if bottom else height, left:width - right if right else width]


class AtomSegNetCandidate(ModelCandidate):
    def __init__(self, model_name: str, cache_dir: Path, allow_download: bool, device: torch.device, iterations: int = 1) -> None:
        self.name = f'atomsegnet_{model_name}'
        self.family = 'atomsegnet'
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.allow_download = allow_download
        self.device = device
        self.iterations = max(1, int(iterations))
        self.model: Optional[nn.Module] = None
        self.is_gen1 = model_name.startswith('Gen1-')

    def setup(self) -> None:
        weight_file = ATOMSEGNET_WEIGHT_FILES.get(self.model_name)
        if weight_file is None:
            raise ValueError(f'Unknown AtomSegNet model {self.model_name!r}. Known models: {sorted(ATOMSEGNET_WEIGHT_FILES)}')
        weight_path = download_file('https://raw.githubusercontent.com/xinhuolin/AtomSegNet/master/model_weights/' + quote(weight_file), self.cache_dir / 'atomsegnet' / weight_file, self.allow_download)
        self.model = NestedUNet() if self.is_gen1 else ClassicUNet()
        state_dict = torch.load(weight_path, map_location='cpu')
        if not isinstance(state_dict, dict):
            raise TypeError(f'Expected a state_dict in {weight_path}, got {type(state_dict).__name__}')
        self.model.load_state_dict(translate_nested_unet_keys(state_dict) if self.is_gen1 else strip_dataparallel_prefix(state_dict))
        self.model.to(self.device).eval()

    def predict_heatmap(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('AtomSegNetCandidate.setup() must be called before prediction.')
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None].to(self.device)
        with torch.no_grad():
            if self.is_gen1:
                padded, padding = pad_to(tensor, size=128)
                output = unpad(self.model(padded), padding)
            else:
                padded, padding = pad_to(tensor, multiple=4)
                output = padded
                for _ in range(self.iterations):
                    output = self.model(output)
                output = unpad(output, padding)
        return postprocess_heatmap(output[0, 0].detach().cpu().numpy(), image.shape)


def atomai_prediction_to_heatmap(prediction: Any, image_shape: tuple[int, int], nb_classes: Optional[int]) -> np.ndarray:
    prediction = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
    prediction = np.asarray(prediction, dtype=np.float32).squeeze()
    if prediction.ndim == 2:
        return postprocess_heatmap(prediction, image_shape)
    if prediction.ndim != 3:
        raise ValueError(f'Could not convert AtomAI prediction with shape {prediction.shape} to a heatmap.')
    channels = prediction if prediction.shape[0] <= 8 and prediction.shape[1:] == image_shape else np.moveaxis(prediction, -1, 0)
    heatmap = np.max(channels[1:], axis=0) if nb_classes and nb_classes > 1 and channels.shape[0] > 1 else channels[0]
    return postprocess_heatmap(heatmap, image_shape)


class AtomAICandidate(ModelCandidate):
    def __init__(self, model_name: str, cache_dir: Path, allow_download: bool, device: torch.device) -> None:
        self.name = f'atomai_{model_name}'
        self.family = 'atomai'
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.allow_download = allow_download
        self.device = device
        self.model: Any = None
        self.nb_classes: Optional[int] = None

    def setup(self) -> None:
        try:
            from atomai.models import load_model
        except ImportError as exc:
            raise ImportError('AtomAI is not installed. Install atomai to enable AtomAI pretrained comparisons.') from exc
        url = ATOMAI_PRETRAINED_URLS.get(self.model_name)
        if url is None:
            raise ValueError(f'Unknown AtomAI pretrained model {self.model_name!r}.')
        self.model = load_model(str(download_file(url, self.cache_dir / 'atomai' / f'{self.model_name}.tar', self.allow_download)))
        self.nb_classes = int(getattr(self.model, 'nb_classes', 1))
        target_device = torch.device('cuda') if self.device.type == 'cuda' else torch.device('cpu')
        if hasattr(self.model, 'net'):
            self.model.net.to(target_device).eval()
        if hasattr(self.model, 'device'):
            self.model.device = str(target_device)

    def predict_heatmap(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('AtomAICandidate.setup() must be called before prediction.')
        prediction = self.model.predict(np.asarray(image, dtype=np.float32), compute_coords=False, norm=True, num_batches=1, verbose=False)
        return atomai_prediction_to_heatmap(prediction, image.shape, self.nb_classes)


def maybe_download_tem_imagenet(dataset_dir: Path, should_download: bool) -> None:
    if dataset_dir.exists() or not should_download:
        return
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f'Cloning TEM-ImageNet v1.3 into {dataset_dir}', flush=True)
    subprocess.run(['git', 'clone', '--depth', '1', TEM_IMAGENET_REPO, str(dataset_dir)], check=True)


def load_tem_imagenet_arrays(dataset_dir: Path, max_samples: int, coordinate_order: str) -> list[dict[str, Any]]:
    image_dir = dataset_dir / 'image'
    position_dir = dataset_dir / 'position'
    if not image_dir.exists() or not position_dir.exists():
        raise FileNotFoundError(f'Expected TEM-ImageNet image/ and position/ directories under {dataset_dir}')
    samples: list[dict[str, Any]] = []
    for image_path in sorted(image_dir.glob('*.png'))[: int(max_samples) if max_samples > 0 else None]:
        position_path = position_dir / f'{image_path.stem}.txt'
        if not position_path.exists():
            continue
        image = np.asarray(Image.open(image_path).convert('L'), dtype=np.float32)
        image = image - float(image.min())
        peak = float(image.max())
        coords = np.asarray(np.loadtxt(position_path, dtype=np.float32), dtype=np.float32).reshape(-1, 2)
        samples.append({
            'sample_id': f'tem_{image_path.stem}',
            'image': (image / peak if peak > 0 else image).astype(np.float32),
            'coordinates_yx': coords[:, [1, 0]] if coordinate_order == 'xy' else coords,
            'metadata': {'image_path': str(image_path), 'position_path': str(position_path)},
        })
    return samples
