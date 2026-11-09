"""Portable, explicitly three-channel data access. No paths to the old project."""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode('utf-8')).hexdigest()


def safe_child(root, relative):
    root = Path(root).resolve()
    child = (root / relative).resolve()
    if child != root and root not in child.parents:
        raise ValueError('Path escapes data directory: ' + str(relative))
    return child


def validate_channel_map(channel_map):
    bands = tuple(int(v) for v in channel_map)
    if len(bands) != 3 or any(v not in range(1, 6) for v in bands):
        raise ValueError('Every neural-network input must contain exactly three channels in B1..B5')
    if not (len(set(bands)) == 1 or len(set(bands)) == 3 and bands == tuple(sorted(bands))):
        raise ValueError('Use a triplicated single band or three distinct ascending bands')
    return bands


def d4(array, transform):
    """Eight equiprobable square symmetries; H,W are the first two axes."""
    if transform not in range(8):
        raise ValueError('D4 transform must be 0..7')
    result = np.rot90(array, transform % 4, axes=(0, 1))
    if transform >= 4:
        result = result[:, ::-1]
    return np.ascontiguousarray(result)


class SceneStore:
    def __init__(self, data_root, manifest_path, support_root, geometry_root=None,
                 verify_hashes=False, cache_sites=2):
        self.data_root = Path(data_root)
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8-sig'))
        support_path = Path(support_root)
        self.support_path = support_path if support_path.is_file() else support_path / 'manifest.json'
        self.support_root = self.support_path.parent
        self.support = json.loads(self.support_path.read_text(encoding='utf-8-sig'))
        self.geometry_root = Path(geometry_root) if geometry_root else self.support_root / 'geometry'
        self.sites = sorted(self.manifest['scenes'], key=lambda s: int(s[1:]))
        if set(self.sites) != set(self.support['scenes']):
            raise ValueError('Scene and support identities differ')
        expected_manifest = self.support.get('dataset_sha256')
        if expected_manifest and expected_manifest != sha256(self.manifest_path):
            raise ValueError('Support is bound to a different manifest; override data_root, do not rewrite manifest')
        self.masks = {}
        self.geometry_masks = {}
        for site in self.sites:
            row = self.support['scenes'][site]
            path = safe_child(self.support_root, row['mask'])
            if row.get('mask_sha256') and sha256(path) != row['mask_sha256']:
                raise ValueError('Support mask checksum differs: ' + site)
            with Image.open(path) as image:
                mask = np.asarray(image).copy() > 0
            if list(mask.shape) != row['shape'] or int(mask.sum()) != row['valid_pixels']:
                raise ValueError('Support dimensions/count differ: ' + site)
            self.masks[site] = mask
            geom_path = self.geometry_root / (site + '.png')
            if geom_path.exists():
                if row.get('geometry_sha256') and sha256(geom_path) != row['geometry_sha256']:
                    raise ValueError('Geometry mask checksum differs: ' + site)
                with Image.open(geom_path) as image:
                    geom = np.asarray(image).copy() > 0
                if geom.shape != mask.shape or np.any(mask & ~geom):
                    raise ValueError('Common support must be contained in geometry: ' + site)
                self.geometry_masks[site] = geom
        self.cache_sites = max(1, int(cache_sites))
        self._cache = OrderedDict()
        # Absolute machine paths are intentionally absent from scientific identity.
        self.identity = stable_hash({'scenes': self.manifest['scenes'], 'support': self.support})
        self.support_identity = stable_hash(self.support)
        if verify_hashes:
            self.verify_files()

    def verify_files(self):
        report = []
        for site in self.sites:
            for name, row in self.manifest['scenes'][site].items():
                path = safe_child(self.data_root, row['path'])
                if sha256(path) != row['sha256']:
                    raise ValueError('Image checksum differs: ' + site + '/' + name)
                with Image.open(path) as image:
                    arr = np.asarray(image)
                    if list(arr.shape) != row['shape'] or str(arr.dtype) != row['dtype']:
                        raise ValueError('Image shape/type differs: ' + str(path))
                report.append({'site': site, 'modality': name, 'sha256': row['sha256']})
        return report

    def raw(self, site):
        if site not in self._cache:
            records = self.manifest['scenes'][site]
            arrays = {}
            for name in ['RGB'] + ['B%d' % b for b in range(1, 6)]:
                with Image.open(safe_child(self.data_root, records[name]['path'])) as image:
                    arrays[name] = np.asarray(image).copy()
                if arrays[name].dtype != np.uint8:
                    raise ValueError('This frozen release expects uint8, not implicit per-image scaling')
            shape = self.masks[site].shape
            if arrays['RGB'].shape != shape + (3,) or any(arrays['B%d' % b].shape != shape for b in range(1, 6)):
                raise ValueError('Unpaired scene shapes: ' + site)
            self._cache[site] = arrays
            while len(self._cache) > self.cache_sites:
                self._cache.popitem(last=False)
        self._cache.move_to_end(site)
        return self._cache[site]

    def scene(self, site):
        arrays = self.raw(site)
        return {'bands': np.stack([arrays['B%d' % b] for b in range(1, 6)], -1).astype(np.float32) / 255.,
                'rgb': arrays['RGB'].astype(np.float32) / 255., 'mask': self.masks[site]}

    def input_image(self, site, channel_map):
        bands = validate_channel_map(channel_map)
        arrays = self.raw(site)
        return np.stack([arrays['B%d' % b] for b in bands], -1).astype(np.float32) / 255.

    def target(self, site):
        return self.raw(site)['RGB'].astype(np.float32) / 255.

    def crop(self, site, box, channel_map, input_range='zero_one', transform=0):
        import torch
        channels = validate_channel_map(channel_map)
        x, y, w, h = map(int, box)
        shape = self.masks[site].shape
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > shape[1] or y + h > shape[0]:
            raise ValueError('Window outside scene')
        if transform and w != h:
            raise ValueError('D4 augmentation requires square crops')
        arrays = self.raw(site)
        swir = np.stack([arrays['B%d' % b][y:y+h, x:x+w] for b in channels], -1).astype(np.float32) / 255.
        rgb = arrays['RGB'][y:y+h, x:x+w].astype(np.float32) / 255.
        mask = self.masks[site][y:y+h, x:x+w].astype(np.float32)
        if not mask.any():
            raise ValueError('Training/validation crop contains no valid pixels')
        if input_range == 'minus_one_one':
            swir, rgb = swir * 2 - 1, rgb * 2 - 1
        elif input_range != 'zero_one':
            raise ValueError('Unknown network range: ' + input_range)
        return {'x': torch.from_numpy(d4(swir, transform).transpose(2, 0, 1).copy()),
                'y': torch.from_numpy(d4(rgb, transform).transpose(2, 0, 1).copy()),
                'mask': torch.from_numpy(d4(mask, transform)[None].copy())}

    def load_window(self, window, channel_map=(1, 1, 1), input_range='zero_one', transform=0):
        batch = self.crop(window['site'], window['box'], channel_map, input_range, transform)
        return batch['x'], batch['y'], batch['mask']
