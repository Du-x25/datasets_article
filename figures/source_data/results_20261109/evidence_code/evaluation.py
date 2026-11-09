"""Native-resolution inference, masked metrics and controlled protocol comparisons."""
import json
import hashlib
import inspect
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from .data import SceneStore, sha256, stable_hash, validate_channel_map
from .protocols import atomic_json, coverage, evaluation_mask, full_valid_boxes, grid_boxes, starts

METRIC_VERSION = 'masked-evaluation-v1-native256-stride128-sine2'


@lru_cache(maxsize=None)
def implementation_identity(kind):
    """Independent inference/scoring identities: metric fixes do not rerun models."""
    if kind == 'inference':
        import torch
        functions = [predict_scene, Stitcher, grid_boxes, starts,
                     SceneStore.input_image, validate_channel_map]
        versions = {'numpy':np.__version__, 'torch':torch.__version__}
    elif kind == 'metric':
        import skimage
        import scipy
        functions = [score_scene, full_valid_boxes, grid_boxes, starts, coverage]
        versions = {'numpy':np.__version__, 'scipy':scipy.__version__, 'skimage':skimage.__version__}
    elif kind == 'shift':
        functions = [shift_conditions, shift_bilinear, shift_validity, sensitivity_domain,
                     full_valid_boxes, grid_boxes, starts, coverage]
        versions = {'numpy':np.__version__}
    else:
        raise ValueError('Unknown implementation identity scope')
    return stable_hash({'sources':{f.__qualname__:inspect.getsource(f) for f in functions},
                        'versions':versions})


def _array_identity(array):
    array=np.asarray(array)
    packed=np.packbits(array,axis=None) if array.dtype==bool else np.ascontiguousarray(array)
    return stable_hash({'shape':list(array.shape),'dtype':str(array.dtype),
                        'bytes_sha256':hashlib.sha256(packed.tobytes()).hexdigest()})


def metric_cache_identity(prediction_sha, data_identity, mask, rectangles, lpips_fn, context=None):
    return stable_hash({'prediction_sha256':prediction_sha, 'target_data_identity':data_identity,
                        'mask':_array_identity(mask), 'rectangles':rectangles,
                        'implementation':implementation_identity('metric'),
                        'lpips_identity':getattr(lpips_fn,'identity',None), 'context':context})


def cached_metrics(path, identity, build, allow_reuse=True, context=None):
    """Reuse only complete, intact metrics. Successful reuse leaves mtime intact."""
    path=Path(path)
    if allow_reuse and path.exists():
        try:
            record=json.loads(path.read_text(encoding='utf-8'))
            payload={key:value for key,value in record.items() if key!='payload_sha256'}
            if (record.get('schema')=='scene-metrics-cache-v1' and record.get('identity')==identity
                    and record.get('metrics',{}).get('complete')
                    and record.get('payload_sha256')==stable_hash(payload)):
                return dict(record['metrics']),True
        except (ValueError,OSError):
            pass
    metrics=build()
    payload={'schema':'scene-metrics-cache-v1','identity':identity,'context':context,'metrics':metrics}
    atomic_json(path,dict(payload,payload_sha256=stable_hash(payload)))
    return dict(metrics),False


class EvaluationPaused(RuntimeError):
    """A cooperative stop; already committed scene artifacts remain reusable."""


def _stop_checker(run_dir):
    def check():
        if (Path(run_dir)/'STOP_REQUESTED').exists():
            raise EvaluationPaused('Evaluation paused at a safe tile/metric boundary')
    return check


def atomic_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.save(stream, np.asarray(array, np.float32), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Stitcher:
    def __init__(self, shape, size=256):
        self.total = np.zeros(tuple(shape) + (3,), np.float64)
        self.weight = np.zeros(shape, np.float64)
        axis = np.sin(np.pi * (np.arange(size) + .5) / size) ** 2
        self.kernel = axis[:, None] * axis[None, :]

    def add(self, x, y, prediction):
        prediction = np.asarray(prediction)
        h, w = self.kernel.shape
        if prediction.shape != (h, w, 3) or not np.isfinite(prediction).all():
            raise ValueError('Invalid model output tile')
        self.total[y:y+h, x:x+w] += prediction * self.kernel[..., None]
        self.weight[y:y+h, x:x+w] += self.kernel

    def finish(self, require_full=True):
        valid = self.weight > 0
        if require_full and not valid.all():
            raise ValueError('Inference grid leaves uncovered pixels')
        result = np.zeros_like(self.total, dtype=np.float32)
        result[valid] = (self.total[valid] / self.weight[valid, None]).astype(np.float32)
        return result, valid


def predict_scene(model, image, device='cpu', input_range='zero_one', output_range=None,
                  size=256, stride=128, batch_size=1, boxes=None, stop_check=None):
    """Input image HWC [0,1]. Output raw HWC values in RGB [0,1] convention.

    Values outside the range are preserved for diagnostics; clipping only occurs
    inside scoring and PNG previews. EX4 can pass an explicitly restricted grid.
    """
    import torch
    image = np.asarray(image, np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError('Inference requires finite HWC three-channel input')
    if input_range not in ('zero_one', 'minus_one_one'):
        raise ValueError('Unknown input range')
    output_range = output_range or input_range
    if output_range not in ('zero_one', 'minus_one_one'):
        raise ValueError('Unknown output range')
    tiles = list(boxes) if boxes is not None else grid_boxes(image.shape[:2], size, stride)
    if not tiles:
        raise ValueError('No inference tiles')
    stitcher = Stitcher(image.shape[:2], size)
    previous_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(tiles), batch_size):
                if stop_check:stop_check()
                selected = tiles[start:start+batch_size]
                if any((w, h) != (size, size) for x, y, w, h in selected):
                    raise ValueError('All inference tiles must have the frozen size')
                array = np.stack([image[y:y+h, x:x+w].transpose(2, 0, 1) for x,y,w,h in selected])
                tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
                if input_range == 'minus_one_one':
                    tensor = tensor * 2 - 1
                prediction = model(tensor)
                if not isinstance(prediction, torch.Tensor) or prediction.shape != tensor.shape:
                    raise ValueError('Generator must return BCHW RGB matching input dimensions')
                if output_range == 'minus_one_one':
                    prediction = (prediction + 1) / 2
                output = prediction.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
                for box, predicted in zip(selected, output):
                    stitcher.add(box[0], box[1], predicted)
    finally:
        model.train(previous_training)
    return stitcher.finish(require_full=boxes is None)


def score_scene(prediction, target, mask, lpips_fn=None, rectangles=None,
                require_complete=True, lpips_size=128, stop_check=None):
    """All metrics use the same valid domain; SSIM needs full 11x11 support.

    lpips_fn takes two HWC float32 [0,1] rectangles and returns one float.
    Omitting it is allowed only for an explicitly incomplete functional test.
    """
    p, t, valid = np.asarray(prediction), np.asarray(target), np.asarray(mask, bool)
    if p.shape != t.shape or p.shape != valid.shape + (3,):
        raise ValueError('Prediction/target/support dimensions differ')
    if not np.isfinite(p).all() or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
        raise ValueError('Nonfinite image or target outside [0,1]')
    if not valid.any():
        raise ValueError('Cannot score an empty evaluation domain')
    if stop_check:stop_check()
    clipped = np.clip(p, 0, 1).astype(np.float32, copy=False)
    difference = clipped[valid].astype(np.float64) - t[valid].astype(np.float64)
    mse = float(np.mean(difference ** 2))
    result = {'valid_pixels': int(valid.sum()), 'mse': mse,
              'psnr': float(-10*np.log10(mse)) if mse else 'Infinity',
              'mae': float(np.abs(difference).mean()),
              'out_of_range_channel_fraction': float(((p[valid] < 0) | (p[valid] > 1)).mean())}
    lab_pred = rgb2lab(clipped[valid].astype(np.float64), illuminant='D65', observer='2')
    lab_true = rgb2lab(t[valid].astype(np.float64), illuminant='D65', observer='2')
    delta = deltaE_ciede2000(lab_pred, lab_true)
    result['deltaE00'] = float(delta.mean())
    result['deltaE00_summary'] = {'mean': float(delta.mean()), 'median': float(np.median(delta)),
                                  'p95': float(np.percentile(delta, 95))}
    centres = binary_erosion(valid, structure=np.ones((11, 11), bool), border_value=0)
    result['ssim_centres'] = int(centres.sum())
    result['ssim'] = None
    if centres.any():
        if stop_check:stop_check()
        _, ssim_map = structural_similarity(clipped, t.astype(np.float32), data_range=1,
                                           channel_axis=2, gaussian_weights=True, sigma=1.5,
                                           win_size=11, use_sample_covariance=False, full=True)
        result['ssim'] = float(ssim_map[centres].mean())
    rectangles = list(rectangles) if rectangles is not None else full_valid_boxes(valid, lpips_size, lpips_size)
    area = coverage(valid.shape, rectangles)
    if np.any(area & ~valid):
        raise ValueError('LPIPS rectangle extends outside the test domain')
    scores = []
    if lpips_fn is not None:
        for x, y, w, h in rectangles:
            if stop_check:stop_check()
            value = float(lpips_fn(clipped[y:y+h, x:x+w], t[y:y+h, x:x+w].astype(np.float32)))
            if not np.isfinite(value):
                raise FloatingPointError('LPIPS returned a nonfinite score')
            scores.append(value)
    result.update(lpips=float(np.mean(scores)) if scores else None,
                  lpips_details={'rectangle_count': len(rectangles), 'per_rectangle': scores,
                                 'unique_pixels': int(area.sum()),
                                 'valid_coverage': float(area.sum()/valid.sum())},
                  complete=bool(centres.any() and scores))
    if require_complete and not result['complete']:
        raise ValueError('Incomplete metric domain/dependency: require SSIM centres and computed LPIPS rectangles')
    return result


def save_preview(path, prediction, max_edge=1536):
    from PIL import Image
    image = Image.fromarray(np.round(np.clip(prediction, 0, 1)*255).astype(np.uint8))
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    image.save(temp, format='PNG')
    os.replace(str(temp), str(path))


def cached_prediction(path, identity, build):
    path = Path(path)
    sidecar = path.with_suffix('.json')
    if path.exists() and sidecar.exists():
        try:
            record = json.loads(sidecar.read_text(encoding='utf-8'))
        except (ValueError,OSError):
            record = {}
        if record.get('identity') == identity and record.get('sha256') == sha256(path):
            prediction = np.load(path, allow_pickle=False, mmap_mode='r')
            if list(prediction.shape) == record['shape'] and str(prediction.dtype) == 'float32':
                return prediction, True
    prediction = build()
    atomic_npy(path, prediction)
    atomic_json(sidecar, {'identity': identity, 'sha256': sha256(path),
                          'shape': list(prediction.shape), 'dtype': 'float32'})
    return np.load(path, allow_pickle=False, mmap_mode='r'), False


def evaluate_task(model, store, task, protocol, run_dir, device='cpu',
                  input_range='zero_one', output_range=None, lpips_fn=None,
                  model_identity='', batch_size=1, require_complete=True):
    if not model_identity:
        raise ValueError('Evaluation needs an explicit checkpoint/content identity')
    root = Path(run_dir) / 'evaluation'
    root.mkdir(parents=True, exist_ok=True)
    inference_identity = stable_hash({'data': store.identity, 'model': model_identity,
                                      'channel_map': task['channel_map'], 'input_range': input_range,
                                      'output_range': output_range or input_range, 'grid': METRIC_VERSION,
                                      'inference_implementation':implementation_identity('inference')})
    stop_check = _stop_checker(run_dir)
    result = {'task_id': task['task_id'], 'experiment': task['experiment'], 'model': task['model'],
              'input_id': task['input_id'], 'fold': task.get('fold'), 'split_seed': task.get('split_seed'),
              'identity': stable_hash({'inference': inference_identity, 'protocol': protocol['identity']}),
              'metric_implementation':implementation_identity('metric'),
              'lpips_identity':getattr(lpips_fn,'identity',None),
              'sites': {}, 'prediction_files': {}, 'complete': False}
    for site in protocol['eval_sites']:
        stop_check()
        path = root / 'predictions' / (site + '.npy')
        identity = stable_hash({'inference': inference_identity, 'site': site})
        def build():
            return predict_scene(model, store.input_image(site, task['channel_map']), device,
                                 input_range, output_range, batch_size=batch_size,stop_check=stop_check)[0]
        predicted, reused = cached_prediction(path, identity, build)
        valid = evaluation_mask(store, site, protocol)
        rectangles = (store.support['scenes'][site]['lpips_rectangles']
                      if not protocol['test_windows'] else full_valid_boxes(valid, 128, 128))
        metric_identity=metric_cache_identity(sha256(path),store.identity,valid,rectangles,lpips_fn,
                                              {'inference':identity,'protocol':protocol['identity']})
        metrics,metric_reused=cached_metrics(root/'metrics'/(site+'.json'),metric_identity,
            lambda:score_scene(predicted,store.target(site),valid,lpips_fn,rectangles,require_complete,
                               stop_check=stop_check),
            allow_reuse=getattr(lpips_fn,'identity',None) is not None,
            context={'site':site,'prediction_identity':identity,'evaluation_identity':result['identity']})
        metrics['prediction_reused'] = reused
        metrics['metrics_reused'] = metric_reused
        preview=root/'previews'/(site+'.png')
        if not reused or not preview.exists():save_preview(preview,predicted)
        result['sites'][site] = metrics
        result['prediction_files'][site] = str(path.resolve())
        # Updating summary after each site makes completed work inspectable on crash.
        atomic_json(root / 'summary.json', result)
    result['complete'] = all(row['complete'] for row in result['sites'].values())
    atomic_json(root / 'summary.json', result)
    return result


def score_prediction_domains(store, predictions, protocol, lpips_fn, output_path=None):
    result = {}
    for site in protocol['eval_sites']:
        prediction = np.load(predictions[site], allow_pickle=False, mmap_mode='r')
        valid = evaluation_mask(store, site, protocol)
        result[site] = score_scene(prediction, store.target(site), valid, lpips_fn)
    if output_path:
        atomic_json(output_path, {'protocol_identity': protocol['identity'], 'sites': result})
    return result


def shift_conditions():
    conditions = [{'condition': 'zero', 'dx': 0., 'dy': 0.}]
    for distance in (.5, 1., 2., 4.):
        for direction, dx, dy in [('left',-distance,0.), ('right',distance,0.),
                                  ('up',0.,-distance), ('down',0.,distance)]:
            conditions.append({'condition': '%s_%gpx' % (direction, distance), 'dx': dx, 'dy': dy})
    return conditions


def shift_bilinear(image, geometry, dx, dy):
    """Positive dx moves content right. Every nonzero interpolation source must be valid."""
    image = np.asarray(image, np.float32)
    geometry = np.asarray(geometry, bool)
    height, width = geometry.shape
    xx = np.arange(width, dtype=np.float64) - dx
    yy = np.arange(height, dtype=np.float64) - dy
    x0, y0 = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
    fx, fy = xx-x0, yy-y0
    output = np.zeros_like(image, np.float32)
    valid = shift_validity(geometry,dx,dy)
    for ox, wx in [(0,1-fx), (1,fx)]:
        for oy, wy in [(0,1-fy), (1,fy)]:
            if not wx.any() or not wy.any():
                continue
            xs, ys = x0+ox, y0+oy
            xc, yc = np.clip(xs,0,width-1), np.clip(ys,0,height-1)
            # Chunked rows bound temporary allocations for 20+ megapixel scenes.
            for first in range(0,height,128):
                last = min(first+128,height)
                weight = (wy[first:last,None] * wx[None,:]).astype(np.float32)
                inside = ((ys[first:last,None] >= 0) & (ys[first:last,None] < height)
                          & (xs[None,:] >= 0) & (xs[None,:] < width))
                output[first:last] += image[yc[first:last,None],xc[None,:]] * weight[...,None] * inside[...,None]
    output[~valid] = 0
    return output, valid


def shift_validity(geometry,dx,dy):
    geometry = np.asarray(geometry,bool)
    height,width = geometry.shape
    xx,yy = np.arange(width,dtype=np.float64)-dx,np.arange(height,dtype=np.float64)-dy
    x0,y0 = np.floor(xx).astype(np.int64),np.floor(yy).astype(np.int64)
    fx,fy = xx-x0,yy-y0
    valid = np.ones_like(geometry)
    for ox,wx in [(0,1-fx),(1,fx)]:
        for oy,wy in [(0,1-fy),(1,fy)]:
            if not wx.any() or not wy.any():
                continue
            xs,ys = x0+ox,y0+oy
            xc,yc = np.clip(xs,0,width-1),np.clip(ys,0,height-1)
            for first in range(0,height,256):
                last = min(first+256,height)
                relevant = (wy[first:last,None] > 0) & (wx[None,:] > 0)
                inside = ((ys[first:last,None] >= 0) & (ys[first:last,None] < height)
                          & (xs[None,:] >= 0) & (xs[None,:] < width))
                valid[first:last] &= ~relevant | (inside & geometry[yc[first:last,None],xc[None,:]])
    return valid


def sensitivity_domain(store, site):
    if site not in store.geometry_masks:
        raise ValueError('EX4 requires frozen geometry/privacy mask for ' + site)
    geometry = store.geometry_masks[site]
    common = geometry.copy()
    for condition in shift_conditions():
        valid = shift_validity(geometry,condition['dx'],condition['dy'])
        common &= valid
    boxes = full_valid_boxes(common, 256, 128)
    valid = store.masks[site] & common & coverage(common.shape, boxes)
    if not boxes or not valid.any():
        raise ValueError('No shared geometric inference context for EX4: ' + site)
    rectangles = full_valid_boxes(valid, 128, 128)
    if not rectangles or not binary_erosion(valid, np.ones((11,11),bool)).any():
        raise ValueError('EX4 domain cannot support all metrics: ' + site)
    return {'boxes': boxes, 'mask': valid, 'rectangles': rectangles}


def evaluate_sensitivity(model, store, task, run_dir, device='cpu', input_range='zero_one',
                         output_range=None, lpips_fn=None, model_identity='', batch_size=1,
                         retain_predictions=True):
    if task['experiment'] != 'EX2' or task['model'] != 'pix2pix':
        raise ValueError('EX4 uses EX2 pix2pix test-fold models only')
    if not model_identity:
        raise ValueError('EX4 needs checkpoint identity')
    root = Path(run_dir) / 'sensitivity'
    stop_check = _stop_checker(run_dir)
    result = {'task_id': task['task_id'], 'conditions': {}, 'complete': False,
              'units': 'published-image pixels', 'interpolation': 'bilinear',
              'zero_uses_common_context': True}
    for site in task['test_sites']:
        stop_check()
        domain = sensitivity_domain(store, site)
        image, target = store.input_image(site, task['channel_map']), store.target(site)
        for condition in shift_conditions():
            stop_check()
            name = condition['condition']
            identity = stable_hash({'data': store.identity, 'model': model_identity, 'site':site, 'condition': condition,
                                    'boxes': domain['boxes'], 'input': task['channel_map'],
                                    'input_range': input_range, 'output_range': output_range or input_range,
                                    'version': METRIC_VERSION,
                                    'inference_implementation':implementation_identity('inference'),
                                    'shift_implementation':implementation_identity('shift')})
            def build():
                shifted, _ = shift_bilinear(image, store.geometry_masks[site], condition['dx'], condition['dy'])
                return predict_scene(model, shifted, device, input_range, output_range,
                                     boxes=domain['boxes'], batch_size=batch_size,stop_check=stop_check)[0]
            path = root / name / (site + '.npy')
            if retain_predictions:
                prediction, _ = cached_prediction(path, identity, build)
            else:
                prediction = build()
            prediction_sha=sha256(path) if retain_predictions else _array_identity(prediction)
            metric_identity=metric_cache_identity(prediction_sha,store.identity,domain['mask'],
                domain['rectangles'],lpips_fn,{'inference':identity})
            metrics,metric_reused=cached_metrics(root/name/(site+'_metrics.json'),metric_identity,
                lambda:score_scene(prediction,target,domain['mask'],lpips_fn,domain['rectangles'],
                                   stop_check=stop_check),
                allow_reuse=getattr(lpips_fn,'identity',None) is not None,
                context={'condition':condition,'site':site,'prediction_identity':identity})
            metrics['metrics_reused']=metric_reused
            result['conditions'].setdefault(name, {})[site] = metrics
            atomic_json(root / 'summary.json', result)
    result['complete'] = True
    atomic_json(root / 'summary.json', result)
    return result


def preflight_domains(store, frozen, include_sensitivity=True):
    """Check domains before expensive training; fails on absent metrics support."""
    from .protocols import make_matrix, task_protocol
    rows = []
    protocols = [('EX2', {'test_windows': []})]
    for task in make_matrix():
        if task['experiment'] == 'EX3' and task['input_id'] == 'S1':
            protocols.append(('EX3_%d' % task['split_seed'], task_protocol(task, frozen)))
    for name, protocol in protocols:
        for site in store.sites:
            valid = evaluation_mask(store, site, protocol)
            boxes = (store.support['scenes'][site]['lpips_rectangles'] if name == 'EX2'
                     else full_valid_boxes(valid,128,128))
            centres = binary_erosion(valid,np.ones((11,11),bool))
            if not boxes or not centres.any():
                raise ValueError('Incomplete evaluation support before training: %s/%s' % (name,site))
            area = coverage(valid.shape,boxes)
            rows.append({'protocol':name,'site':site,'valid_pixels':int(valid.sum()),
                         'ssim_centres':int(centres.sum()),'lpips_rectangles':len(boxes),
                         'lpips_unique_coverage':float(area.sum()/valid.sum())})
    if include_sensitivity:
        for site in store.sites:
            domain = sensitivity_domain(store,site)
            rows.append({'protocol':'EX4','site':site,'valid_pixels':int(domain['mask'].sum()),
                         'inference_windows':len(domain['boxes']), 'lpips_rectangles':len(domain['rectangles'])})
    return rows
