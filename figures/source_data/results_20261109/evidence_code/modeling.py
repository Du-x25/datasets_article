"""Three-channel model adapters backed by complete, archived Chen modules.

See provenance/CHEN_IMPLEMENTATION.md before describing this as a reproduction.
The training engine owns optimizers, schedulers, precision, RNG and persistence.
"""
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


def _mask(mask, reference):
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("Expected valid mask [N, 1, H, W]")
    if mask.shape[0] != reference.shape[0] or mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError("Mask and image dimensions differ")
    mask = mask.to(device=reference.device, dtype=torch.float32)
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Training validity mask must be binary")
    if not torch.all(mask.sum((1, 2, 3)) > 0):
        raise ValueError("Every training sample needs nonempty valid support")
    return mask


def masked_l1(pred, target, mask):
    """Mean over valid pixels and RGB channels, with equal weighting per sample."""
    mask = _mask(mask, pred)
    errors = (pred.float() - target.float()).abs() * mask
    return (errors.sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * pred.shape[1])).mean()


def _fill(image, mask):
    """Zero is neutral gray in the GAN's [-1, 1] image space."""
    return image.float() * mask


def _raw_and_weighted(values, weights):
    result = {}
    total = None
    for key, value in values.items():
        weighted = value * weights[key]
        result[key] = value.detach()
        result[key + "_weighted"] = weighted.detach()
        total = weighted if total is None else total + weighted
    result["total"] = total.detach()
    return total, result


class ModelBundle:
    """Model and objective. No hidden optimizer, image pool, RNG or history."""

    def __init__(self, name, generator, discriminator, config, perceptual=None, color=None):
        self.name = name
        self.generator = generator
        self.discriminator = discriminator
        self.config = dict(config)
        self.perceptual = perceptual
        self.color = color
        self.input_range = "zero_one" if name == "drunet" else "minus_one_one"
        self.description = {
            "name": name, "input_nc": 3, "output_nc": 3,
            "input_range": self.input_range, "pool_size": 0,
            "config": self.config,
            "generator_parameters": sum(p.numel() for p in generator.parameters()),
            "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()) if discriminator is not None else 0,
            "mask_protocol": "valid-only per-sample L1; GAN/FM/VGG/color share zero-filled invalid input and output; feature losses are not pixel-masked perceptual metrics",
        }

    def state_dict_extra(self):
        return {}

    def load_state_dict_extra(self, state):
        if state:
            raise ValueError("This model uses no image pool or additional mutable state")

    def _check(self, fake, x, y, mask):
        if fake.shape != y.shape or x.shape != y.shape or y.ndim != 4 or y.shape[1] != 3:
            raise ValueError("All model inputs and RGB outputs must be equal-shape N x 3 x H x W tensors")
        return _mask(mask, y)

    def _predictions(self, x, image, mask):
        if self.discriminator is None:
            raise RuntimeError("Inference-only model has no discriminator")
        return self.discriminator(torch.cat((_fill(x, mask), _fill(image, mask)), dim=1))

    def generator_loss(self, fake, x, y, mask):
        mask = self._check(fake, x, y, mask)
        if self.name == "drunet":
            return _raw_and_weighted({"l1": masked_l1(fake, y, mask)}, {"l1": 1.0})
        fake_features = self._predictions(x, fake, mask)
        if self.name == "pix2pix":
            logits = fake_features.float()
            adversarial = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
            return _raw_and_weighted(
                {"adversarial": adversarial, "l1": masked_l1(fake, y, mask)},
                {"adversarial": 1.0, "l1": float(self.config["lambda_l1"])})

        # Full archived multi-scale D features, preserving its sum-of-scales GAN
        # and 4/(n_layers+1)/num_D feature-matching convention.
        with torch.no_grad():
            real_features = self._predictions(x, y, mask)
        adversarial = sum((branch[-1].float() - 1).square().mean() for branch in fake_features)
        fm = fake.float().new_zeros(())
        feature_weight = 4.0 / (int(self.config["n_layers_D"]) + 1) / int(self.config["num_D"])
        for fake_scale, real_scale in zip(fake_features, real_features):
            for left, right in zip(fake_scale[:-1], real_scale[:-1]):
                fm = fm + feature_weight * F.l1_loss(left.float(), right.detach().float())
        a, b = _fill(fake, mask), _fill(y, mask)
        # Disable autocast for both pretrained perception and the low-frequency
        # color kernel. This does not change the precision of G or D forward.
        with torch.autocast(device_type=a.device.type, enabled=False):
            if self.config["vgg_normalization"] == "imagenet":
                mean = a.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = a.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                va, vb = (a * 0.5 + 0.5 - mean) / std, (b * 0.5 + 0.5 - mean) / std
            else:
                va, vb = a, b
            perceptual = self.perceptual(va, vb)
            color = self.color(a, b)
        return _raw_and_weighted(
            {"adversarial": adversarial, "feature_matching": fm, "perceptual": perceptual, "color": color},
            {"adversarial": 1.0, "feature_matching": float(self.config["lambda_fm"]),
             "perceptual": float(self.config["lambda_vgg"]), "color": float(self.config["lambda_color"])})

    def discriminator_loss(self, fake, x, y, mask):
        mask = self._check(fake, x, y, mask)
        if self.discriminator is None:
            raise RuntimeError("This bundle has no trainable discriminator")
        fake_features = self._predictions(x, fake.detach(), mask)
        real_features = self._predictions(x, y, mask)
        if self.name == "pix2pix":
            a, b = fake_features.float(), real_features.float()
            d_fake = F.binary_cross_entropy_with_logits(a, torch.zeros_like(a))
            d_real = F.binary_cross_entropy_with_logits(b, torch.ones_like(b))
        else:
            d_fake = sum(branch[-1].float().square().mean() for branch in fake_features)
            d_real = sum((branch[-1].float() - 1).square().mean() for branch in real_features)
        return _raw_and_weighted({"d_fake": d_fake, "d_real": d_real}, {"d_fake": 0.5, "d_real": 0.5})


DEFAULT_CONFIGS = {
    "pix2pix": {"ngf": 64, "ndf": 64, "num_downs": 8, "n_layers_D": 3,
                "lambda_l1": 100.0, "use_dropout": True, "init_std": 0.02},
    "tfd": {"ngf": 64, "ndf": 64, "n_downsampling": 4, "n_blocks": 12,
            "num_D": 3, "n_layers_D": 3, "lambda_fm": 10.0,
            "lambda_vgg": 10.0, "lambda_color": 5.0, "vgg_normalization": "imagenet"},
    "drunet": {"channels": [64, 128, 256, 512], "nb": 4, "init_gain": 0.2},
}


def _vgg_path(weights_dir):
    folder = Path(weights_dir)
    candidates = [folder / "losses" / "vgg19-dcbb9e9d.pth", folder / "vgg19-dcbb9e9d.pth"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("TFD requires offline pretrained VGG19 weights before training: %s" % candidates[0])


def build_bundle(name: str, device, weights_dir, config: Optional[Dict] = None, training=True):
    """Build the complete architecture; model checkpoints are loaded separately, strictly.

    Config keys outside model-specific defaults are rejected to catch misspellings
    before a long run. Three input/output channels are fixed by this API.
    """
    name = name.lower()
    if name not in DEFAULT_CONFIGS:
        raise ValueError("Unknown model: %s" % name)
    supplied = dict(config or {})
    unknown = set(supplied).difference(DEFAULT_CONFIGS[name])
    if unknown:
        raise ValueError("Unknown %s model config keys: %s" % (name, sorted(unknown)))
    resolved = dict(DEFAULT_CONFIGS[name])
    resolved.update(supplied)
    device = torch.device(device)
    discriminator, perceptual, color = None, None, None

    if name == "pix2pix":
        from .chen_pix2pix import networks as module
        generator = module.UnetGenerator(3, 3, int(resolved["num_downs"]), int(resolved["ngf"]),
                                         norm_layer=nn.BatchNorm2d, use_dropout=bool(resolved["use_dropout"]))
        if training:
            module.init_weights(generator, "normal", float(resolved["init_std"]))
            discriminator = module.NLayerDiscriminator(6, int(resolved["ndf"]), int(resolved["n_layers_D"]), norm_layer=nn.BatchNorm2d)
            module.init_weights(discriminator, "normal", float(resolved["init_std"]))
    elif name == "tfd":
        from .chen_tfd import networks as module
        if resolved["vgg_normalization"] not in ("imagenet", "archive_raw"):
            raise ValueError("vgg_normalization must be imagenet or archive_raw")
        if min(float(resolved[key]) for key in ("lambda_fm", "lambda_vgg", "lambda_color")) <= 0:
            raise ValueError("The complete TFD profile requires active FM, VGG and color objectives")
        if training:
            weights_path = _vgg_path(weights_dir)
        generator = module.GlobalGenerator(3, 3, int(resolved["ngf"]), int(resolved["n_downsampling"]),
                                           int(resolved["n_blocks"]), norm_layer=module.get_norm_layer("instance"))
        if training:
            generator.apply(module.weights_init)
            discriminator = module.MultiscaleDiscriminator(6, int(resolved["ndf"]), int(resolved["n_layers_D"]),
                                                            module.get_norm_layer("instance"), False, int(resolved["num_D"]), True)
            discriminator.apply(module.weights_init)
            perceptual = module.VGGLoss(weights_path=weights_path).to(device).eval().requires_grad_(False)
            color = module.ColorLoss().to(device)
    else:
        from .chen_drunet.network_unet import UNetRes
        channels = [int(value) for value in resolved["channels"]]
        if len(channels) != 4:
            raise ValueError("DRUNet requires four channel widths")
        generator = UNetRes(in_nc=3, out_nc=3, nc=channels, nb=int(resolved["nb"]), act_mode="R",
                            downsample_mode="strideconv", upsample_mode="convtranspose", bias=False)
        if training:
            # The archived KAIR init_weights branch calls orthogonal_ with gain.
            # Device initialization avoids expensive CPU QR on the full profile.
            generator = generator.to(device)
            for layer in generator.modules():
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.orthogonal_(layer.weight, gain=float(resolved["init_gain"]))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    generator = generator.to(device)
    generator.train(training)
    if discriminator is not None:
        discriminator = discriminator.to(device).train()
    return ModelBundle(name, generator, discriminator, resolved, perceptual=perceptual, color=color)
