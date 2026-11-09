"""Resolve an installed CJK plotting font without a machine-specific path."""
import os
from pathlib import Path


def select_plot_font():
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    preferred = os.environ.get('MSWIR_PAPER_FONT')
    if preferred and Path(preferred).is_file():
        font_manager.fontManager.addfont(preferred)
        name = font_manager.FontProperties(fname=preferred).get_name()
        candidates = [(name, preferred)]
    elif preferred:
        candidates = [(preferred, None)]
    else:
        candidates = [(name, None) for name in (
            'Microsoft YaHei', 'Noto Sans CJK SC', 'Noto Sans SC',
            'Source Han Sans SC', 'PingFang SC', 'Heiti SC',
            'WenQuanYi Zen Hei', 'SimHei')]
    for name, path in candidates:
        try:
            path = path or font_manager.findfont(name, fallback_to_default=False)
            glyphs = FT2Font(path).get_charmap()
            if all(ord(c) in glyphs for c in '地点验证波段训练颜色均值'):
                return name, str(path)
        except (ValueError, RuntimeError, OSError):
            continue
    raise RuntimeError(
        'No usable CJK plotting font. Install Noto Sans SC / Noto Sans CJK SC, '
        'or set MSWIR_PAPER_FONT to an installed font family or a .ttf/.otf path. '
        'Compiling the existing paper figures does not require this plotting font.')
