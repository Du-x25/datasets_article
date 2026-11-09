"""Check the local tools for compiling and validating the Chinese manuscript."""
import argparse
import importlib
from importlib.metadata import version
import shutil
import subprocess
import sys

TEX_FILES = (
    'ctexart.cls', 'xeCJK.sty', 'fontspec.sty', 'geometry.sty', 'graphicx.sty',
    'booktabs.sty', 'longtable.sty', 'tabularx.sty', 'array.sty', 'amsmath.sty',
    'xcolor.sty', 'hyperref.sty', 'caption.sty', 'subcaption.sty', 'float.sty',
    'placeins.sty', 'natbib.sty', 'unsrtnat.bst',
)
PYTHON_PACKAGES = {'numpy':'numpy', 'pandas':'pandas', 'matplotlib':'matplotlib',
                   'Pillow':'PIL', 'pypdf':'pypdf'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-only', action='store_true', help='Skip Python packages and Poppler checks')
    parser.add_argument('--figures', action='store_true', help='Also require an installed CJK plotting font')
    args = parser.parse_args()
    if args.compile_only and args.figures:
        parser.error('--figures cannot be combined with --compile-only')
    failures = []
    tools = ['git', 'xelatex', 'latexmk', 'bibtex', 'kpsewhich']
    if not args.compile_only:
        tools.append('pdftotext')
    for name in tools:
        path = shutil.which(name)
        print(f'{"OK" if path else "MISSING"} {name}: {path or "not on PATH"}')
        if not path:
            failures.append(name)
    kpsewhich = shutil.which('kpsewhich')
    if kpsewhich:
        missing = []
        for name in TEX_FILES:
            r = subprocess.run([kpsewhich, name], capture_output=True, timeout=20)
            if r.returncode or not r.stdout.strip():
                missing.append(name)
        print('TeX packages:', ', '.join(missing) if missing else f'all {len(TEX_FILES)} found')
        failures.extend(missing)
    if not args.compile_only:
        print('Python:', sys.version.split()[0], sys.executable)
        if sys.version_info < (3, 11):
            failures.append('Python >= 3.11 (the Conda environment selects 3.11)')
        for package, module in PYTHON_PACKAGES.items():
            try:
                importlib.import_module(module)
                print(f'OK {package} {version(package)}')
            except (ImportError, OSError) as exc:
                print(f'MISSING/BROKEN {package}: {exc}')
                failures.append(package)
        if args.figures and not failures:
            from plot_fonts import select_plot_font
            try:
                family, path = select_plot_font()
                print(f'OK plotting font: {family} ({path})')
            except RuntimeError as exc:
                print(str(exc))
                failures.append('CJK plotting font')
    print('Environment:', 'FAIL — see README.md for installation steps' if failures else 'PASS')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
