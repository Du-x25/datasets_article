"""Check tracked/staged manuscript files without loading local research data."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".tif", ".tiff", ".npy", ".npz", ".h5", ".hdf5", ".mat",
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pkl", ".pickle",
    ".joblib", ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz", ".bundle",
    ".pem", ".key",
}
FORBIDDEN_DIRS = {
    "automation_completion_20261015", "delivery_zh_20261015", "data", "datasets",
    "raw", "raw_data", "checkpoints", "weights", "predictions", "runs", "outputs",
    "build", "tmp", "__pycache__", ".venv", "venv", "qa_six_figures",
}


def main():
    raw = subprocess.check_output(["git", "ls-files", "-s", "-z"], cwd=ROOT)
    files = {}
    errors = []
    total = 0
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        meta, name = entry.split(b"\t", 1)
        mode, oid, stage = meta.decode("ascii").split()
        relative = name.decode("utf-8")
        files[relative] = oid
        path = Path(relative)
        if stage != "0" or mode not in {"100644", "100755"}:
            errors.append(f"unexpected index entry: {relative} ({mode}, stage {stage})")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or set(path.parts) & FORBIDDEN_DIRS:
            errors.append(f"excluded material is tracked: {relative}")
        if path.name == ".env" or path.name.startswith(".env."):
            errors.append(f"local environment file is tracked: {relative}")
        size = int(subprocess.check_output(["git", "cat-file", "-s", oid], cwd=ROOT))
        total += size
        if size > 20 * 1024 * 1024:
            errors.append(f"file exceeds manuscript limit of 20 MiB: {relative}")

    for language in ("zh", "en"):
        source_name = f"main_{language}.tex"
        if source_name not in files:
            errors.append(f"untracked manuscript: {source_name}")
            continue
        source = subprocess.check_output(["git", "cat-file", "blob", files[source_name]], cwd=ROOT).decode("utf-8")
        dependencies = ["refs.bib"]
        dependencies += [name if Path(name).suffix else name + ".tex"
                         for name in re.findall(r"\\input\{([^}]+)\}", source)]
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source):
            dependencies.append(name if name in files else "figures/" + name)
        for name in dependencies:
            if name not in files:
                errors.append(f"untracked compile dependency for {language}: {name}")
    for error in errors:
        print("FAIL:", error)
    print(f"Checked {len(files)} tracked/staged files, {total / 1024**2:.2f} MiB.")
    print("Repository scope and manuscript dependencies:", "FAIL" if errors else "PASS")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
