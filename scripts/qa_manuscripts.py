from __future__ import annotations

import hashlib
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDFS = (ROOT / "build" / "en" / "main_en.pdf", ROOT / "build" / "zh" / "main_zh.pdf")


def main() -> None:
    for path in PDFS:
        if not path.exists():
            raise SystemExit(f"Missing compiled manuscript: {path}")
        reader = PdfReader(path)
        texts = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                raise SystemExit(f"Blank page: {path} page {page_number}")
            if "\ufffd" in text:
                raise SystemExit(
                    f"Unicode replacement character: {path} page {page_number}"
                )
            texts.append(text)
        joined = "\n".join(texts)
        for required in ("M-SWIR", "1691", "60"):
            if required not in joined:
                raise SystemExit(f"Required field {required!r} missing from {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(
            f"{path.name}: pages={len(reader.pages)} chars={len(joined)} sha256={digest}"
        )


if __name__ == "__main__":
    main()

