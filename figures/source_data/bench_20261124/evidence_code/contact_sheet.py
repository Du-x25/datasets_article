"""Render every page of a PDF into contact sheets (3 x 3 pages each) for a layout pass."""
import sys
from pathlib import Path

import fitz

CELL_W, COLS, ROWS = 400, 3, 3


def main(pdf, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    cell_h = int(CELL_W * 841.89 / 595.276)
    per = COLS * ROWS
    for s in range(0, doc.page_count, per):
        sheet = fitz.open()
        page = sheet.new_page(width=COLS * CELL_W, height=ROWS * (cell_h + 18))
        for i in range(per):
            k = s + i
            if k >= doc.page_count:
                break
            x, y = (i % COLS) * CELL_W, (i // COLS) * (cell_h + 18)
            page.show_pdf_page(fitz.Rect(x, y, x + CELL_W, y + cell_h), doc, k)
            page.insert_text((x + 4, y + cell_h + 14), "p%d" % (k + 1), fontsize=12)
        name = out / ("sheet_%02d-%02d.png" % (s + 1, min(s + per, doc.page_count)))
        page.get_pixmap(dpi=72).save(str(name))
        print(name)
    print("pages:", doc.page_count)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
