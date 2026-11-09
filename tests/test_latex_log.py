"""Regression cases for warnings that must stop automatic PDF publication."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from check_latex_log import find_issues, main


class LatexLogChecks(unittest.TestCase):
    def test_unresolved_and_wrapped_references(self):
        log = "LaTeX Warning: Reference `fig:missing' on page 2 undefined on input line 40.\n"
        log += "Package natbib Warning: Citation `missing' on page 3\n(natbib)                undefined on input line 70.\n"
        self.assertEqual(len(find_issues(log)), 2)

    def test_oversized_float_and_horizontal_vertical_overflow(self):
        log = "LaTeX Warning: Float too large for page by 10.36777pt on input line 311.\n"
        log += "Overfull \\hbox (1.0pt too wide) in paragraph at lines 5--6\n"
        log += "Overfull \\vbox (2.0pt too high) has occurred while \\output is active\n"
        self.assertEqual({x['category'] for x in find_issues(log)}, {'oversized_float', 'overfull_box'})
        self.assertEqual(len(find_issues(log)), 3)

    def test_duplicate_labels_missing_glyphs_and_stale_refs(self):
        log = "LaTeX Warning: There were multiply-defined labels.\n"
        log += "Missing character: There is no 中 in font Example!\n"
        log += "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.\n"
        self.assertEqual({x['category'] for x in find_issues(log)}, {'duplicate_label', 'missing_glyph', 'rerun_required'})

    def test_normal_log_and_underfull_box_do_not_fail(self):
        log = "Package: infwarerr 2019/12/03 Providing info/warning/error messages\n"
        log += "Underfull \\hbox (badness 1000) in paragraph at lines 4--6\n"
        log += "Output written on build/zh/main_zh.xdv (19 pages, 383216 bytes).\n"
        self.assertEqual(find_issues(log), [])

    def test_tex_error_fails(self):
        self.assertEqual(find_issues('! LaTeX Error: File missing.sty not found.')[0]['category'], 'tex_error')

    def test_command_exit_status(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path = Path(folder) / 'paper.log'
            self.assertEqual(main([str(path)]), 1)
            path.write_text('Output written on paper.xdv (19 pages).', encoding='utf-8')
            self.assertEqual(main([str(path)]), 0)
            path.write_text('LaTeX Warning: Float too large for page by 1pt.', encoding='utf-8')
            self.assertEqual(main([str(path)]), 1)


if __name__ == '__main__':
    unittest.main()
