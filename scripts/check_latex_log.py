"""Fail on unresolved references, oversized floats and other unusable PDF output.

Read the final XeLaTeX log, not latexmk's combined output from earlier passes:
temporary undefined references on a first pass are expected to resolve later.
"""
import argparse
import json
from pathlib import Path
import re

PATTERNS = {
    'undefined_reference_or_citation': r'\bundefined\b',
    'duplicate_label': r'multiply[- ]defined|Label .* multiply defined',
    'oversized_float': r'Float too large',
    'overfull_box': r'Overfull \\[hv]box',
    'missing_glyph': r'Missing character:',
    'rerun_required': r'Rerun to get|Label\(s\) may have changed|Please rerun LaTeX',
    'tex_error': r'^!|(?:LaTeX|Package \S+|Class \S+) Error:',
}


def find_issues(log):
    issues = []
    for number, line in enumerate(log.splitlines(), 1):
        for category, pattern in PATTERNS.items():
            if re.search(pattern, line, re.IGNORECASE):
                issues.append({'category': category, 'log_line': number, 'message': line.strip()})
    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path, help='Final .log file written by XeLaTeX')
    args = parser.parse_args(argv)
    if not args.log.is_file():
        print(json.dumps({'status':'failed', 'error':f'Missing log: {args.log}'}))
        return 1
    issues = find_issues(args.log.read_text(encoding='utf-8', errors='replace'))
    print(json.dumps({'status':'failed' if issues else 'passed', 'issues':issues}, ensure_ascii=False, indent=2))
    return int(bool(issues))


if __name__ == '__main__':
    raise SystemExit(main())
