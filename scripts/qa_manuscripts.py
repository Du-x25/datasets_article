"""Unified manuscript QA entrypoint for Chinese and English Data Descriptors."""
import argparse
from qa_results_20261109 import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['zh', 'en', 'both'], default='both')
    args = parser.parse_args()
    langs = ['zh', 'en'] if args.language == 'both' else [args.language]
    for lang in langs:
        validate(lang)


if __name__ == '__main__':
    main()
