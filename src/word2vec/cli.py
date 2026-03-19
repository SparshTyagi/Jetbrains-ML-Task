from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Word2Vec utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run training", add_help=False)
    subparsers.add_parser("eval", help="Run nearest-neighbor evaluation", add_help=False)

    # Only consume the subcommand name; leave the rest for the target module.
    parsed, remaining = parser.parse_known_args()

    if parsed.command == "train":
        sys.argv = [sys.argv[0]] + remaining
        from word2vec.train import main as _main
        _main()
    else:
        sys.argv = [sys.argv[0]] + remaining
        from word2vec.eval import main as _main
        _main()


if __name__ == "__main__":
    main()
