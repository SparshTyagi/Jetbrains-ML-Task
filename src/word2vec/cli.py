from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
	parser = argparse.ArgumentParser(description="Word2Vec utilities")
	subparsers = parser.add_subparsers(dest="command", required=True)

	train_parser = subparsers.add_parser("train", help="Run training")
	train_parser.add_argument("args", nargs=argparse.REMAINDER)

	eval_parser = subparsers.add_parser("eval", help="Run nearest-neighbor evaluation")
	eval_parser.add_argument("args", nargs=argparse.REMAINDER)

	parsed = parser.parse_args()

	module = "word2vec.train" if parsed.command == "train" else "word2vec.eval"
	command = [sys.executable, "-m", module, *parsed.args]
	completed = subprocess.run(command, check=False)
	raise SystemExit(completed.returncode)


if __name__ == "__main__":
	main()
