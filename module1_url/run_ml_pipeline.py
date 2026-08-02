"""Simple entry point for training and testing the phishing URL model."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(command: list[str], description: str) -> None:
    print(f"\n=== {description} ===")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    print("Phishing URL ML pipeline")
    print("Training model...")
    run_step([sys.executable, "models/train.py"], "Training")
    run_step([sys.executable, "test_model.py"], "Testing")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
