"""
Main Runner Script
Executes the full pipeline: collect data -> engineer features -> train model -> predict
"""

import subprocess
import sys


def run_step(description: str, script: str):
    """Run a Python script and handle errors."""
    print("\n" + "="*60)
    print(f"  {description}")
    print("="*60 + "\n")

    result = subprocess.run([sys.executable, script], capture_output=False)

    if result.returncode != 0:
        print(f"\nError running {script}. Exiting.")
        sys.exit(1)


def main():
    print("\n" + "#"*60)
    print("  FOOTBALL MATCH PREDICTOR - FULL PIPELINE")
    print("#"*60)

    # Step 1: Collect data
    run_step("STEP 1: Collecting historical match data", "collect_data.py")

    # Step 2: Engineer features
    run_step("STEP 2: Engineering features", "features.py")

    # Step 3: Train model
    run_step("STEP 3: Training prediction model", "train_model.py")

    print("\n" + "#"*60)
    print("  PIPELINE COMPLETE!")
    print("#"*60)
    print("\nYou can now make predictions by running:")
    print("  python predict.py")
    print("\nOr predict specific fixtures:")
    print("  python predict.py --fixtures")


if __name__ == "__main__":
    main()
