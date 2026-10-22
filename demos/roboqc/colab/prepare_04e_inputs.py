"""Archive existing /data photos for upload. Does not generate data or run models."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase_1/src"))
from roboqc.industrial.review_bundle import pack_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE.parents[2] / "data")
    parser.add_argument("--output", type=Path, default=HERE / "assets/data_04e_inputs.zip")
    args = parser.parse_args()
    print(pack_images(args.source, args.output))
