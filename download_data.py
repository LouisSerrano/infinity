import argparse
from pathlib import Path
import airfrans as af

def main(data_dir):
    af.dataset.download(root=Path(data_dir), unzip=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AirFRANS dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory where the dataset should be downloaded"
    )
    args = parser.parse_args()
    main(args.data_dir)

