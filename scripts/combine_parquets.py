import argparse
from pathlib import Path

import pandas as pd

def combine_parquets(input_folder: str, output_path: str, add_source_col: bool=True) -> None:
    input_folder = Path(input_folder)
    output_path = Path(output_path)

    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_folder}")

    parquet_files = sorted(input_folder.glob("*.parquet"))
    parquet_files = [f for f in parquet_files if f.resolve() != output_path.resolve()]

    if not parquet_files:
        print(f"No .parquet files found in {input_folder}")
        return

    print(f"found {len(parquet_files)} parquet file(s) in {input_folder}")
    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        if add_source_col:
            df["__source_file__"] = f.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    print(f"Combined {len(dfs)} parquets into {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine all Parquet files in a folder into one.")
    parser.add_argument("input_folder", help="Folder containing .parquet files")
    parser.add_argument("output_file", help="Path to the combined output .parquet file")
    parser.add_argument(
        "--no-source-col",
        action="store_true",
        help="Do not add a '__source_file__' column tracking each row's origin file",
    )
    args = parser.parse_args()
 
    combine_parquets(args.input_folder, args.output_file, add_source_col=not args.no_source_col)
 
 
if __name__ == "__main__":
    main()   