import argparse
import gzip
import subprocess
import tarfile
from pathlib import Path

import pandas as pd


def download_corpus(username, password):
    """Download the MIMIC-III corpus from PhysioNet"""
    print("Downloading MIMIC-III corpus from PhysioNet...")

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Construct wget command
    url = "https://physionet.org/files/mimic-iii-ext-verifact-bhc/1.0.0/"
    wget_cmd = [
        "wget",
        "-r",
        "-N",
        "-c",
        "-np",
        "--user",
        username,
        "--password",
        password,
        "--directory-prefix",
        "data",
        url,
    ]

    try:
        print(
            f"Running: wget -r -N -c -np --user {username} --password [HIDDEN] \
                --directory-prefix data {url}"
        )
        subprocess.run(wget_cmd, check=True, capture_output=True, text=True)
        print("Download completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: wget command not found. Please install wget first.")
        return False


def extract_files():
    """Extract GZ files"""
    reference_ehr_dir = Path(
        "data/physionet.org/files/mimic-iii-ext-verifact-bhc/1.0.0/reference_ehr"
    )

    if not reference_ehr_dir.exists():
        print(f"Directory {reference_ehr_dir} does not exist")
        return False

    print(f"Extracting files in: {reference_ehr_dir}")

    # Find all .gz files
    gz_files = list(reference_ehr_dir.glob("*.gz"))

    for gz_file in gz_files:
        print(f"Processing: {gz_file.name}")

        if gz_file.name.endswith(".tar.gz"):
            # Handle tar.gz files
            with tarfile.open(gz_file, "r:gz") as tar:
                tar.extractall(path=reference_ehr_dir)
                print(f"Extracted {gz_file.name}")

        elif gz_file.name.endswith(".csv.gz"):
            # Handle CSV gz files
            csv_name = gz_file.stem
            csv_path = reference_ehr_dir / csv_name

            with gzip.open(gz_file, "rb") as f_in:
                with open(csv_path, "wb") as f_out:
                    f_out.write(f_in.read())

            print(f"Extracted {gz_file.name} to {csv_name}")

    return True


def merge_data():
    """Merge MIMIC-III data to create parquet output"""
    print("Merging data...")

    reference_ehr_dir = Path(
        "data/physionet.org/files/mimic-iii-ext-verifact-bhc/1.0.0/reference_ehr"
    )
    admissions_file = reference_ehr_dir / "admissions.csv"
    noteevents_file = reference_ehr_dir / "ehr_noteevents.csv"
    output_file = Path("data/mimic-iii-verifact-bhc.parquet")

    # Check if files exist
    if not admissions_file.exists():
        print(f"Error: {admissions_file} not found")
        return False

    if not noteevents_file.exists():
        print(f"Error: {noteevents_file} not found")
        return False

    print("Loading noteevents data...")
    noteevents_df = pd.read_csv(noteevents_file)

    print(f"Noteevents shape: {noteevents_df.shape}")

    # Select relevant columns and rename
    result_df = noteevents_df[["ROW_ID", "SUBJECT_ID", "CATEGORY", "TEXT"]].copy()
    result_df = result_df.rename(
        columns={
            "SUBJECT_ID": "subject_id",
            "TEXT": "text",
            "CATEGORY": "category",
            "ROW_ID": "row_id",
        }
    )

    # Clean data - remove rows with missing text
    initial_count = len(result_df)
    result_df = result_df.dropna(subset=["text"])
    final_count = len(result_df)

    print(f"Removed {initial_count - final_count} rows with missing text")
    print(f"Final data shape: {result_df.shape}")

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    print(f"Saving to {output_file}...")
    result_df.to_parquet(output_file, index=False)

    print("Merge completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="MIMIC-III Data Processor")
    parser.add_argument("--user", required=True, help="PhysioNet username")
    parser.add_argument("--password", required=True, help="PhysioNet password")

    args = parser.parse_args()

    print("Starting MIMIC-III processing pipeline...")
    print("=" * 50)

    # Step 1: Download
    if not download_corpus(args.user, args.password):
        print("Download failed. Aborting.")
        return

    # Step 2: Extract
    if not extract_files():
        print("Extraction failed. Aborting.")
        return

    # Step 3: Merge
    if not merge_data():
        print("Merge failed. Aborting.")
        return

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("Output: data/mimic-iii-verifact-bhc.parquet")


if __name__ == "__main__":
    main()
