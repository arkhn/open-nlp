import argparse
import gzip
import subprocess
import tarfile
from pathlib import Path
from typing import List

import pandas as pd


class MIMICDataProcessor:
    """Processes MIMIC-III corpus data with optimized data extraction and cleaning"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.reference_ehr_dir = (
            self.data_dir / "physionet.org/files/mimic-iii-ext-verifact-bhc/1.0.0/reference_ehr"
        )
        self.output_file = self.data_dir / "mimic-iii-verifact-bhc.parquet"

        # Single source of truth for all column mappings
        self.column_mappings = {
            "ROW_ID": "row_id",
            "SUBJECT_ID": "subject_id",
            "CATEGORY": "category",
            "TEXT": "text",
            "CHARTDATE": "chart_date",
            "CHARTTIME": "chart_time",
            "STORETIME": "store_time",
        }

    def download_corpus(self, username: str, password: str) -> bool:
        """Download the MIMIC-III corpus from PhysioNet"""
        print("Downloading MIMIC-III corpus from PhysioNet...")

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

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
            str(self.data_dir),
            url,
        ]

        try:
            print(
                f"Running: wget -r -N -c -np --user {username} \
                    --password [HIDDEN] --directory-prefix {self.data_dir} {url}"
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

    def extract_files(self) -> bool:
        """Extract GZ files with improved error handling"""
        if not self.reference_ehr_dir.exists():
            print(f"Directory {self.reference_ehr_dir} does not exist")
            return False

        print(f"Extracting files in: {self.reference_ehr_dir}")

        # Find all .gz files
        gz_files = list(self.reference_ehr_dir.glob("*.gz"))

        if not gz_files:
            print("No .gz files found to extract")
            return False

        extracted_count = 0
        for gz_file in gz_files:
            try:
                print(f"Processing: {gz_file.name}")

                if gz_file.name.endswith(".tar.gz"):
                    self._extract_tar_gz(gz_file)
                elif gz_file.name.endswith(".csv.gz"):
                    self._extract_csv_gz(gz_file)
                else:
                    print(f"Unknown file type: {gz_file.name}")
                    continue

                extracted_count += 1

            except Exception as e:
                print(f"Error extracting {gz_file.name}: {e}")
                continue

        print(f"Successfully extracted {extracted_count}/{len(gz_files)} files")
        return extracted_count > 0

    def _extract_tar_gz(self, gz_file: Path) -> None:
        """Extract tar.gz files"""
        with tarfile.open(gz_file, "r:gz") as tar:
            tar.extractall(path=self.reference_ehr_dir)
        print(f"Extracted {gz_file.name}")

    def _extract_csv_gz(self, gz_file: Path) -> None:
        """Extract CSV gz files"""
        csv_name = gz_file.stem
        csv_path = self.reference_ehr_dir / csv_name

        with gzip.open(gz_file, "rb") as f_in:
            with open(csv_path, "wb") as f_out:
                f_out.write(f_in.read())

        print(f"Extracted {gz_file.name} to {csv_name}")

    def _get_available_columns(self, df: pd.DataFrame) -> List[str]:
        """Get all available columns from the dataframe that we want to process"""
        df_columns = set(df.columns)  # Convert to set for faster lookups

        # Get all columns we want to process (base + date columns)
        available_columns = [col for col in self.column_mappings.keys() if col in df_columns]

        return available_columns

    def _get_date_columns(self, available_columns: List[str]) -> List[str]:
        """Get list of date columns that exist in the available columns"""
        date_column_names = ["CHARTDATE", "CHARTTIME", "STORETIME"]
        return [col for col in date_column_names if col in available_columns]

    def _convert_date_columns(self, df: pd.DataFrame, date_column_names: List[str]) -> None:
        """Convert date columns to datetime with error handling"""
        # Only process date columns that actually exist in the dataframe
        available_date_columns = [col for col in date_column_names if col in df.columns]

        for date_col in available_date_columns:
            print(f"Converting {date_col} to datetime...")

            try:
                # Convert the original column in place
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                non_null_count = df[date_col].notna().sum()
                print(f"  {date_col}: {non_null_count}/{len(df)} non-null values")

                if non_null_count > 0:
                    print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")

            except Exception as e:
                print(f"  Error converting {date_col}: {e}")
                # Continue with other columns even if one fails

    def merge_data(self) -> bool:
        """Merge MIMIC-III data to create parquet output with optimized processing"""
        print("Merging data...")

        # Check if required files exist
        required_files = {
            "admissions": self.reference_ehr_dir / "admissions.csv",
            "noteevents": self.reference_ehr_dir / "ehr_noteevents.csv",
        }

        for name, file_path in required_files.items():
            if not file_path.exists():
                print(f"Error: {name} file not found at {file_path}")
                return False

        try:
            # Load noteevents data
            print("Loading noteevents data...")
            noteevents_df = pd.read_csv(required_files["noteevents"])
            print(f"Noteevents shape: {noteevents_df.shape}")
            print(f"Available columns: {list(noteevents_df.columns)}")

            # Get available columns
            available_columns = self._get_available_columns(noteevents_df)

            if not available_columns:
                print("Error: No required columns found")
                return False

            print(f"Found columns to process: {available_columns}")

            # Select columns and create mapping
            result_df = noteevents_df[available_columns].copy()

            # Convert date columns BEFORE renaming (so we can access original column names)go
            date_columns_to_convert = self._get_date_columns(available_columns)
            if date_columns_to_convert:
                self._convert_date_columns(result_df, date_columns_to_convert)

            # Create column mapping and rename columns
            column_mapping = {col: self.column_mappings[col] for col in available_columns}
            result_df = result_df.rename(columns=column_mapping)

            # Clean data - remove rows with missing text
            initial_count = len(result_df)
            result_df = result_df.dropna(subset=["text"])
            final_count = len(result_df)

            removed_count = initial_count - final_count
            print(f"Removed {removed_count} rows with missing text")
            print(f"Final data shape: {result_df.shape}")

            # Save as parquet
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving to {self.output_file}...")
            result_df.to_parquet(self.output_file, index=False)

            print("Merge completed successfully!")
            return True

        except Exception as e:
            print(f"Error during data merge: {e}")
            return False

    def process(self, username: str, password: str) -> bool:
        """Run the complete MIMIC-III processing pipeline"""
        print("Starting MIMIC-III processing pipeline...")
        print("=" * 50)

        # Step 1: Download
        if not self.download_corpus(username, password):
            print("Download failed. Aborting.")
            return False

        # Step 2: Extract
        if not self.extract_files():
            print("Extraction failed. Aborting.")
            return False

        # Step 3: Merge
        if not self.merge_data():
            print("Merge failed. Aborting.")
            return False

        print("\n" + "=" * 50)
        print("Pipeline completed successfully!")
        print(f"Output: {self.output_file}")
        return True


def main():
    """Main entry point with improved argument handling"""
    parser = argparse.ArgumentParser(
        description="MIMIC-III Data Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --user myuser --password mypass
  %(prog)s --user myuser --password mypass --data-dir custom_data
        """,
    )

    parser.add_argument("--user", required=True, help="PhysioNet username")
    parser.add_argument("--password", required=True, help="PhysioNet password")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")

    args = parser.parse_args()

    # Initialize processor and run pipeline
    processor = MIMICDataProcessor(data_dir=args.data_dir)
    success = processor.process(args.user, args.password)

    # Exit with appropriate code
    import sys

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
