# Clinical Document Conflict Pipeline

A modular Python pipeline that uses the Groq API models to implement a three-agent system for
clinical document editing and validation. The system creates conflicts between clinical documents
and validates the modifications through an automated workflow.

## System Architecture

### Three-Agent System

1. **Doctor Agent**

   - Analyzes two clinical documents
   - Determines the most appropriate conflict type to introduce
   - Provides specific modification instructions

2. **Editor Agent**

   - Modifies documents based on Doctor Agent's instructions
   - Creates realistic clinical conflicts while maintaining document integrity
   - Provides detailed change summaries

3. **Moderator Agent**
   - Validates the Editor's modifications
   - Scores the quality and realism of conflicts
   - Approves or rejects modifications

### Pipeline Flow

```
Document Pair → Doctor Agent → Editor Agent → Moderator Agent → Database
                                     ↑              ↓
                                     └── Retry Loop (if invalid) ──┘
```

## Setup

### 1. Prerequisites

- Python 3.8+
- Groq API key
- Clinical document dataset (MIMIC-III or sample data)

### 2. Installation

```bash
# Clone or download the pipeline files
cd libs/conflicts

# Install dependencies
pip install -r requirements.txt

# Set up your Groq API key
export GROQ_API_KEY="your_groq_api_key_here"
```

### 3. Data Setup

Option A: Use MIMIC-III dataset

```bash
# Run preprocessing script (requires PhysioNet credentials)
python preprocess_dataset.py --user your_username --password your_password
```

Option B: Use sample data (for testing)

```bash
# Sample data will be automatically created when running the pipeline
# if no dataset is found
```

## Usage

### Command Line Interface

The pipeline provides several commands for different use cases:

#### Process Document Batches

```bash
# Process 5 document pairs with default settings
python main.py batch

# Process 10 pairs with custom settings
python main.py batch --size 10 --max-retries 5 --min-score 80

# Process pairs from same subjects only
python main.py batch --size 3 --same-subject

# Filter by document categories
python main.py batch --categories "Discharge summary" "Progress note"
```

#### Test Specific Conflict Types

```bash
# Test opposition conflicts
python main.py test-conflict opposition

# Test anatomical conflicts with more pairs
python main.py test-conflict anatomical --count 5
```

#### View Statistics

```bash
# Show pipeline and dataset statistics
python main.py stats

# Show database information
python main.py db-info

# List all available conflict types
python main.py list-conflicts
```

## Healthcare Data Viewer

A user-friendly web interface designed specifically for healthcare domain experts to review and
validate conflict detection results without requiring technical background.

### Quick Start

```bash
cd lib/conflicts
pip install -r requirements.txt
streamlit run scripts/clinical_data_viewer.py
```

### Using Custom Data File

```bash
# Use a different parquet file as default
streamlit run scripts/clinical_data_viewer.py -- --data-file "path/to/your/data.parquet"

# Use the default file (same as Quick Start)
streamlit run scripts/clinical_data_viewer.py -- --data-file "processed/186fbae0_02092025.parquet"
```

## License

This pipeline is for research and educational purposes. Please ensure compliance with data usage
agreements when using clinical datasets.
