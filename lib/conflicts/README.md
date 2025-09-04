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

```bash
# Run with smaller dataset for testing
python -m conflicts.run pipeline.dataset_size=5

# Run with specific conflict types
python -m conflicts.run doctor.conflict_types=[opposition,anatomical]

# Run with custom model
python -m conflicts.run model.name="openai/gpt-4o-mini"

# Run with multiple overrides
python -m conflicts.run pipeline.dataset_size=10 doctor.conflict_types=[opposition,anatomical] model.name="openai/gpt-4o-mini"
```

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review logs in `pipeline.log`
3. Test with sample data first
4. Verify API connectivity
