import sys
from pathlib import Path

from llm_baselines.llm_baseline import main

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    main()
