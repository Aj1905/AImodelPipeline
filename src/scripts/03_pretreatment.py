import sys
from importlib import import_module
from pathlib import Path

# Add project root to sys.path so that 'src' is importable
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the refactored pretreatment package and expose the main entry point
run_pretreatment = import_module("src.pretreatment.03").run_pretreatment

if __name__ == "__main__":
    run_pretreatment()
