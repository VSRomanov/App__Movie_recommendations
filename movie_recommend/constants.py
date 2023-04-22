from pathlib import Path

CODE_DIR = Path(__file__).parents[0]
REPO_DIR = CODE_DIR.parents[0]
DATA_DIR = REPO_DIR / "raw_data"
DATA_FULL_DIR = DATA_DIR / "ml-latest"
DATA_SMALL_DIR = DATA_DIR / "ml-latest-small"

PKL_DIR = REPO_DIR / "app_data"

OUTPUT_DIR = REPO_DIR / "output"
OUTPUT_FIG = OUTPUT_DIR / "figures"

# import pdb; pdb.set_trace()
