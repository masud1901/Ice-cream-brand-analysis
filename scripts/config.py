"""Configuration settings for ice cream analysis project."""
import os
from pathlib import Path

# Get the absolute path of the project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directory paths
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
REPORTS_DIR = ROOT_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
LOGS_DIR = ROOT_DIR / 'logs'

# Create necessary directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file paths
RAW_DATA_FILE = RAW_DATA_DIR / 'Ice_cream_data.csv'
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / 'cleaned_ice_cream_data.csv'

# Visualization settings
FIGURE_SIZE = (12, 8)
STYLE_SHEET = 'seaborn'
COLOR_PALETTE = 'viridis'

# Analysis settings
RANDOM_SEED = 42

# Verify data file existence
def verify_data_file():
    """Verify that the raw data file exists."""
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nData file not found at {RAW_DATA_FILE}\n"
            f"Please place your data file at: {RAW_DATA_FILE}"
        ) 