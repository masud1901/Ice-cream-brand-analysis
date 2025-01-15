"""Utility functions for ice cream analysis."""
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from . import config

logger = logging.getLogger(__name__)

def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use(config.STYLE_SHEET)
    plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def save_figure(fig: plt.Figure, filename: str, subdir: Optional[str] = None):
    """Save figure to reports directory."""
    save_dir = config.FIGURES_DIR
    if subdir:
        save_dir = save_dir / subdir
        save_dir.mkdir(exist_ok=True)
    
    filepath = save_dir / f"{filename}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Figure saved to: {filepath}")

def load_data(processed: bool = False) -> pd.DataFrame:
    """Load data from file.
    
    Parameters:
    -----------
    processed : bool, optional
        If True, load processed data, otherwise load raw data
    
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    try:
        # Verify raw data file exists
        if not processed:
            config.verify_data_file()
        
        file_path = config.CLEANED_DATA_FILE if processed else config.RAW_DATA_FILE
        logger.info(f"Loading data from: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_analysis_results(results: Dict[str, Any], filename: str):
    """Save analysis results to reports directory.
    
    Parameters:
    -----------
    results : dict
        Analysis results to save
    filename : str
        Base name for the saved file
    """
    try:
        # Create summaries directory if it doesn't exist
        save_dir = config.REPORTS_DIR / 'summaries'
        save_dir.mkdir(exist_ok=True)
        
        # Convert results to DataFrames and save as CSV
        for key, value in results.items():
            if isinstance(value, pd.Series):
                # For simple series, convert to DataFrame
                df = value.to_frame(name='count')
                df.index.name = 'category'
                save_path = save_dir / f"{filename}_{key}.csv"
                df.to_csv(save_path)
            elif isinstance(value, pd.DataFrame):
                # For DataFrames, save directly
                save_path = save_dir / f"{filename}_{key}.csv"
                value.to_csv(save_path)
            elif isinstance(value, dict):
                # For nested dictionaries (like brand analysis results)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.Series):
                        df = sub_value.to_frame(name='count')
                        df.index.name = 'category'
                        save_path = save_dir / f"{filename}_{key}_{sub_key}.csv"
                        df.to_csv(save_path)
                    elif isinstance(sub_value, pd.DataFrame):
                        save_path = save_dir / f"{filename}_{key}_{sub_key}.csv"
                        sub_value.to_csv(save_path)
        
        logger.info(f"Results saved to CSV files in: {save_dir}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def save_and_close_figure(fig: plt.Figure, filename: str, subdir: Optional[str] = None):
    """Save figure and close it without displaying.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Name for the saved file
    subdir : str, optional
        Subdirectory within figures directory
    """
    save_figure(fig, filename, subdir)
    plt.close(fig)  # Close figure to free memory 