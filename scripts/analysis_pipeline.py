"""Main analysis pipeline for ice cream survey data."""
from pathlib import Path
from datetime import datetime
from . import config, utils
from .demographic_analysis import DemographicAnalyzer
from .preference_analysis import PreferenceAnalyzer
from .brand_analysis import BrandAnalyzer
from .statistical_analysis import StatisticalAnalyzer
from .advanced_analysis import AdvancedAnalyzer
from .data_cleaning import IceCreamDataCleaner
import sys
from io import StringIO


class OutputCapture:
    """Capture stdout to both display and save to file."""
    
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
        self.output = StringIO()
    
    def write(self, text):
        self.terminal.write(text)
        self.output.write(text)
    
    def flush(self):
        self.terminal.flush()
    
    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(self.output.getvalue())


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.REPORTS_DIR,
        config.FIGURES_DIR,
        config.LOGS_DIR,
        config.REPORTS_DIR / 'summaries',
        config.REPORTS_DIR / 'statistics',
        config.REPORTS_DIR / 'advanced',
        config.FIGURES_DIR / 'demographics',
        config.FIGURES_DIR / 'preferences',
        config.FIGURES_DIR / 'brands',
        config.FIGURES_DIR / 'statistical',
        config.FIGURES_DIR / 'advanced'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def run_analysis_pipeline(show_plots: bool = False):
    """Run complete analysis pipeline."""
    # Setup output capture
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.REPORTS_DIR / f'analysis_report_{timestamp}.txt'
    output_capture = OutputCapture(output_file)
    sys.stdout = output_capture
    
    try:
        # Setup
        setup_directories()
        utils.setup_plotting_style()
        
        print("Starting ice cream market analysis...")
        
        # Initialize analyzers
        data_cleaner = IceCreamDataCleaner()
        
        # Load and clean the raw data
        print("\nLoading and cleaning data...")
        raw_data = utils.load_data(processed=False)
        cleaned_data = data_cleaner.run_data_cleaning(raw_data)
        
        # Initialize analyzers with cleaned data
        demographic_analyzer = DemographicAnalyzer(cleaned_data)
        preference_analyzer = PreferenceAnalyzer(cleaned_data)
        brand_analyzer = BrandAnalyzer(cleaned_data)
        statistical_analyzer = StatisticalAnalyzer(cleaned_data)
        advanced_analyzer = AdvancedAnalyzer(cleaned_data)
        
        # Run analyses
        print("\nRunning demographic analysis...")
        demographic_results = demographic_analyzer.run_demographic_analysis(
            show_plots=show_plots
        )
        
        print("\nRunning preference analysis...")
        preference_results = preference_analyzer.run_preference_analysis(
            show_plots=show_plots
        )
        
        print("\nRunning brand analysis...")
        brand_results = brand_analyzer.run_brand_analysis(
            show_plots=show_plots
        )
        
        print("\nRunning statistical analysis...")
        statistical_results = statistical_analyzer.run_statistical_analysis(
            show_plots=show_plots
        )
        
        print("\nRunning advanced statistical analysis...")
        advanced_results = advanced_analyzer.run_advanced_analysis(
            show_plots=show_plots
        )
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved in: {config.REPORTS_DIR}")
        print(f"Figures saved in: {config.FIGURES_DIR}")
        print(f"Analysis report saved to: {output_file}")
        
        # Save captured output
        output_capture.save()
        
        return {
            'demographic': demographic_results,
            'preference': preference_results,
            'brand': brand_results,
            'statistical': statistical_results,
            'advanced': advanced_results
        }
    
    finally:
        # Restore original stdout
        sys.stdout = output_capture.terminal


if __name__ == "__main__":
    run_analysis_pipeline()