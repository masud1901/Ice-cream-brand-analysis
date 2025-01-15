"""Main entry point for ice cream market analysis."""
from scripts.analysis_pipeline import run_analysis_pipeline


def main(show_plots: bool = False):
    """Run the analysis."""
    try:
        print("Starting ice cream market analysis...")
        results = run_analysis_pipeline(show_plots)
        print("\nAnalysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main(show_plots=False)