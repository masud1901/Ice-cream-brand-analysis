import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from . import utils


class BrandAnalyzer:
    """Class for analyzing ice cream brand preferences and perceptions."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with survey dataframe."""
        self.df = df

    def get_brand_preferences(self) -> Dict[str, pd.Series]:
        """Analyze basic brand preferences.
        
        Returns:
        --------
        dict
            Dictionary containing brand preferences with counts and percentages
        """
        # Get brand preferences
        preferred_brands = self.df[
            'which_brand_of_ice_cream_do_you_prefer_the_most'
        ].value_counts()
        preferred_percentages = (preferred_brands / len(self.df) * 100).round(1)
        
        return {
            'counts': preferred_brands,
            'percentages': preferred_percentages
        }

    def get_brand_perceptions(self) -> Dict[str, Dict[str, pd.Series]]:
        """Analyze brand perceptions (premium, tasty, etc.).
        
        Returns:
        --------
        dict
            Dictionary containing different brand perceptions
        """
        # Premium perception
        premium_brands = self.df[
            'which_brand_of_ice_cream_do_you_consider_the_most_premium'
        ].value_counts()
        premium_percentages = (premium_brands / len(self.df) * 100).round(1)
        
        # Taste perception
        tasty_brands = self.df[
            'which_brand_of_ice_cream_do_you_consider_the_tastiest'
        ].value_counts()
        tasty_percentages = (tasty_brands / len(self.df) * 100).round(1)
        
        # Least preferred
        least_preferred = self.df[
            'which_brand_of_ice_cream_do_you_prefer_the_least'
        ].value_counts()
        least_percentages = (least_preferred / len(self.df) * 100).round(1)
        
        return {
            'premium': {
                'counts': premium_brands,
                'percentages': premium_percentages
            },
            'tasty': {
                'counts': tasty_brands,
                'percentages': tasty_percentages
            },
            'least_preferred': {
                'counts': least_preferred,
                'percentages': least_percentages
            }
        }

    def plot_brand_comparison(
        self,
        data: pd.Series,
        title: str,
        figsize: Tuple[int, int] = (10, 6),
        show_plot: bool = False
    ) -> None:
        """Create bar plot for brand comparison.
        
        Parameters:
        -----------
        data : pandas.Series
            Brand data to plot
        title : str
            Plot title
        figsize : tuple, optional
            Figure size
        show_plot : bool, optional
            If True, display the plot (default: False)
        """
        fig = plt.figure(figsize=figsize)
        sns.barplot(x=data.values, y=data.index, palette='viridis')
        plt.title(title, pad=15, fontsize=12)
        plt.xlabel('Number of Consumers')
        plt.ylabel('Brand')
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        else:
            utils.save_and_close_figure(
                fig, 
                f"{title.lower().replace(' ', '_')}", 
                'brands'
            )

    def create_brand_heatmap(self, show_plot: bool = False) -> None:
        """Create heatmap showing brand preferences across different questions."""
        # Get brand-related columns
        brand_columns = [
            col for col in self.df.columns 
            if 'brand' in col.lower() and 'mind' not in col.lower()
        ]
        
        # Convert all values to strings before value_counts to ensure consistent types
        brand_data = self.df[brand_columns].astype(str).apply(
            lambda x: pd.Series(x).value_counts()
        ).fillna(0)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            brand_data,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd'
        )
        plt.title('Brand Preferences Across Different Questions')
        plt.xlabel('Questions')
        plt.ylabel('Brands')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        else:
            utils.save_and_close_figure(
                plt.gcf(),
                'brand_preferences_heatmap',
                'brands'
            )

    def print_brand_summary(
        self,
        preferences: Dict[str, pd.Series],
        perceptions: Dict[str, Dict[str, pd.Series]]
    ) -> None:
        """Print summary of brand analysis.
        
        Parameters:
        -----------
        preferences : dict
            Brand preferences data
        perceptions : dict
            Brand perceptions data
        """
        print("\nBrand Analysis Summary")
        print("=" * 50)
        
        print("\nTop 3 Most Preferred Brands:")
        for brand, percentage in preferences['percentages'].head(3).items():
            print(f"{brand}: {percentage}%")
        
        print("\nTop 3 Premium Brands:")
        for brand, percentage in perceptions['premium']['percentages'].head(3).items():
            print(f"{brand}: {percentage}%")
        
        print("\nTop 3 Tastiest Brands:")
        for brand, percentage in perceptions['tasty']['percentages'].head(3).items():
            print(f"{brand}: {percentage}%")

    def run_brand_analysis(self, show_plots: bool = False) -> Dict:
        """Main entry point for brand analysis."""
        print("Starting brand analysis...")
        
        # Get brand analyses
        preferences = self.get_brand_preferences()
        perceptions = self.get_brand_perceptions()
        
        # Create visualizations
        self.create_brand_plots(preferences, perceptions, show_plots)
        
        # Print summary
        self.print_brand_summary(preferences, perceptions)
        
        # Save results
        results = {
            'preferences': preferences,
            'perceptions': perceptions
        }
        utils.save_analysis_results(results, 'brand_analysis')
        
        print("Brand analysis completed!")
        return results 

    def create_brand_plots(
        self,
        preferences: Dict[str, pd.Series],
        perceptions: Dict[str, Dict[str, pd.Series]], 
        show_plots: bool = False
    ) -> None:
        """Create all brand-related plots.
        
        Parameters:
        -----------
        preferences : dict
            Brand preferences data
        perceptions : dict
            Brand perceptions data
        show_plots : bool, optional
            If True, display plots (default: False)
        """
        # Plot brand preferences
        self.plot_brand_comparison(
            preferences['counts'],
            'Most Preferred Ice Cream Brands',
            figsize=(12, 6),
            show_plot=show_plots
        )
        
        # Plot premium perception
        self.plot_brand_comparison(
            perceptions['premium']['counts'],
            'Premium Brand Perception',
            figsize=(12, 6),
            show_plot=show_plots
        )
        
        # Plot taste perception
        self.plot_brand_comparison(
            perceptions['tasty']['counts'],
            'Tastiest Brands Perception',
            figsize=(12, 6),
            show_plot=show_plots
        )
        
        # Plot least preferred brands
        self.plot_brand_comparison(
            perceptions['least_preferred']['counts'],
            'Least Preferred Brands',
            figsize=(12, 6),
            show_plot=show_plots
        )
        
        # Create brand perception heatmap
        self.create_brand_heatmap(show_plot=show_plots)