import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from . import utils


class PreferenceAnalyzer:
    """Class for analyzing ice cream preferences from survey data."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with survey dataframe."""
        self.df = df

    def plot_horizontal_preferences(
        self,
        data: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Tuple[int, int] = (10, 6),
        show_plot: bool = False
    ):
        """Create horizontal bar plot for preferences."""
        fig, ax = plt.subplots(figsize=figsize)
        
        data.plot(
            kind='barh',
            ax=ax,
            color='skyblue'
        )
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add value labels
        for i, v in enumerate(data):
            ax.text(v, i, f' {v}', va='center')
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        else:
            utils.save_and_close_figure(
                fig, 
                f"{title.lower().replace(' ', '_')}", 
                'preferences'
            )

    def analyze_ice_cream_types(self) -> Dict[str, pd.Series]:
        """Analyze ice cream type preferences."""
        ice_cream_types = (
            self.df['which_type_of_ice_cream_do_you_prefer']
            .str.get_dummies(sep=',')
        )
        
        type_preferences = ice_cream_types.sum().sort_values(ascending=False)
        type_percentages = (type_preferences / len(self.df) * 100).round(1)
        
        return {
            'counts': type_preferences,
            'percentages': type_percentages
        }

    def analyze_purchase_factors(self) -> Dict[str, pd.Series]:
        """Analyze factors influencing purchase decisions."""
        purchase_factors = (
            self.df['which_factors_influence_your_decision_when_purchasing_ice_cream']
            .str.get_dummies(sep=',')
        )
        
        factor_counts = purchase_factors.sum().sort_values(ascending=False)
        factor_percentages = (factor_counts / len(self.df) * 100).round(1)
        
        return {
            'counts': factor_counts,
            'percentages': factor_percentages
        }

    def analyze_flavor_preferences(self) -> Dict[str, pd.Series]:
        """Analyze flavor preferences and willingness to try new flavors."""
        new_flavors = (
            self.df['which_of_the_following_ice_cream_flavor_would_you_try']
            .str.get_dummies(sep=',')
        )
        
        flavor_interest = new_flavors.sum().sort_values(ascending=False)
        flavor_percentages = (flavor_interest / len(self.df) * 100).round(1)
        
        return {
            'counts': flavor_interest,
            'percentages': flavor_percentages
        }

    def analyze_brand_awareness(self) -> Dict[str, pd.Series]:
        """Analyze brand awareness for different ice cream types."""
        brand_columns = [
            col for col in self.df.columns 
            if 'what_brand_comes_to_your_mind' in col
        ]
        
        brand_awareness = {}
        for col in brand_columns:
            product_type = col.split('_')[-1]
            brand_counts = self.df[col].value_counts()
            brand_percentages = (brand_counts / len(self.df) * 100).round(1)
            
            brand_awareness[product_type] = {
                'counts': brand_counts,
                'percentages': brand_percentages
            }
        
        return brand_awareness

    def analyze_satisfaction_likelihood(self) -> Dict[str, pd.Series]:
        """Analyze satisfaction and likelihood of trying new brands."""
        satisfaction = self.df[
            'how_satisfied_are_you_with_the_range_of_flavors_currently_available_in_the_bangladeshi_market'
        ].value_counts()
        
        likelihood = self.df[
            'how_likely_are_you_to_try_a_new_ice_cream_brand'
        ].value_counts()
        
        return {
            'satisfaction': {
                'counts': satisfaction,
                'percentages': (satisfaction / len(self.df) * 100).round(1)
            },
            'likelihood': {
                'counts': likelihood,
                'percentages': (likelihood / len(self.df) * 100).round(1)
            }
        }

    def create_preference_summary(self, results: Dict) -> None:
        """Print summary of all preference analyses."""
        print("\nIce Cream Preference Analysis")
        print("=" * 50)
        
        # Type preferences
        print("\nTop 5 Ice Cream Types:")
        for type_name, percentage in results['types']['percentages'].head().items():
            print(f"{type_name}: {percentage}%")
        
        # Purchase factors
        print("\nTop 5 Purchase Factors:")
        for factor, percentage in results['factors']['percentages'].head().items():
            print(f"{factor}: {percentage}%")
        
        # Flavor preferences
        print("\nTop 5 Desired Flavors:")
        for flavor, percentage in results['flavors']['percentages'].head().items():
            print(f"{flavor}: {percentage}%")
        
        # Brand awareness
        print("\nTop Brand by Product Type:")
        for product_type, data in results['brand_awareness'].items():
            top_brand = data['percentages'].index[0]
            percentage = data['percentages'].iloc[0]
            print(f"{product_type}: {top_brand} ({percentage}%)")
        
        # Satisfaction and likelihood
        print("\nSatisfaction Levels:")
        for level, percentage in results['satisfaction_likelihood']['satisfaction']['percentages'].items():
            print(f"{level}: {percentage}%")

    def create_preference_plots(self, results: Dict, show_plots: bool = False) -> None:
        """Create all preference-related plots.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing all preference analysis results
        show_plots : bool, optional
            If True, display plots (default: False)
        """
        # Plot ice cream type preferences
        self.plot_horizontal_preferences(
            results['types']['counts'],
            'Ice Cream Type Preferences',
            'Number of Consumers',
            'Ice Cream Type',
            show_plot=show_plots
        )
        
        # Plot purchase factors
        self.plot_horizontal_preferences(
            results['factors']['counts'],
            'Purchase Decision Factors',
            'Number of Consumers',
            'Factor',
            show_plot=show_plots
        )
        
        # Plot flavor preferences
        self.plot_horizontal_preferences(
            results['flavors']['counts'],
            'Desired New Flavors',
            'Number of Consumers',
            'Flavor',
            show_plot=show_plots
        )
        
        # Plot satisfaction and likelihood
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Satisfaction plot
        satisfaction_data = results['satisfaction_likelihood']['satisfaction']['counts']
        sns.barplot(x=satisfaction_data.index, y=satisfaction_data.values, ax=ax1, palette='viridis')
        ax1.set_title('Satisfaction with Available Flavors')
        ax1.set_xlabel('Satisfaction Level')
        ax1.set_ylabel('Number of Consumers')
        ax1.tick_params(axis='x', rotation=45)
        
        # Likelihood plot
        likelihood_data = results['satisfaction_likelihood']['likelihood']['counts']
        sns.barplot(x=likelihood_data.index, y=likelihood_data.values, ax=ax2, palette='viridis')
        ax2.set_title('Likelihood to Try New Brands')
        ax2.set_xlabel('Likelihood Level')
        ax2.set_ylabel('Number of Consumers')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if show_plots:
            plt.show()
        else:
            utils.save_and_close_figure(fig, 'satisfaction_likelihood', 'preferences')
        
        # Plot brand awareness by product type
        fig, ax = plt.subplots(figsize=(12, 8))
        brand_data = pd.DataFrame({
            product: data['counts'].head(3) 
            for product, data in results['brand_awareness'].items()
        })
        
        brand_data.plot(kind='bar', ax=ax)
        ax.set_title('Top 3 Brands by Product Type')
        ax.set_xlabel('Brand')
        ax.set_ylabel('Number of Mentions')
        ax.legend(title='Product Type')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if show_plots:
            plt.show()
        else:
            utils.save_and_close_figure(fig, 'brand_awareness', 'preferences')

    def run_preference_analysis(self, show_plots: bool = False) -> Dict:
        """Main entry point for preference analysis."""
        print("Starting preference analysis...")
        
        # Run analyses
        results = {
            'types': self.analyze_ice_cream_types(),
            'factors': self.analyze_purchase_factors(),
            'flavors': self.analyze_flavor_preferences(),
            'brand_awareness': self.analyze_brand_awareness(),
            'satisfaction_likelihood': self.analyze_satisfaction_likelihood()
        }
        
        # Create visualizations
        self.create_preference_plots(results, show_plots)
        
        # Print summary
        self.create_preference_summary(results)
        
        # Save results
        utils.save_analysis_results(results, 'preference_analysis')
        
        print("Preference analysis completed!")
        return results