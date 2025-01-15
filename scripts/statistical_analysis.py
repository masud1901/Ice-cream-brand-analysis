"""Statistical analysis module for ice cream survey data."""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, norm, poisson, multinomial
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from . import utils


class StatisticalAnalyzer:
    """Class for performing statistical analysis on ice cream survey data."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with survey dataframe."""
        self.df = df
        self.satisfaction_map = {
            'Very Satisfied': 5,
            'Satisfied': 4,
            'Neutral': 3,
            'Dissatisfied': 2,
            'Very Dissatisfied': 1
        }
        self.likelihood_map = {
            'Very Likely': 5,
            'Likely': 4,
            'Neutral': 3,
            'Unlikely': 2,
            'Very Unlikely': 1
        }
        self.df = self.create_numeric_columns()

    def create_numeric_columns(self) -> pd.DataFrame:
        """Create numeric versions of categorical columns."""
        df = self.df.copy()
        
        # Create numeric satisfaction score (1-5)
        df['satisfaction_score'] = df[
            'how_satisfied_are_you_with_the_range_of_flavors_currently_available_in_the_bangladeshi_market'
        ].map(self.satisfaction_map)
        
        # Create numeric likelihood score (1-5)
        df['likelihood_score'] = df[
            'how_likely_are_you_to_try_a_new_ice_cream_brand'
        ].map(self.likelihood_map)
        
        return df

    def analyze_brand_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between brand preferences and other factors."""
        results = {}
        
        # Chi-square test for brand preference vs demographics
        demographic_vars = ['age', 'gender', 'occupation']
        brand_var = 'which_brand_of_ice_cream_do_you_prefer_the_most'
        
        for demo_var in demographic_vars:
            contingency = pd.crosstab(self.df[demo_var], self.df[brand_var])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            results[f'brand_vs_{demo_var}'] = {
                'test': 'Chi-square test of independence',
                'statistic': chi2,
                'p_value': p_value,
                'dof': dof,
                'contingency_table': contingency
            }
        
        return results

    def analyze_satisfaction_factors(self) -> Dict[str, Any]:
        """Analyze factors affecting satisfaction levels."""
        results = {}
        
        # ANOVA test for satisfaction across income groups
        income_groups = self.df.groupby('which_income_range_do_you_your_family_belong_to_monthly')['satisfaction_score']
        f_stat, p_value = stats.f_oneway(*[group for _, group in income_groups])
        
        results['satisfaction_by_income'] = {
            'test': 'One-way ANOVA',
            'statistic': f_stat,
            'p_value': p_value,
            'means': income_groups.mean()
        }
        
        # T-test for satisfaction between genders
        gender_groups = [group for _, group in self.df.groupby('gender')['satisfaction_score']]
        t_stat, p_value = stats.ttest_ind(*gender_groups)
        
        results['satisfaction_by_gender'] = {
            'test': 'Independent t-test',
            'statistic': t_stat,
            'p_value': p_value,
            'means': self.df.groupby('gender')['satisfaction_score'].mean()
        }
        
        return results

    def analyze_price_sensitivity(self) -> Dict[str, Any]:
        """Analyze price sensitivity across different demographics."""
        results = {}
        
        # Create price sensitivity indicator
        self.df['price_sensitive'] = self.df['which_factors_influence_your_decision_when_purchasing_ice_cream'].str.contains('Price').astype(int)
        
        # Chi-square test for price sensitivity vs income
        contingency = pd.crosstab(
            self.df['which_income_range_do_you_your_family_belong_to_monthly'],
            self.df['price_sensitive']
        )
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        results['price_sensitivity_vs_income'] = {
            'test': 'Chi-square test of independence',
            'statistic': chi2,
            'p_value': p_value,
            'dof': dof,
            'contingency_table': contingency,
            'percentage_by_income': (contingency[1] / contingency.sum(axis=1) * 100).round(1)
        }
        
        return results

    def analyze_correlations(self, show_plots: bool = False) -> Dict[str, Any]:
        """Analyze correlations between numeric variables."""
        # Select numeric columns
        numeric_cols = ['satisfaction_score', 'likelihood_score', 'num_types_preferred',
                       'num_purchase_factors', 'num_marketing_channels']
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Create correlation heatmap
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    square=True)
        plt.title('Correlation Matrix of Key Metrics')
        
        if show_plots:
            plt.show()
        else:
            utils.save_and_close_figure(fig, 'correlation_matrix', 'statistical')
        
        return {
            'correlation_matrix': corr_matrix
        }

    def analyze_purchase_patterns(self) -> Dict[str, Any]:
        """Analyze purchase patterns using probability distributions."""
        results = {}
        
        # Analyze number of ice cream types preferred (Poisson distribution)
        num_types = self.df['num_types_preferred']
        lambda_mle = num_types.mean()
        
        # Get observed frequencies
        observed_freq = num_types.value_counts().sort_index()
        
        # Generate expected frequencies for the same range as observed
        k = np.arange(observed_freq.index.min(), observed_freq.index.max() + 1)
        poisson_pmf = stats.poisson.pmf(k, lambda_mle)
        
        # Normalize expected frequencies to match observed total
        expected_freq = pd.Series(
            poisson_pmf * len(self.df), 
            index=k
        ).reindex(observed_freq.index).fillna(0)
        
        # Scale expected frequencies to match observed total
        expected_freq = expected_freq * (observed_freq.sum() / expected_freq.sum())
        
        # Now both observed and expected have same total
        chi2, p_value = stats.chisquare(observed_freq, expected_freq)
        
        results['types_poisson'] = {
            'lambda': lambda_mle,
            'chi2_stat': chi2,
            'p_value': p_value,
            'observed': observed_freq,
            'expected': expected_freq
        }
        
        # Analyze satisfaction scores (Normal distribution)
        satisfaction = self.df['satisfaction_score']
        mu, std = stats.norm.fit(satisfaction)
        
        # Test for normality using D'Agostino and Pearson's test
        normality_stat, norm_p_value = stats.normaltest(satisfaction)
        
        # Create histogram data with fixed bins for better visualization
        hist_data = np.histogram(
            satisfaction, 
            bins=np.linspace(1, 5, 9),
            density=True
        )
        
        results['satisfaction_normal'] = {
            'mean': mu,
            'std': std,
            'normality_stat': normality_stat,
            'p_value': norm_p_value,
            'histogram': hist_data
        }
        
        # Analyze purchase factors distribution
        factor_counts = self.df['num_purchase_factors']
        factor_mean = factor_counts.mean()
        factor_std = factor_counts.std()
        
        # Test for normality
        _, factor_p_value = stats.normaltest(factor_counts)
        
        results['purchase_factors'] = {
            'mean': factor_mean,
            'std': factor_std,
            'p_value': factor_p_value,
            'distribution': factor_counts.value_counts().sort_index()
        }
        
        return results

    def create_markov_chain(self) -> Dict[str, Any]:
        """Create Markov chain for brand preferences and transitions."""
        # Get brand columns
        brand_columns = [
            'which_brand_of_ice_cream_do_you_prefer_the_most',
            'which_brand_of_ice_cream_do_you_consider_the_most_premium',
            'which_brand_of_ice_cream_do_you_consider_the_tastiest'
        ]
        
        # Get common brands across all columns
        common_brands = set(self.df[brand_columns[0]].unique())
        for col in brand_columns[1:]:
            common_brands = common_brands.intersection(set(self.df[col].unique()))
        
        # Filter data to include only common brands
        mask = self.df[brand_columns[0]].isin(common_brands) & \
               self.df[brand_columns[1]].isin(common_brands) & \
               self.df[brand_columns[2]].isin(common_brands)
        
        filtered_df = self.df[mask]
        
        # Create transition matrix
        transition_matrix = pd.DataFrame(
            0, 
            index=sorted(common_brands), 
            columns=sorted(common_brands)
        )
        
        # Calculate transitions between preferred -> premium -> tasty
        for _, row in filtered_df.iterrows():
            pref = row[brand_columns[0]]
            prem = row[brand_columns[1]]
            tasty = row[brand_columns[2]]
            
            transition_matrix.loc[pref, prem] += 1
            transition_matrix.loc[prem, tasty] += 1
        
        # Normalize to get probabilities
        transition_probs = transition_matrix.div(
            transition_matrix.sum(axis=1), 
            axis=0
        ).fillna(0)
        
        # Calculate steady state
        try:
            eigenvals, eigenvects = np.linalg.eig(transition_probs.T)
            steady_state = pd.Series(
                eigenvects[:, 0].real / eigenvects[:, 0].real.sum(),
                index=transition_probs.index
            )
        except np.linalg.LinAlgError:
            # Fallback if eigenvalue calculation fails
            steady_state = transition_probs.mean()
        
        return {
            'transition_matrix': transition_matrix,
            'transition_probabilities': transition_probs,
            'steady_state': steady_state,
            'n_transitions': len(filtered_df)
        }

    def analyze_multinomial_preferences(self) -> Dict[str, Any]:
        """Analyze preferences using multinomial distribution."""
        # Get ice cream type columns
        type_cols = [col for col in self.df.columns if col.startswith('prefers_type_')]
        type_counts = self.df[type_cols].sum()
        
        # Calculate probabilities
        n_trials = type_counts.sum()
        probs = type_counts / n_trials
        
        # Get top preferences
        top_prefs = probs.nlargest(5)
        
        return {
            'probabilities': probs,
            'top_preferences': top_prefs,
            'n_trials': n_trials,
            'chi_square_stat': stats.chisquare(type_counts)[0],
            'p_value': stats.chisquare(type_counts)[1]
        }

    def plot_distributions(self, results: Dict[str, Any], show_plots: bool = False) -> None:
        """Plot probability distributions and Markov chain results."""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Poisson fit for number of types
        poisson_data = results['purchase_patterns']['types_poisson']
        axes[0, 0].bar(
            poisson_data['observed'].index, 
            poisson_data['observed'].values, 
            alpha=0.5, 
            label='Observed',
            color='skyblue'
        )
        axes[0, 0].plot(
            poisson_data['expected'].index, 
            poisson_data['expected'].values, 
            'r-', 
            label='Expected (Poisson)',
            linewidth=2
        )
        axes[0, 0].set_title('Number of Ice Cream Types Preferred\n'
                            'Poisson Distribution Fit')
        axes[0, 0].set_xlabel('Number of Types')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Plot 2: Normal distribution fit for satisfaction
        norm_data = results['purchase_patterns']['satisfaction_normal']
        hist_counts, hist_bins = norm_data['histogram']
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        
        axes[0, 1].bar(
            bin_centers, 
            hist_counts,
            width=np.diff(hist_bins)[0] * 0.8,
            alpha=0.5,
            color='skyblue',
            label='Observed'
        )
        
        x = np.linspace(1, 5, 100)
        axes[0, 1].plot(
            x, 
            stats.norm.pdf(x, norm_data['mean'], norm_data['std']), 
            'r-', 
            label='Normal fit',
            linewidth=2
        )
        axes[0, 1].set_title('Satisfaction Score Distribution\n'
                            'Normal Distribution Fit')
        axes[0, 1].set_xlabel('Satisfaction Score')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        
        # Plot 3: Markov chain transition probabilities
        markov_data = results['markov_chain']['transition_probabilities']
        sns.heatmap(
            markov_data, 
            ax=axes[1, 0], 
            cmap='YlOrRd', 
            annot=True, 
            fmt='.2f'
        )
        axes[1, 0].set_title('Brand Transition Probabilities')
        
        # Plot 4: Correlation heatmap
        sns.heatmap(
            results['correlations']['correlation_matrix'],
            ax=axes[1, 1],
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f'
        )
        axes[1, 1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        
        if show_plots:
            plt.show()
        else:
            utils.save_and_close_figure(fig, 'probability_distributions', 'statistical')

    def print_statistical_summary(self, results: Dict[str, Any]) -> None:
        """Print summary of statistical analyses."""
        print("\nStatistical Analysis Summary")
        print("=" * 50)
        
        # Brand relationships
        print("\nBrand Preference Analysis:")
        print("-" * 30)
        for test_name, test_results in results['brand_relationships'].items():
            if 'p_value' in test_results:
                print(f"\n{test_name.replace('_', ' ').title()}:")
                print(f"Test: {test_results['test']}")
                print(f"p-value: {test_results['p_value']:.4f}")
                print(f"Significant: {'Yes' if test_results['p_value'] < 0.05 else 'No'}")
        
        # Satisfaction analysis
        print("\nSatisfaction Analysis:")
        print("-" * 30)
        for test_name, test_results in results['satisfaction_factors'].items():
            print(f"\n{test_name.replace('_', ' ').title()}:")
            print(f"Test: {test_results['test']}")
            print(f"p-value: {test_results['p_value']:.4f}")
            print(f"Significant: {'Yes' if test_results['p_value'] < 0.05 else 'No'}")
        
        # Price sensitivity
        print("\nPrice Sensitivity Analysis:")
        print("-" * 30)
        price_results = results['price_sensitivity']['price_sensitivity_vs_income']
        print(f"Test: {price_results['test']}")
        print(f"p-value: {price_results['p_value']:.4f}")
        print(f"Significant: {'Yes' if price_results['p_value'] < 0.05 else 'No'}")
        
        # Print percentage of price-sensitive consumers by income
        print("\nPrice Sensitivity by Income Group:")
        for income, percentage in price_results['percentage_by_income'].items():
            print(f"{income}: {percentage}% price-sensitive")

    def print_advanced_statistical_summary(self, results: Dict[str, Any]) -> None:
        """Print summary of advanced statistical analyses."""
        print("\nAdvanced Statistical Analysis Summary")
        print("=" * 50)
        
        # Purchase patterns
        print("\nPurchase Pattern Analysis:")
        print("-" * 30)
        poisson_results = results['purchase_patterns']['types_poisson']
        print(f"Poisson Distribution λ: {poisson_results['lambda']:.2f}")
        print(f"Goodness of fit p-value: {poisson_results['p_value']:.4f}")
        
        norm_results = results['purchase_patterns']['satisfaction_normal']
        print(f"\nSatisfaction Score Distribution:")
        print(f"Mean: {norm_results['mean']:.2f}")
        print(f"Standard Deviation: {norm_results['std']:.2f}")
        print(f"Normality test p-value: {norm_results['p_value']:.4f}")
        
        # Markov Chain
        print("\nMarkov Chain Analysis:")
        print("-" * 30)
        steady_state = results['markov_chain']['steady_state']
        print("\nSteady State Probabilities:")
        for brand, prob in steady_state.nlargest(3).items():
            print(f"{brand}: {prob:.2%}")
        
        # Multinomial Analysis
        print("\nMultinomial Preference Analysis:")
        print("-" * 30)
        multi_results = results['multinomial']['type_preferences']
        probs = multi_results['probabilities']
        print("\nPreference Probabilities:")
        for type_name, prob in probs.nlargest(3).items():
            print(f"{type_name}: {prob:.2%}")

    def run_statistical_analysis(self, show_plots: bool = False) -> Dict:
        """Main entry point for statistical analysis."""
        print("Starting statistical analysis...")
        
        # Run analyses
        results = {
            'brand_relationships': self.analyze_brand_relationships(),
            'satisfaction_factors': self.analyze_satisfaction_factors(),
            'price_sensitivity': self.analyze_price_sensitivity(),
            'correlations': self.analyze_correlations(show_plots),
            'purchase_patterns': self.analyze_purchase_patterns(),
            'markov_chain': self.create_markov_chain(),
            'multinomial': {
                'type_preferences': self.analyze_multinomial_preferences()
            }
        }
        
        # Create visualizations
        self.plot_distributions(results, show_plots)
        
        # Print summaries
        print("\nStatistical Analysis Summary")
        print("=" * 50)
        self.print_statistical_summary(results)
        print("\nAdvanced Statistical Analysis Summary")
        print("=" * 50)
        self.print_advanced_statistical_summary(results)
        
        # Save results
        utils.save_analysis_results(results, 'statistical_analysis')
        
        print("Statistical analysis completed!")
        return results