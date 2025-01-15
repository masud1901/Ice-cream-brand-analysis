"""Advanced statistical analysis module for ice cream survey data."""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2, f, t, norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from . import utils


class AdvancedAnalyzer:
    """Class for performing advanced statistical analysis on ice cream survey data."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with survey dataframe."""
        self.df = df
        self.df_clean = self.handle_missing_data(df)

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data through imputation."""
        df_clean = df.copy()
        
        # Numeric imputation
        numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        num_imputer = SimpleImputer(strategy='mean')
        df_clean[numeric_cols] = num_imputer.fit_transform(df_clean[numeric_cols])
        
        # Categorical imputation
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_clean[cat_cols] = cat_imputer.fit_transform(df_clean[cat_cols])
        
        return df_clean

    def detect_outliers(self, columns: list) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        outliers = {}
        
        for col in columns:
            Q1 = self.df_clean[col].quantile(0.25)
            Q3 = self.df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers[col] = {
                'outliers': self.df_clean[(self.df_clean[col] < lower_bound) | 
                                        (self.df_clean[col] > upper_bound)][col],
                'bounds': (lower_bound, upper_bound)
            }
        
        return outliers

    def run_pca_analysis(self, columns: list) -> Dict[str, Any]:
        """Perform PCA on selected features."""
        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df_clean[columns])
        
        # Apply PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Create scree plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'bo-')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance Ratio')
        ax.set_title('PCA Scree Plot')
        
        utils.save_and_close_figure(fig, 'pca_scree_plot', 'advanced')
        
        return {
            'pca_components': pca_result,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'feature_importance': pd.DataFrame(
                pca.components_,
                columns=columns
            )
        }

    def perform_cluster_analysis(self, features: list) -> Dict[str, Any]:
        """Perform k-means clustering."""
        # Prepare data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df_clean[features])
        
        # Find optimal number of clusters using elbow method
        inertias = []
        K = range(1, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(K, inertias, 'bo-')
        ax.set_xlabel('k')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal k')
        
        utils.save_and_close_figure(fig, 'kmeans_elbow_plot', 'advanced')
        
        # Perform clustering with optimal k
        optimal_k = 3  # This should be determined from elbow plot
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        return {
            'clusters': clusters,
            'inertias': inertias,
            'cluster_centers': kmeans.cluster_centers_,
            'optimal_k': optimal_k
        }

    def analyze_correlations(self) -> Dict[str, Any]:
        """Perform advanced correlation analysis."""
        # Select numeric columns
        numeric_cols = self.df_clean.select_dtypes(include=['float64', 'int64']).columns
        
        # Calculate correlations
        pearson_corr = self.df_clean[numeric_cols].corr(method='pearson')
        spearman_corr = self.df_clean[numeric_cols].corr(method='spearman')
        
        # Create correlation heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        sns.heatmap(pearson_corr, ax=ax1, annot=True, fmt='.2f', cmap='coolwarm')
        ax1.set_title('Pearson Correlation')
        
        sns.heatmap(spearman_corr, ax=ax2, annot=True, fmt='.2f', cmap='coolwarm')
        ax2.set_title('Spearman Correlation')
        
        utils.save_and_close_figure(fig, 'correlation_heatmaps', 'advanced')
        
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr
        }

    def calculate_effect_sizes(self) -> Dict[str, Any]:
        """Calculate various effect sizes for relationships."""
        results = {}
        
        # Cohen's d for gender differences in satisfaction
        gender_groups = [
            group for _, group in self.df_clean.groupby('gender')['satisfaction_score']
        ]
        if len(gender_groups) == 2:
            n1, n2 = len(gender_groups[0]), len(gender_groups[1])
            var1, var2 = np.var(gender_groups[0], ddof=1), np.var(gender_groups[1], ddof=1)
            
            # Pooled standard deviation
            pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            # Cohen's d
            cohens_d = (np.mean(gender_groups[0]) - np.mean(gender_groups[1])) / pooled_sd
            
            results['cohens_d'] = {
                'value': cohens_d,
                'interpretation': self.interpret_cohens_d(cohens_d)
            }
        
        # Eta-squared for income groups and satisfaction
        income_groups = self.df_clean.groupby(
            'which_income_range_do_you_your_family_belong_to_monthly')['satisfaction_score']
        ss_between = sum(len(group) * (group.mean() - 
                        self.df_clean['satisfaction_score'].mean())**2 
                        for _, group in income_groups)
        ss_total = sum((self.df_clean['satisfaction_score'] - 
                       self.df_clean['satisfaction_score'].mean())**2)
        eta_squared = ss_between / ss_total
        
        results['eta_squared'] = {
            'value': eta_squared,
            'interpretation': self.interpret_eta_squared(eta_squared)
        }
        
        return results

    def perform_bootstrap_analysis(self, n_iterations: int = 1000) -> Dict[str, Any]:
        """Perform bootstrap analysis for key metrics."""
        results = {}
        
        # Bootstrap mean satisfaction score
        satisfaction_boots = []
        original_mean = self.df_clean['satisfaction_score'].mean()
        
        for _ in range(n_iterations):
            boot_sample = self.df_clean['satisfaction_score'].sample(
                n=len(self.df_clean), replace=True)
            satisfaction_boots.append(boot_sample.mean())
        
        ci_lower = np.percentile(satisfaction_boots, 2.5)
        ci_upper = np.percentile(satisfaction_boots, 97.5)
        
        results['satisfaction_bootstrap'] = {
            'original_mean': original_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'standard_error': np.std(satisfaction_boots)
        }
        
        # Plot bootstrap distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(satisfaction_boots, kde=True, ax=ax)
        ax.axvline(original_mean, color='r', linestyle='--', label='Original Mean')
        ax.axvline(ci_lower, color='g', linestyle=':', label='95% CI')
        ax.axvline(ci_upper, color='g', linestyle=':')
        ax.set_title('Bootstrap Distribution of Mean Satisfaction Score')
        ax.legend()
        
        utils.save_and_close_figure(fig, 'bootstrap_distribution', 'advanced')
        
        return results

    def calculate_power_analysis(self) -> Dict[str, Any]:
        """Perform statistical power analysis."""
        results = {}
        
        # Power analysis for gender difference in satisfaction
        gender_groups = [
            group for _, group in self.df_clean.groupby('gender')['satisfaction_score']
        ]
        
        if len(gender_groups) == 2:
            effect_size = np.abs(np.mean(gender_groups[0]) - np.mean(gender_groups[1])) / \
                         np.sqrt((np.var(gender_groups[0]) + np.var(gender_groups[1])) / 2)
            
            n1, n2 = len(gender_groups[0]), len(gender_groups[1])
            df = n1 + n2 - 2
            
            # Non-centrality parameter
            ncp = effect_size * np.sqrt(n1 * n2 / (n1 + n2))
            
            # Critical value
            cv = t.ppf(0.975, df)
            
            # Power calculation
            power = 1 - t.cdf(cv, df, ncp)
            
            results['gender_difference_power'] = {
                'effect_size': effect_size,
                'power': power,
                'sample_size': n1 + n2,
                'alpha': 0.05
            }
        
        return results

    def perform_factor_analysis(self) -> Dict[str, Any]:
        """Perform exploratory factor analysis on numeric variables."""
        # Select numeric columns
        numeric_cols = self.df_clean.select_dtypes(include=['float64', 'int64']).columns
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df_clean[numeric_cols])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(scaled_data.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvects = np.linalg.eig(corr_matrix)
        
        # Sort eigenvalues and eigenvectors
        sorted_idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[sorted_idx]
        eigenvects = eigenvects[:, sorted_idx]
        
        # Calculate proportion of variance explained
        total_var = eigenvals.sum()
        prop_var = eigenvals / total_var
        cum_var = np.cumsum(prop_var)
        
        # Plot scree plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(eigenvals) + 1), eigenvals, 'bo-')
        ax.set_xlabel('Factor')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Scree Plot')
        
        utils.save_and_close_figure(fig, 'factor_analysis_scree', 'advanced')
        
        return {
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvects,
            'prop_variance': prop_var,
            'cum_variance': cum_var,
            'factor_loadings': pd.DataFrame(
                eigenvects,
                columns=[f'Factor_{i+1}' for i in range(len(eigenvects))],
                index=numeric_cols
            )
        }

    def run_advanced_analysis(self, show_plots: bool = False) -> Dict[str, Any]:
        """Main entry point for advanced analysis."""
        print("Starting advanced statistical analysis...")
        
        # Run analyses
        results = {
            'outliers': self.detect_outliers(
                self.df_clean.select_dtypes(include=['float64', 'int64']).columns
            ),
            'effect_sizes': self.calculate_effect_sizes(),
            'bootstrap': self.perform_bootstrap_analysis(),
            'power_analysis': self.calculate_power_analysis(),
            'factor_analysis': self.perform_factor_analysis(),
            'pca': self.run_pca_analysis(
                self.df_clean.select_dtypes(include=['float64', 'int64']).columns
            ),
            'clustering': self.perform_cluster_analysis(
                self.df_clean.select_dtypes(include=['float64', 'int64']).columns
            ),
            'correlations': self.analyze_correlations()
        }
        
        # Print summary
        print("\nAdvanced Analysis Summary")
        print("=" * 50)
        
        # Effect sizes
        if 'effect_sizes' in results:
            print("\nEffect Sizes:")
            if 'cohens_d' in results['effect_sizes']:
                d = results['effect_sizes']['cohens_d']['value']
                print(f"Cohen's d: {d:.3f} "
                      f"({results['effect_sizes']['cohens_d']['interpretation']})")
            if 'eta_squared' in results['effect_sizes']:
                eta = results['effect_sizes']['eta_squared']['value']
                print(f"Eta-squared: {eta:.3f} "
                      f"({results['effect_sizes']['eta_squared']['interpretation']})")
        
        # Bootstrap results
        if 'bootstrap' in results:
            boot = results['bootstrap']['satisfaction_bootstrap']
            print("\nBootstrap Analysis:")
            print(f"Mean Satisfaction: {boot['original_mean']:.2f}")
            print(f"95% CI: [{boot['ci_lower']:.2f}, {boot['ci_upper']:.2f}]")
        
        # Power analysis
        if 'power_analysis' in results:
            power = results['power_analysis'].get('gender_difference_power', {})
            if power:
                print("\nPower Analysis:")
                print(f"Statistical Power: {power['power']:.3f}")
                print(f"Effect Size: {power['effect_size']:.3f}")
        
        # Save results
        utils.save_analysis_results(results, 'advanced_analysis')
        
        print("\nAdvanced statistical analysis completed!")
        return results

    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        if abs(d) < 0.2:
            return "negligible effect"
        elif abs(d) < 0.5:
            return "small effect"
        elif abs(d) < 0.8:
            return "medium effect"
        else:
            return "large effect"

    @staticmethod
    def interpret_eta_squared(eta: float) -> str:
        """Interpret eta-squared effect size."""
        if eta < 0.01:
            return "negligible effect"
        elif eta < 0.06:
            return "small effect"
        elif eta < 0.14:
            return "medium effect"
        else:
            return "large effect"