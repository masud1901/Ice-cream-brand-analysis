import pandas as pd
import re
from pathlib import Path


class IceCreamDataCleaner:
    """Class to handle ice cream survey data cleaning pipeline."""
    
    def __init__(self):
        """Initialize data cleaner with mapping dictionaries."""
        self.satisfaction_map = {
            'Very Satisfied': 5,
            'Satisfied': 4, 
            'Neutral': 3,
            'Dissatisfied': 2,
            'Very Dissatisfied': 1
        }
        
        self.likelihood_map = {
            'Highly Likely': 5,
            'Likely': 4,
            'Neutral': 3,
            'Unlikely': 2,
            'Very Unlikely': 1
        }
        
        self.age_map = {
            'Below 18': 17,
            '18-24': 21,
            '25-34': 29.5,
            '25-35': 30,
            '35-44': 39.5,
            '45-55': 50
        }
        
        self.income_map = {
            'Below 25,000': 12.5,
            '25,000 - 50,000': 37.5,
            '50,001 - 100,000': 75.0,
            '100,001 - 150,000': 125.0,
            '150,001- 250,000': 200.0,
            'Above 250,000': 300.0
        }
        
        self.multiple_choice_mapping = {
            'which_type_of_ice_cream_do_you_prefer': {
                'prefix': 'prefers_type',
                'choices': [
                    'Regular Cups (Chocolate, Vanilla, Strawberry, Mango)',
                    'Premium Cups (Exclusive Flavors)',
                    'Cone',
                    'Lolly', 
                    'Ice Cream Sticks (Chocbar, Crunchy Bar etc.)',
                    'Box or tubs',
                    'Scoops (from Ice Cream Parlor)'
                ]
            },
            'which_factors_influence_your_decision_when_purchasing_ice_cream': {
                'prefix': 'factor',
                'choices': [
                    'Taste',
                    'Price', 
                    'Availability',
                    'Brand reputation',
                    'Flavors',
                    'Packaging',
                    'Health/nutritional value'
                ]
            },
            'which_of_the_following_ice_cream_flavor_would_you_try': {
                'prefix': 'willing_to_try',
                'choices': [
                    'Pistachio',
                    'Orange',
                    'Carrot', 
                    'Jalpai',
                    'None of these'
                ]
            },
            'how_do_you_usually_hear_about_new_ice_cream_brands_or_flavors': {
                'prefix': 'marketing_channel',
                'choices': [
                    'Social media',
                    'TV ads',
                    'Word of mouth',
                    'In-store promotions / Shopkeeper Recommendation',
                    'Billboards',
                    'Online ads'
                ]
            },
            'what_would_make_you_try_a_new_ice_cream_brand': {
                'prefix': 'motivation',
                'choices': [
                    'Price discounts',
                    'New flavors',
                    'Availability',
                    'Attractive packaging',
                    'Brand reputation',
                    'Word of mouth',
                    'Advertisement'
                ]
            }
        }

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names."""
        df.columns = (df.columns
                     .str.strip()
                     .str.lower()
                     .str.replace('?', '', regex=False)
                     .str.replace('/', '_', regex=False)
                     .str.replace('[', '', regex=False)
                     .str.replace(']', '', regex=False)
                     .str.replace('(', '', regex=False)
                     .str.replace(')', '', regex=False)
                     .str.replace('  ', ' ', regex=False)
                     .str.replace(' ', '_', regex=False))
        return df

    def _handle_multiple_choice_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process multiple-choice columns into binary indicators."""
        for column, config in self.multiple_choice_mapping.items():
            if column in df.columns:
                prefix = config['prefix']
                choices = config['choices']
                
                for choice in choices:
                    col_name = (f"{prefix}_"
                              f"{choice.lower().replace(' ', '_').replace('/', '_')}")
                    if choice == 'None of these':
                        df[col_name] = (df[column]
                                      .fillna('None of these')
                                      .str.contains('None of these', 
                                                  case=False, 
                                                  regex=False)
                                      .astype(int))
                    else:
                        df[col_name] = (df[column]
                                      .fillna('')
                                      .str.contains(re.escape(choice), 
                                                  case=False, 
                                                  regex=True)
                                      .astype(int))
                
                # Handle 'Other' responses
                other_responses = df[column].fillna('').str.split(',').explode()
                known_choices = set(choice.lower() for choice in choices)
                other_values = (other_responses[
                    ~other_responses.str.lower().str.strip().isin(
                        [choice.lower() for choice in choices]
                    )].dropna().unique())
                
                if len(other_values) > 0:
                    col_name = f"{prefix}_other"
                    df[col_name] = df[column].fillna('').apply(
                        lambda x: 1 if any(
                            val.lower().strip() not in known_choices 
                            for val in str(x).split(',')
                        ) else 0
                    )
        
        # Clean up column names
        for col in df.columns:
            if any(col.startswith(prefix['prefix']) 
                  for prefix in self.multiple_choice_mapping.values()):
                new_name = col
                new_name = new_name.replace(
                    'regular_cups_chocolate_vanilla_strawberry_mango', 
                    'regular_cups')
                new_name = new_name.replace(
                    'premium_cups_exclusive_flavors', 
                    'premium_cups')
                new_name = new_name.replace(
                    'ice_cream_sticks_chocbar_crunchy_bar_etc', 
                    'ice_cream_sticks')
                new_name = new_name.replace(
                    'scoops_from_ice_cream_parlor', 
                    'scoops')
                new_name = new_name.replace(
                    'in_store_promotions_shopkeeper_recommendation', 
                    'in_store_promotions')
                new_name = new_name.replace(
                    'health_nutritional_value', 
                    'health_value')
                if new_name != col:
                    df = df.rename(columns={col: new_name})
        
        return df

    def _create_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create numeric features from categorical data."""
        # Apply mappings
        satisfaction_col = ('how_satisfied_are_you_with_the_range_of_flavors'
                          '_currently_available_in_the_bangladeshi_market')
        df['satisfaction_score'] = df[satisfaction_col].map(self.satisfaction_map)
        
        likelihood_col = 'how_likely_are_you_to_try_a_new_ice_cream_brand'
        df['likelihood_score'] = df[likelihood_col].map(self.likelihood_map)
        
        df['age_numeric'] = df['age'].map(self.age_map)
        
        income_col = 'which_income_range_do_you_your_family_belong_to_monthly'
        df['income_numeric'] = df[income_col].map(self.income_map)
        
        # Count features
        type_cols = [col for col in df.columns if col.startswith('prefers_type_')]
        df['num_types_preferred'] = df[type_cols].sum(axis=1)
        
        factor_cols = [col for col in df.columns if col.startswith('factor_')]
        df['num_purchase_factors'] = df[factor_cols].sum(axis=1)
        
        channel_cols = [col for col in df.columns 
                       if col.startswith('marketing_channel_')]
        df['num_marketing_channels'] = df[channel_cols].sum(axis=1)
        
        return df

    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns by removing special characters and title casing."""
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            df[col] = (df[col]
                      .astype(str)
                      .str.replace(r'[^\x00-\x7F]+', '', regex=True)
                      .str.title()
                      .str.strip())
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by filling them with 'Not Specified'."""
        return df.fillna('Not Specified')

    def _filter_valid_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter valid responses by ensuring 'do_you_consume_ice_cream' is 'Yes'."""
        return df[df['do_you_consume_ice_cream'] == 'Yes'].reset_index(drop=True)

    def _save_cleaned_data(self, df: pd.DataFrame):
        """Save cleaned data to processed directory."""
        processed_dir = Path('data/processed')
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = processed_dir / 'cleaned_ice_cream_data.csv'
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")

    def run_data_cleaning(self, df: pd.DataFrame, 
                         save_cleaned: bool = True) -> pd.DataFrame:
        """Main entry point for data cleaning pipeline."""
        print("Starting data cleaning process...")
        
        df_clean = df.copy()
        
        # Run cleaning steps in logical order
        df_clean = self._clean_column_names(df_clean)
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        df_clean = self._handle_multiple_choice_columns(df_clean)
        df_clean = self._create_numeric_features(df_clean)
        df_clean = self._clean_text_columns(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        df_clean = self._filter_valid_responses(df_clean)
        
        if save_cleaned:
            self._save_cleaned_data(df_clean)
        
        print("Data cleaning completed successfully!")
        return df_clean