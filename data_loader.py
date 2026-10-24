"""
OLYMPIQ Data Loader
===================

Loads and cleans authentic historical Olympic data.
Handles missing values, duplicates, country name changes, and data quality issues.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

class OlympicDataLoader:
    """
    Handles loading and cleaning of Olympic historical data.
    Uses authentic data sources only - no synthetic data generation.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file. If None, searches common locations.
        """
        self.data_path = self._find_data_path(data_path)
        self.raw_data = None
        self.cleaned_data = None
        self.country_medals = None
        
    def _find_data_path(self, data_path: Optional[str]) -> Optional[str]:
        """Find the data file in common locations."""
        if data_path and os.path.exists(data_path):
            return data_path
        
        # Check environment variable
        env_path = os.environ.get('OLYMPIQ_DATA_PATH')
        if env_path and os.path.exists(env_path):
            return env_path
        
        # Check common locations
        possible_paths = [
            'data/olympics.csv',
            'olympics.csv',
            '../data/olympics.csv',
            'data/athlete_events.csv',
            'athlete_events.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the raw Olympic data.
        
        Returns:
            DataFrame with raw Olympic data
            
        Raises:
            FileNotFoundError: If data file cannot be found
        """
        if self.raw_data is not None:
            return self.raw_data
        
        if self.data_path is None or not os.path.exists(self.data_path):
            raise FileNotFoundError(
                "Olympic data file not found. Please download the dataset:\n"
                "1. From TidyTuesday: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-07-27/olympics.csv\n"
                "2. Place it as 'data/olympics.csv' in the project directory\n"
                "3. Or set OLYMPIQ_DATA_PATH environment variable"
            )
        
        try:
            print(f"Loading data from: {self.data_path}")
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.raw_data)} rows")
            return self.raw_data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the Olympic data.
        Handles missing values, duplicates, and standardizes formats.
        
        Returns:
            Cleaned DataFrame
        """
        if self.cleaned_data is not None:
            return self.cleaned_data
        
        df = self.load_data().copy()
        
        print("Cleaning data...")
        
        # Remove exact duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"  Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        # Medal can be NaN (no medal) - this is valid
        # Age, height, weight may be missing - we'll keep them but note limitations
        
        # Standardize column names and formats
        df['noc'] = df['noc'].str.upper().str.strip()
        df['team'] = df['team'].str.strip()
        df['sport'] = df['sport'].str.strip()
        df['event'] = df['event'].str.strip()
        df['city'] = df['city'].str.strip()
        df['season'] = df['season'].str.strip()
        
        # Standardize medal categories
        df['medal'] = df['medal'].astype(str).str.strip()
        df['medal'] = df['medal'].replace('nan', np.nan)
        # Convert to categorical with consistent order
        medal_order = ['Gold', 'Silver', 'Bronze']
        df['medal_cat'] = pd.Categorical(df['medal'], categories=medal_order, ordered=True)
        
        # Handle age - convert to numeric, keep NaN for missing
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        
        # Create a flag for medal winners
        df['has_medal'] = df['medal'].notna()
        
        # Create medal point system for analysis (optional)
        medal_points = {'Gold': 3, 'Silver': 2, 'Bronze': 1}
        df['medal_points'] = df['medal'].map(medal_points).fillna(0).astype(float)
        
        self.cleaned_data = df
        print(f"  Cleaned data shape: {df.shape}")
        return self.cleaned_data
    
    def get_country_medal_summary(self, season: Optional[str] = None, 
                                   year_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        """
        Get country-wise medal summary.
        
        Args:
            season: Filter by 'Summer' or 'Winter', None for both
            year_range: Tuple of (start_year, end_year)
            
        Returns:
            DataFrame with country medal counts
        """
        df = self.clean_data()
        
        # Apply filters
        if season:
            df = df[df['season'] == season]
        if year_range:
            df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        
        # Every NOC that participated in an edition, including countries that won
        # no medal. Keeping zero-medal countries is essential: they are the vast
        # majority of Olympic participants and leaving them out would make any
        # accuracy metric (MAE) look much better than it really is.
        participation = (
            df[['noc', 'year', 'season']].drop_duplicates().reset_index(drop=True)
        )

        # Count medals by country, year, event, medal to avoid double counting
        # team medals (team events have one row per athlete).
        medal_events = df[df['has_medal']].copy()
        unique_medals = medal_events.drop_duplicates(
            subset=['noc', 'year', 'season', 'event', 'medal']
        )

        counts = (
            unique_medals.groupby(['noc', 'year', 'season', 'medal'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=['Gold', 'Silver', 'Bronze'], fill_value=0)
            .rename(columns={'Gold': 'gold', 'Silver': 'silver', 'Bronze': 'bronze'})
            .reset_index()
        )

        summary = participation.merge(
            counts, on=['noc', 'year', 'season'], how='left'
        )
        for col in ['gold', 'silver', 'bronze']:
            summary[col] = pd.to_numeric(summary[col], errors='coerce').fillna(0).astype(int)
        summary['total'] = summary['gold'] + summary['silver'] + summary['bronze']

        return summary.sort_values(['year', 'noc']).reset_index(drop=True)
    
    def get_sport_summary(self, season: Optional[str] = None) -> pd.DataFrame:
        """
        Get sport-wise medal summary.
        
        Args:
            season: Filter by season
            
        Returns:
            DataFrame with sport medal counts
        """
        df = self.clean_data()
        if season:
            df = df[df['season'] == season]

        medal_events = df[df['has_medal']].copy()
        # One medal award per edition/event/colour/NOC.  Including the NOC keeps
        # both bronze medals awarded in events such as boxing, while still
        # counting a team medal only once instead of once per athlete.
        unique_awards = medal_events.drop_duplicates(
            subset=['year', 'season', 'sport', 'event', 'medal', 'noc']
        )
        summary = (
            unique_awards.groupby(['sport', 'medal'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=['Gold', 'Silver', 'Bronze'], fill_value=0)
            .rename(columns={'Gold': 'gold', 'Silver': 'silver', 'Bronze': 'bronze'})
        )
        summary['total'] = summary['gold'] + summary['silver'] + summary['bronze']
        return summary.reset_index().sort_values('total', ascending=False)
    
    def get_data_info(self) -> dict:
        """Get information about the dataset."""
        df = self.clean_data()
        return {
            'total_rows': len(df),
            'unique_athletes': df['name'].nunique(),
            'unique_countries': df['noc'].nunique(),
            'unique_sports': df['sport'].nunique(),
            'unique_events': df['event'].nunique(),
            'years': (int(df['year'].min()), int(df['year'].max())),
            'seasons': df['season'].unique().tolist(),
            'medal_counts': df['medal'].value_counts().to_dict()
        }

    def data_quality(self) -> pd.DataFrame:
        """Return a small, displayable data-quality summary."""
        df = self.clean_data()
        seasons = sorted(df['season'].dropna().unique().tolist())
        rows = [
            {'check': 'Rows after de-duplication', 'value': f"{len(df):,}",
             'status': 'ok', 'comment': 'Exact duplicate athlete rows removed.'},
            {'check': 'NOCs', 'value': str(df['noc'].nunique()),
             'status': 'ok', 'comment': 'Country codes kept as supplied; political-era codes are not merged.'},
            {'check': 'Editions', 'value': f"{int(df['year'].min())}-{int(df['year'].max())}",
             'status': 'ok', 'comment': 'Historical editions present in the source file.'},
            {'check': 'Seasons', 'value': ', '.join(seasons),
             'status': 'ok', 'comment': 'Summer and Winter are analysed separately.'},
            {'check': 'Medal rows', 'value': f"{int(df['has_medal'].sum()):,}",
             'status': 'ok', 'comment': 'Rows describing a gold, silver or bronze medal.'},
            {'check': 'Zero-medal NOC-editions',
             'value': f"{int((df.groupby(['noc','year','season'])['has_medal'].sum() == 0).sum()):,}",
             'status': 'ok', 'comment': 'Retained so accuracy metrics are not flattered by dropping the long tail.'},
            {'check': 'Missing age values', 'value': f"{int(df['age'].isna().sum()):,}",
             'status': 'info', 'comment': 'Age is not used by the forecasting model.'},
            {'check': 'Data cutoff', 'value': str(int(df['year'].max())),
             'status': 'warning', 'comment': 'No observations after 2016; later forecasts are extrapolations.'},
        ]
        return pd.DataFrame(rows)

    def country_code_change_note(self) -> pd.DataFrame:
        """Document well-known historical NOC changes without silently merging them."""
        rows = [
            {'historical_noc': 'URS', 'modern_equivalent': 'RUS / other ex-USSR states',
             'treatment': 'Kept separate; successor states have their own short histories.'},
            {'historical_noc': 'GDR', 'modern_equivalent': 'GER',
             'treatment': 'Kept separate; East German medals are not added to Germany.'},
            {'historical_noc': 'TCH', 'modern_equivalent': 'CZE / SVK',
             'treatment': 'Kept separate.'},
            {'historical_noc': 'YUG', 'modern_equivalent': 'SRB / CRO / SLO / BIH and others',
             'treatment': 'Kept separate.'},
            {'historical_noc': 'EUN', 'modern_equivalent': 'Unified team (1992)',
             'treatment': 'Kept separate; not reassigned to individual states.'},
            {'historical_noc': 'FRG', 'modern_equivalent': 'GER',
             'treatment': 'Kept separate; reunification is a structural break.'},
        ]
        return pd.DataFrame(rows)

# Convenience functions
def load_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load the raw Olympic athlete/event data.

    This name is what ``prediction.load_history()`` looks for, so the
    predictive module can discover the dataset automatically.
    """
    return OlympicDataLoader(data_path).load_data()

def load_olympic_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load and clean Olympic data."""
    loader = OlympicDataLoader(data_path)
    return loader.clean_data()

def get_country_medals(data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Get country medal summary."""
    loader = OlympicDataLoader(data_path)
    return loader.get_country_medal_summary(**kwargs)

if __name__ == "__main__":
    # Test the loader
    try:
        loader = OlympicDataLoader()
        df = loader.clean_data()
        print("\nDataset Info:")
        info = loader.get_data_info()
        for k, v in info.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease download the dataset as instructed in README.md")
