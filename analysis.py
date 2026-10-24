"""
OLYMPIQ Analysis Module
=======================

Provides analysis functions for Olympic data including trends, insights,
and the unique hackathon features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from data_loader import OlympicDataLoader

class OlympicAnalyzer:
    """
    Analyzes Olympic data to provide insights and unique features.
    """
    
    def __init__(self, data_loader: Optional[OlympicDataLoader] = None):
        self.loader = data_loader or OlympicDataLoader()
        self.df = self.loader.clean_data()
        self.country_medals = self.loader.get_country_medal_summary()
        # Reused by several pages; computing these once keeps the UI responsive.
        self.country_names = (
            self.df.dropna(subset=['team'])
            .groupby('noc')['team']
            .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0])
            .to_dict()
        )
    
    def get_performance_trajectory(self, noc: str, season: Optional[str] = None) -> pd.DataFrame:
        """
        Olympic Performance Trajectory - shows how a country's medal performance
        changes across editions.
        
        Args:
            noc: Country code
            season: Filter by season
            
        Returns:
            DataFrame with trajectory data
        """
        data = self.country_medals.copy()
        data = data[data['noc'] == noc.upper()]
        if season:
            data = data[data['season'] == season]
        
        # Calculate rolling averages and trends
        data = data.sort_values('year')
        data['gold_rolling'] = data['gold'].rolling(window=3, min_periods=1).mean()
        data['total_rolling'] = data['total'].rolling(window=3, min_periods=1).mean()
        data['gold_change'] = data['gold'].diff()
        data['total_change'] = data['total'].diff()
        
        # Get country name
        data['country_name'] = self.country_names.get(noc.upper(), noc.upper())
        return data
    
    def get_breakthrough_watch(self, min_editions: int = 3, improvement_threshold: float = 0.2) -> pd.DataFrame:
        """
        Breakthrough Watch - identifies countries with improving historical trends,
        without guaranteeing future medals.
        
        Args:
            min_editions: Minimum number of editions to consider
            improvement_threshold: Minimum improvement ratio
            
        Returns:
            DataFrame of countries showing improvement
        """
        results = []
        
        for noc in self.country_medals['noc'].unique():
            for season in ['Summer', 'Winter']:
                data = self.country_medals[
                    (self.country_medals['noc'] == noc) & 
                    (self.country_medals['season'] == season)
                ].sort_values('year')
                
                if len(data) < min_editions:
                    continue
                
                # Calculate trend using linear regression on recent editions
                recent = data.tail(min(5, len(data)))
                if len(recent) < 2:
                    continue
                
                years = recent['year'].values
                totals = recent['total'].values
                
                # Simple linear trend
                if len(years) > 1:
                    z = np.polyfit(years, totals, 1)
                    trend_slope = z[0]
                    
                    # Calculate improvement ratio
                    first_avg = recent['total'].iloc[:max(1, len(recent)//2)].mean()
                    last_avg = recent['total'].iloc[max(1, len(recent)//2):].mean()
                    
                    if first_avg > 0:
                        improvement_ratio = (last_avg - first_avg) / first_avg
                    elif last_avg > 0:
                        improvement_ratio = 1.0  # From zero to medals
                    else:
                        improvement_ratio = 0
                    
                    if improvement_ratio > improvement_threshold and trend_slope > 0:
                        country_name = self.country_names.get(noc, noc)
                        
                        results.append({
                            'noc': noc,
                            'country': country_name,
                            'season': season,
                            'editions': len(data),
                            'first_medals': int(data['total'].iloc[0]),
                            'last_medals': int(data['total'].iloc[-1]),
                            'recent_avg': round(last_avg, 2),
                            'improvement_ratio': round(improvement_ratio, 3),
                            'trend_slope': round(trend_slope, 3),
                            'trend': 'Improving',
                            'note': 'Historical trend only - no guarantee of future medals'
                        })
        
        if results:
            return pd.DataFrame(results).sort_values('improvement_ratio', ascending=False)
        else:
            return pd.DataFrame(columns=['noc', 'country', 'season', 'editions', 
                                        'first_medals', 'last_medals', 'recent_avg',
                                        'improvement_ratio', 'trend_slope', 'trend', 'note'])
    
    def get_sport_opportunity_explorer(self, noc: str, top_n: int = 10) -> pd.DataFrame:
        """
        Sport Opportunity Explorer - highlights sports in which a country has
        historically performed relatively well or could investigate further.
        
        Args:
            noc: Country code
            top_n: Number of sports to return
            
        Returns:
            DataFrame of sport opportunities
        """
        noc = noc.upper()
        df_country = self.df[(self.df['noc'] == noc) & (self.df['has_medal'])]
        
        if df_country.empty:
            # Countries with no medals - show sports they participated in
            df_participated = self.df[self.df['noc'] == noc]
            if df_participated.empty:
                return pd.DataFrame()
            
            sport_participation = df_participated.groupby('sport').agg({
                'event': 'nunique',
                'name': 'nunique'
            }).rename(columns={'event': 'events_participated', 'name': 'athletes'})
            sport_participation['medals'] = 0
            sport_participation['medal_rate'] = 0
            sport_participation['opportunity_score'] = sport_participation['events_participated'] * 0.1
            sport_participation['category'] = 'Participation Only - No Medals Yet'
            sport_participation['note'] = 'Potential area for investigation'
            return sport_participation.sort_values('opportunity_score', ascending=False).head(top_n).reset_index()
        
        # Count one medal award per edition/event/colour (team medals appear once
        # per athlete in the raw file and must not be counted repeatedly).
        country_awards = df_country.drop_duplicates(
            subset=['year', 'season', 'event', 'medal']
        )
        sport_stats = country_awards.groupby('sport').agg({
            'medal': lambda x: (x == 'Gold').sum(),
            'event': 'nunique'
        }).rename(columns={'medal': 'gold', 'event': 'events_won_medals'})

        sport_stats['silver'] = country_awards.groupby('sport')['medal'].apply(lambda x: (x == 'Silver').sum())
        sport_stats['bronze'] = country_awards.groupby('sport')['medal'].apply(lambda x: (x == 'Bronze').sum())
        sport_stats['total_medals'] = sport_stats['gold'] + sport_stats['silver'] + sport_stats['bronze']

        # Get total participation (unique events entered across all editions).
        total_participation = self.df[self.df['noc'] == noc].groupby('sport')['event'].nunique()
        sport_stats['events_participated'] = total_participation
        sport_stats['medal_conversion'] = sport_stats['events_won_medals'] / sport_stats['events_participated']
        sport_stats['medal_conversion'] = sport_stats['medal_conversion'].fillna(0)
        
        # Calculate opportunity score (based on historical success)
        sport_stats['opportunity_score'] = (
            sport_stats['total_medals'] * 0.4 +
            sport_stats['gold'] * 0.3 +
            sport_stats['medal_conversion'] * 10 * 0.3
        )
        
        # Categorize
        def categorize(row):
            if row['gold'] >= 3:
                return 'Strong Historical Performance'
            elif row['total_medals'] >= 5:
                return 'Consistent Performer'
            elif row['total_medals'] >= 1:
                return 'Has Won Medals'
            else:
                return 'Other'
        
        sport_stats['category'] = sport_stats.apply(categorize, axis=1)
        sport_stats['note'] = 'Based on historical data only - past success not guarantee of future'
        
        return sport_stats.sort_values('opportunity_score', ascending=False).head(top_n).reset_index()
    
    def get_historical_trends(self, season: Optional[str] = None, 
                             year_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        """Get historical medal trends over time."""
        data = self.country_medals.copy()
        if season:
            data = data[data['season'] == season]
        if year_range:
            data = data[(data['year'] >= year_range[0]) & (data['year'] <= year_range[1])]
        
        trends = data.groupby('year').agg({
            'gold': 'sum',
            'silver': 'sum',
            'bronze': 'sum',
            'total': 'sum',
            'noc': 'nunique'
        }).rename(columns={'noc': 'countries_participating'}).reset_index()
        
        return trends
    
    def get_top_countries(self, season: Optional[str] = None, 
                         year: Optional[int] = None, top_n: int = 10) -> pd.DataFrame:
        """Get top countries by medal count."""
        data = self.country_medals.copy()
        if season:
            data = data[data['season'] == season]
        if year:
            data = data[data['year'] == year]
        else:
            # Overall all-time
            data = data.groupby('noc').agg({
                'gold': 'sum',
                'silver': 'sum',
                'bronze': 'sum',
                'total': 'sum'
            }).reset_index()
        
        # Add country names
        data = data.copy()
        data['country'] = data['noc'].map(self.country_names).fillna(data['noc'])
        
        return data.sort_values('total', ascending=False).head(top_n)
    
    def get_dataset_explanation(self) -> Dict:
        """Get explanation of dataset and its source."""
        if not hasattr(self, '_dataset_info'):
            self._dataset_info = self.loader.get_data_info()
        info = self._dataset_info
        return {
            'source': 'TidyTuesday Olympics Dataset (Kaggle: 120 years of Olympic history)',
            'url': 'https://github.com/rfordatascience/tidytuesday/tree/master/data/2021/2021-07-27',
            'description': 'Historical data on athletes and results from Athens 1896 to Rio 2016',
            'coverage': f"{info['years'][0]} - {info['years'][1]}",
            'seasons': info['seasons'],
            'stats': info,
            'limitations': [
                'Data ends at 2016 - no data for 2020, 2024, or future Games',
                'Country codes may have changed due to political events',
                'Some events/medals may be recorded differently across eras',
                'Team events can be complex to count (handled by deduplication)',
                'Missing data for some athlete attributes (age, height, weight)',
                'Winter and Summer Olympics have different sports/programs',
                'Historical comparisons across eras should be done with caution'
            ],
            'notes': 'All data is authentic historical data from public sources. No synthetic data was used.'
        }

if __name__ == "__main__":
    analyzer = OlympicAnalyzer()
    print("Olympic Analyzer initialized")
    print(analyzer.get_dataset_explanation()['source'])
