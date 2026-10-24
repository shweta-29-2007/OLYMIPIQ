# OLYMPIQ - Olympic Data Analytics and Predictive Visualization

OLYMPIQ is a professional, interactive Streamlit dashboard for exploring historical Olympic data and generating transparent, defensible medal forecasts.

## Features

- **Historical Trends**: Explore medal counts by year, season (Summer/Winter), country, sport, and medal type
- **Country & Sport Analysis**: Compare performances across countries and sports with interactive Plotly charts
- **Performance Trajectory**: Track how a country's medal performance evolves across Olympic editions
- **Breakthrough Watch**: Identify countries showing improving trends (without guaranteeing future medals)
- **Sport Opportunity Explorer**: Discover sports where a country has historically performed relatively well
- **Predictive Analytics**: Transparent medal forecasts using chronological validation (train on past, test on future)
- **Prediction vs Actual**: Backtesting view showing model performance against known results
- **Uncertainty & Transparency**: Clear explanations of methodology, limitations, and confidence

## Data Source

This app uses the [TidyTuesday Olympics dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2021/2021-07-27) (Kaggle "120 years of Olympic history: athletes and results"), which contains authentic historical Olympic data from Athens 1896 to Rio 2016.

**Note**: This dataset ends at 2016. Predictions for future Games (e.g., 2020, 2024, 2028) are extrapolations based on historical trends and must be interpreted as estimates, not guarantees.

## Installation & Setup

### Prerequisites
- Python 3.8+ (tested with Python 3.14)
- pip

### Steps

1. **Clone or navigate to the project directory**
```bash
cd "D:\sem 3\hackathon\OLYMPIQ"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Obtain the dataset**
The app looks for data in the following order:
- `data/olympics.csv` (recommended)
- `OLYMPIQ_DATA_PATH` environment variable
- `data_loader` module

If not present, download from:
https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-07-27/olympics.csv

Place it as `data/olympics.csv`.

4. **Run the application**
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## Project Structure

```
OLYMPIQ/
├── app.py              # Main Streamlit dashboard
├── data_loader.py      # Data loading and cleaning
├── analysis.py         # Analysis functions and insights
├── prediction.py       # Predictive modeling with validation
├── requirements.txt    # Dependencies
├── README.md          # This file
├── .gitignore         # Git ignore rules
└── data/              # Data directory (gitignored)
    └── olympics.csv   # Historical dataset
```

## Methodology Notes

- **No data leakage**: Predictions use chronological validation - models are trained only on past editions
- **Transparent modeling**: Simple, interpretable approaches (weighted recent performance, host effects where applicable)
- **Uncertainty acknowledged**: All forecasts are estimates with clear limitations
- **Zero-medal handling**: Countries with no medals are appropriately handled without fabricating results
- **Baseline comparison**: Model performance compared against simple historical baselines

## Limitations

- Data only goes to 2016; recent Games may not be fully represented
- Country codes/names may change over time (political changes)
- Some historical records may be incomplete
- Sport program changes affect cross-edition comparisons
- Predictions cannot account for unforeseen events (injuries, boycotts, etc.)
- Association does not imply causation

## License & Disclaimer

This project is for educational/hackathon purposes. Olympic trademarks are not used inappropriately. All data used is from publicly accessible sources.
