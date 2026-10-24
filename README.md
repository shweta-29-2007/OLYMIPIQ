# OLYMPIQ — Olympic history and forecasting

OLYMPIQ is a Streamlit dashboard for exploring authentic historical Olympic athlete and event results, comparing countries and sports, and reviewing cautious country-level medal estimates.

## Run it on Windows

Install Python 3.12 (and its `py` launcher) if it is not already installed. Open this repository folder in VS Code. In **Terminal → New Terminal**, run:

```powershell
cd C:\Users\HP\Desktop\hackathon\OLYMIPIQ
```

Start the one-command launcher:

```powershell
.\run_olympiq.bat
```

The script installs Python 3.12 with the Python Launcher if needed, creates a separate `.venv-olympiq312` environment inside this repository so it leaves any existing environment alone, installs the four direct app dependencies, and starts Streamlit. Version ranges let pip choose compatible releases instead of pinning unnecessary transitive packages. Streamlit prints a local address, usually `http://localhost:8501`, and opens it in your browser. To run both the Streamlit analytics dashboard and React landing frontend together, run `run_olympiq_full.bat`. To run Streamlit manually after setup: `.venv-olympiq312\Scripts\python.exe -m streamlit run app.py`.

## Historical dataset

The repository's `data/olympics_raw.csv` is the real athlete-event table from the Kaggle **120 years of Olympic history: athletes and results** dataset, reproduced and documented by TidyTuesday. It has 271,116 source rows and covers Athens 1896 through Rio 2016. The source data was scraped from Sports Reference in 2018; it is not an official IOC live feed and does not include later Games.

Source and field descriptions:

- [TidyTuesday source notes and data dictionary](https://github.com/rfordatascience/tidytuesday/blob/main/data/2021/2021-07-27/readme.md)
- [Original Kaggle dataset](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)

The source notes identify Kaggle and Sports Reference provenance but do not state a clear redistribution licence for the original data. Keep attribution with the dataset and check the current Kaggle/source terms before republishing or redistributing it. The CSV is ignored by Git so this repository does not upload the historical file by default.

If the CSV is missing, download `athlete_events.csv` from the Kaggle page, place it at `data/olympics_raw.csv`, or set `OLYMPIQ_DATA_PATH` to its full path. The loader does not create substitute or synthetic records.

### Source columns

`ID`, `Name`, `Sex`, `Age`, `Height`, `Weight`, `Team`, `NOC`, `Games`, `Year`, `Season`, `City`, `Sport`, `Event`, and `Medal`. A blank medal means the athlete did not win a medal. Missing athlete height, weight, or age values remain missing; these variables are not used in the forecast.

### Counting and historical limitations

- The source contains one row per athlete-event entry, so a team award can appear once for each teammate. Medal summaries de-duplicate awards by country, edition, event, and medal colour rather than treating every athlete row as a separate medal.
- Every country edition with participation remains in the panel, including zero-medal results.
- Historical NOC codes (for example URS, GDR, TCH, and YUG) are not silently reassigned to modern successor countries.
- Sports, event programmes, participation, and medal opportunities change across eras. Historical totals are not perfectly comparable.
- Source results stop at 2016. The dashboard does not claim live or real-time medal data.

## Dashboard sections

- **Welcome and Overview** — project entry point, source coverage, and an India historical-result spotlight.
- **Historical Trends, Country Analysis, and Sport Analysis** — interactive season, edition, nation, and sport comparisons.
- **Performance Trajectory, Breakthrough Watch, and Sport Opportunity** — descriptive country insights with calculation explanations.
- **Predictive Analytics and Prediction vs Actual** — a transparent hurdle model, chronological holdout folds, baseline metrics, and uncertainty notes.
- **Dataset & Methodology** — data definitions, quality notes, the model card, and metric descriptions.

## Forecast method

`prediction.py` builds a country-edition panel including zero-medal participants and validates with earlier Olympic editions before each test edition. It compares predictions with persistence and recent-mean baselines. The hurdle model estimates the chance of any medal and the expected count conditional on medalling. Results are estimates, not exact medal counts; host and historical trends are associations and do not prove causation.

## Useful checks

```powershell
python -m py_compile app.py data_loader.py analysis.py prediction.py
python -m unittest -v test_prediction
```

The authentic-data integration test uses the dataset discovered by `data_loader.py`; the rest of the test suite uses small fixtures to check data handling and chronological logic.
