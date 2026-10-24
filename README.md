# OLYMPIQ — Olympic Data Analytics & Predictive Visualization

OLYMPIQ is a beginner-friendly Streamlit dashboard for exploring historical Summer and Winter Olympic athlete-event records. It includes event-aware medal tables, country trajectories, a retrospective breakthrough screen, sport opportunity analysis, and a time-based forecast backtest.

> **Data vintage:** the included historical source ends at Rio 2016. The forecast page therefore presents long-range scenarios from old data, not current-cycle predictions.

## Included source data

The application uses Kaggle's [120 years of Olympic history: athletes and results](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results), a source dataset describing modern Games from 1896 through 2016. The public CSVs are downloaded into `data/` for local use but excluded from Git because the files are large and the Kaggle dataset's separate redistribution terms should be checked before republishing.

The project accepts either of these athlete-file names:

- `data/athlete_events.csv` or `data/olympics_raw.csv` — 271,116 source rows and 15 columns.
- `data/noc_regions.csv` — optional display-name lookup by historic NOC code.

They can be retrieved from the public [athlete CSV mirror](https://raw.githubusercontent.com/rgriff23/Olympic_history/master/data/athlete_events.csv) and [NOC mapping mirror](https://raw.githubusercontent.com/rgriff23/Olympic_history/master/data/noc_regions.csv). If your existing file is named `olympics_raw.csv`, the loader uses it automatically. If the athlete file is absent, download it from the Kaggle page (or linked mirror), create a `data` folder beside `app.py`, and place it there. The loader reports a clear error instead of substituting generated records.

## Run on Windows

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/windows/) and enable **Add python.exe to PATH** during setup.
2. Open PowerShell in this project folder.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the libraries:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

Streamlit prints a local address (usually `http://localhost:8501`) and opens the dashboard in your browser.

## What the app calculates

- Medal awards are counted once per NOC + sport + event + Olympic edition + medal category. This avoids counting every athlete in a team roster as an extra medal.
- Historical NOC codes remain distinct across political changes. The region mapping is for display; it does not merge successor states or rewrite history.
- Country trajectories include zero-medal editions when athletes from that NOC appeared in the source edition.
- Breakthrough Watch fits a linear slope to up to five recent edition totals and displays only positive slopes with at least four editions. It is a retrospective signal, not a guarantee.
- The forecast uses three previous edition medal totals, their average and trend in a scikit-learn Random Forest. Rolling chronological validation trains only on earlier editions. The baseline repeats the last edition. New NOCs without prior data are not forecast.
- If Random Forest has a higher rolling-validation MAE than the persistence baseline, the displayed scenario uses the simpler baseline instead.
- Forecast uncertainty uses historical absolute residuals as a rough, widening band. It is not a calibrated confidence interval. Host effects are omitted because the source does not give a reliable host NOC field.

Historical comparisons have important limits: country borders and NOCs change, the sport and event program changes, participation rules and delegation size shift, some editions were canceled, and records from the earliest Games are less complete. Medal counts describe results; they do not establish why outcomes changed.

## Files

- `app.py` — Streamlit interface and interactive Plotly views.
- `data_loader.py` — reusable validation, cleaning, and loading.
- `analysis.py` — medal counts and country/sport insight calculations.
- `prediction.py` — rolling time-based validation and future scenarios.
- `requirements.txt` — Python dependencies.
- `data/` — downloaded source CSVs (ignored by Git).
