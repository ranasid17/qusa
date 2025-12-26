# qusa
Practice application 


## Structure 

```markdown
qusa/ 
│
├── README.md                           # Project overview and instructions
├── requirements.txt                    # Python dependencies 
├── setup.py                            # Package installation script
├── config.yaml                         # Configuration settings (paths, tickers, etc.)
├── .env                                # Environment variables and secrets
├── .gitignore                          # Files to ignore in git
│
├── data/                               # Storage for raw, processed data and models
│   ├── raw/                            # Original input CSV files
│   │   ├── {ticker1}.csv
│   │   └── {ticker2}.csv
│   │
│   ├── processed/                      # Processed data ready for analysis
│   │   ├── {ticker1}_processed.csv
│   │   └── {ticker2}_processed.csv 
│   │
│   └── models/                         # Saved ML models
│       └── cluster.pkl                 
│
├── qusa/                               # Main Python package
│   ├── __init__.py 
│   │
│   ├── features/                       # Feature engineering
│   │   ├── __init__.py 
│   │   ├── overnight.py                # Overnight change calculations
│   │   ├── technicals.py               # Technical indicator calculations
│   │   ├── calendar.py                 # Calendar/temporal features
│   │   └── pipeline.py                 # Complete feature pipeline
│   │
│   ├── analysis/                       # Signal analysis
│   │   ├── __init__.py
│   │   ├── signals.py                  # Signal identification and lift calculation
│   │   ├── clustering.py               # Unsupervised learning and clustering
│   │   └── statistics.py               # Statistical analysis of signals
│
├── scripts/                            # Standalone scripts for running analysis
|   ├── __init__.py
|   ├── get_most_recent_day.py          # Get most recent OHLCV data for given ticker 
│   ├── run_clustering.py               # Run clustering pipeline, visualize results
|   └── run_FE_pipeline.py              # Run feature engineering pipeline 
```