# qusa
Practice application 


## Structure 

qusa/ 
│
├── README.md
├── requirements.txt                    # Python dependencies 
├── setup.py                            # package installation 
├── config.yaml                         # configuration settings 
├── .env                                # secrets file 
├── .gitignore 
│
├── data/ 
│   ├── raw/                            # original input data 
│   │   ├── {ticker1}.csv
│   │   ├── ... 
│   │   └── {ticker2}.csv
│   │
│   ├── processed/                      # processed input data 
│   │   ├── {ticker1}_processed.csv
│   │   ├── {ticker2}_processed.csv 
│   └── models/                         # trained ML models 
│       └── cluster.pkl                 
│
├── qusa/ 
│   ├── __init__.py 
│   │
│   ├── features/                       # feature engineering 
│   │   ├── __init__.py 
│   │   ├── overnight.py                # overnight change calculations 
│   │   ├── technicals.py               # technical indicator calculations 
│   │   ├── calendar.py                 # calendar indicator calculations 
│   │   └── pipeline.py                 # feature engineering script 
│   │
│   ├── analysis/                       # signal analysis 
│   │   ├── __init__.py
│   │   ├── signals.py                  # identify signals, calculate lift 
│   │   ├── clustering.py               # unsupervised learning models 
│   │   └── statistics.py               # identify statistically significant signals 