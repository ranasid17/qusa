"""
Simple script to download stock data from Yahoo Finance
Works around authentication issues by using proper headers and cookies
"""
import pandas as pd
import requests
from datetime import datetime
import json

def download_yahoo_data(ticker, start_date='2021-01-01'):
    """Download historical stock data from Yahoo Finance"""
    
    print(f'Downloading {ticker} data from {start_date}...')
    
    # Step 1: Get cookies and crumb
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # Get crumb
    crumb_url = f'https://query2.finance.yahoo.com/v1/test/getcrumb'
    crumb_response = session.get(crumb_url)
    crumb = crumb_response.text
    
    # Step 2: Download historical data
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp.now().timestamp())
    
    download_url = (
        f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}'
        f'?period1={start_ts}&period2={end_ts}&interval=1d&events=history'
        f'&crumb={crumb}'
    )
    
    response = session.get(download_url)
    response.raise_for_status()
    
    # Step 3: Parse CSV
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    
    # Clean column names
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    df = df.rename(columns={'adj_close': 'adj close'})
    
    # Rename Date to date
    if 'date' in df.columns:
        pass
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    
    print(f'✓ Downloaded {len(df)} rows')
    print(f'✓ Date range: {df["date"].min()} to {df["date"].max()}')
    
    return df

if __name__ == '__main__':
    try:
        df = download_yahoo_data('AMZN', '2021-01-01')
        
        # Save to CSV
        output_path = 'C:/Users/rraga/Projects/qusa/data/raw/AMZN_2021-01-01_2026-01-10.csv'
        df.to_csv(output_path, index=False)
        
        print(f'✓ Saved to: {output_path}')
        print(f'\nFirst 5 rows:')
        print(df.head())
        print(f'\nLast 5 rows:')
        print(df.tail())
        
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
