# QUSA/scripts/run_FE_pipeline.py

import logging
import os 
import pandas as pd 
import sys 
import yaml 

from datetime import datetime

# add parent directory to sys.path for module imports
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from qusa.features.pipeline import FeaturePipeline


def setup_logger(name): 
    """
    Setup logger with console and file handlers. 
    
    Parameters:
        1) name (str): Name of the logger.
        
    Returns:
        1) logger (logging.Logger): Configured logger object.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()


    # create log directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # console handler 
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    # file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        f'{log_dir}/fe_pipeline_{timestamp}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_format)

    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger 


def load_config(config_path): 
    """ 
    Load configuration from YAML file. 
    
    Parameters: 
        1) config_path (str): Path to the configuration YAML file.
    
    Returns: 
        1) config (dict): Configuration dictionary.
    """

    try: 
        with open(config_path, "r") as f: 
            config = yaml.safe_load(f)
        return config
    
    except FileNotFoundError: 
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e: 
        raise ValueError(f"Error parsing YAML file: {e}")
    

def validate_dataframe(df, required_columns): 
    """ 
    Validate that the DataFrame contains all required columns. 
    
    Parameters: 
        1) df (pd.DataFrame): DataFrame to validate.
        2) required_columns (list): List of required column names.
    
    Raises: 
        1) ValueError: If any required column is missing.
    """

    # store required columns not found in input DataFrame 
    missing_columns = set(required_columns) - set(df.columns)

    # raise error when missing columns exist 
    if missing_columns: 
        raise ValueError(
            f"DataFrame is missing required columns: {missing_columns}"
        )
    

def main(): 
    """ 
    Main function to run the feature engineering pipeline.
    """

    logger = setup_logger("FE_pipeline")
    logger.info("=" * 80)
    logger.info("Starting Feature Engineering Pipeline")
    logger.info("=" * 80)

    
    try:  # load config file 
        logger.info("Loading configuration file...")

        config = load_config("config.yaml")

        logger.info("✓ Configuration loaded successfully")
        logger.info(f"  Data directory: {config['data']['data_dir']}")
        logger.info(f"  Output directory: {config['data']['processed_data_dir']}")

    except Exception as e:  # unable to load config 
        logger.error(f"✗ Error loading configuration: {e}")
        return 1

    try:  # load data 
        logger.info("Loading data...")
        data_path = os.path.expanduser(config['data']['data_dir'])
        tickers = config['data']['tickers']

        # for simplicity, process only the first ticker
        ticker = tickers[0]  

        data_path = os.path.join(data_path, f"{ticker}_2023-12-01_2025-12-01.csv")

        # handle case where path to data does not exist 
        if not os.path.exists(data_path): 
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        data = pd.read_csv(data_path)
        logger.info("✓ Data loaded successfully")
        logger.info(f"  Shape: {data.shape}")
        logger.info(f"  Columns: {list(data.columns)}")
        logger.info(f"  Date range: {data['date'].min()} to {data['date'].max()}")

    except Exception as e:  # unable to load data
        logger.error(f"✗ Error loading data: {e}")
        return 1
    
    try: # validate dataframe 
        logger.info("Validating input DataFrame...")
        required_columns = [
            config.get('fe_params', {}).get('date_col', 'date'), 
            config.get('fe_params', {}).get('open_col', 'open'), 
            config.get('fe_params', {}).get('close_col', 'close'), 
            config.get('fe_params', {}).get('high_col', 'high'), 
            config.get('fe_params', {}).get('low_col', 'low'), 
            config.get('fe_params', {}).get('volume_col', 'volume')
        ]
        validate_dataframe(data, required_columns)
        logger.info("✓ Input DataFrame validation successful")
    
    except Exception as e:  # DataFrame missing required columns 
        logger.error(f"✗ DataFrame validation error: {e}")
        return 1

    try:  # run FE pipeline
        logger.info("Running Feature Engineering Pipeline...")

        # initialize pipeline object and run on DataFrame
        fe_pipeline = FeaturePipeline({
            'date_col': 'date', 
            'open_col': 'open', 
            'close_col': 'close', 
            'high_col': 'high', 
            'low_col': 'low', 
            'volume_col': 'volume',
            'overnight': {
                'abnormal_threshold': config["analysis"]["abnormal_threshold"]
            }, 
            'technical_params': config['features']
        })

        processed_data = fe_pipeline.run(data)

        logger.info("Feature Engineering Pipeline completed successfully.")
        logger.info(f"  Output shape: {processed_data.shape}")
        logger.info(f"  Features added: {processed_data.shape[1] - data.shape[1]}")

    except Exception as e:  # FE pipeline error
        logger.error(f"✗ Error during Feature Engineering: {e}")
        logger.exception("Full traceback:")
        return 1
    
    try:  # save processed data
        logger.info("Saving processed data...")

        # extract output path from config
        processed_dir = os.path.expanduser(config['data']['processed_data_dir'])
        os.makedirs(processed_dir, exist_ok=True)

        output_path = os.path.join(
            processed_dir, 
            f"{ticker}_processed.csv"
        )

        # create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        processed_data.to_csv(output_path, index=False)
        logger.info(f"✓ Processed data saved to {output_path}")

    except Exception as e:  # unable to save data
        logger.error(f"✗ Error saving processed data: {e}")
        return 1
    
    # Summary
    logger.info("=" * 80)
    logger.info("Pipeline Execution Summary")
    logger.info("=" * 80)
    logger.info(f"Input file: {data_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Rows processed: {len(processed_data)}")
    logger.info(f"Features created: {len(fe_pipeline.get_engineered_features())}")
    logger.info("=" * 80)
    logger.info("✓ Pipeline completed successfully!")
    logger.info("=" * 80)
    
    return 0

    

if __name__ == "__main__": 
    exit_code = main()
    sys.exit(exit_code)

