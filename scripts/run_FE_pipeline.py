# QUSA/scripts/run_FE_pipeline.py

import pandas as pd 
import yaml 

from qusa.features.pipelien import FE_pipeline


def main(): 
    """ 
    Main function to run the feature engineering pipeline.
    """

    logger = setup_logger("FE_pipeline")

    logger.info("Starting Feature Engineering Pipeline...")

    
    try:  # load config file 
        logger.info("Loading configuration file...")
        with open("./config.yaml", "r") as f: 
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")

    except Exception as e: 
        logger.error(f"Error loading configuration file: {e}")
        return

    try:  # load data 
        logger.info("Loading data...")
        data = pd.read_csv(config["data_path"])
        logger.info("Data loaded successfully.")
    
    except Exception as e: 
        logger.error(f"Error loading data: {e}")
        return
    
    try:  # run FE pipeline
        logger.info("Running Feature Engineering Pipeline...")

        # initialize FE pipeline object 
        fe_pipeline = FE_pipeline(config["fe_params"])
        # execute FE pipeline on input data
        processed_data = fe_pipeline.run(data)

        logger.info("Feature Engineering Pipeline completed successfully.")

    except Exception as e: 
        logger.error(f"Error during Feature Engineering Pipeline execution: {e}")
        return
    

if __name__ == "__main__": 
    main()

