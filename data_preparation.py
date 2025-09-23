import pandas as pd
import numpy as np
from typing import Tuple

class DataPreparator:
    """Class for data preparation and feature engineering."""
    
    def __init__(self):
        pass
    
    def prepare_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Prepare data from CSV file with feature engineering.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Tuple of (df, X, y) where:
            - df: Original dataframe with engineered features
            - X: Feature matrix
            - y: Target variable (Close price)
        """
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
        except FileNotFoundError:
            print(f"Error: '{csv_path}' not found. Please ensure the file is in the same directory.")
            raise
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            raise

        df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
        df = df.sort_values("Date")

        # Feature engineering
        df['Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Close'].rolling(7).std()       
        df['Volume_Log'] = np.log1p(df['Volume'])
        df['Return_Volume'] = df['Return'] * df['Volume_Log']
        
        # Drop NaN values after rolling calculations
        df.dropna(inplace=True)

        X = df[['Return', 'Volatility','Volume_Log', 'Return_Volume']]
        y = df['Close']
        
        return df, X, y
