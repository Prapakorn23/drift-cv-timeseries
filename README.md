# Time Series Model Comparison with Concept Drift Detection

This program is used to compare the performance of different models in stock price prediction, featuring concept drift detection and adaptive cross-validation.

## File Structure

- `main.py` - Main file for running the program
- `data_preparation.py` - Class for data preparation and feature engineering
- `models.py` - Various model classes (RNN, LSTM, GRU, Linear Regression)
- `drift_detection.py` - Class for concept drift detection using ADWIN
- `cross_validation.py` - Cross-validation strategies (Adaptive and Baseline)
- `model_comparison.py` - Class for model comparison
- `requirements.txt` - List of dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Method 1: Pass path via command line
```bash
python main.py "path/to/your/data.csv"
```

### Method 2: Enter path when prompted
```bash
python main.py
```
Then enter the CSV file path when prompted

## New Features

### 📊 Enhanced Display
- Easy-to-read table format
- Clear data sections with emojis and dividers
- Shows status of each fold (Valid/No Data)
- Highlights the best performing model

### 💾 Result Export
- Automatically saves results to .txt file
- Filename includes timestamp to avoid duplicates
- Can specify custom filename
- Exported file contains detailed information including results for each fold

## Supported CSV File Format

The CSV file must contain the following columns:
- `Date` - Date (format dd/mm/yyyy)
- `Close` - Closing price
- `Volume` - Trading volume
- `High` - Highest price (if available)
- `Low` - Lowest price (if available)

## Generated Features

- `Return` - Return rate
- `Volatility` - Volatility (7-day rolling standard deviation)
- `Volume_Log` - Logarithm of trading volume
- `Return_Volume` - Product of Return and Volume_Log

## Supported Models

1. **RNN** - Simple Recurrent Neural Network
2. **LSTM** - Long Short-Term Memory
3. **GRU** - Gated Recurrent Unit
4. **LINEAR** - Linear Regression

## Cross-Validation Strategies

1. **Adaptive CV** - Uses detected drift points to split data
2. **Baseline CV** - Standard 5-fold data splitting

## Usage Example

```python
from data_preparation import DataPreparator
from drift_detection import ADWINDriftDetector
from model_comparison import ModelComparison

# Prepare data
preparator = DataPreparator()
df, X, y = preparator.prepare_data("data.csv")

# Detect drift
detector = ADWINDriftDetector(delta=0.01, min_fold_len=15)
drift_points = detector.detect(df, 'Close')

# Compare models
comparator = ModelComparison()
results = comparator.compare_models(X, y, drift_points)
comparator.print_summary(results)
```

## Notes

- The program sets random seed for result stability
- TensorFlow is configured to use single thread for stability
- The program skips folds with insufficient data for train/test