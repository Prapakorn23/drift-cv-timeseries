# Must be before any TensorFlow usage!
import os
import random
import numpy as np
import tensorflow as tf
import warnings
import sys
from pathlib import Path

# Import custom modules
from data_preparation import DataPreparator
from drift_detection import ADWINDriftDetector
from model_comparison import ModelComparison

# ===== Set stability and reproducibility =====
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Disable TensorFlow multi-threading
tf.config.threading.set_intra_op_parallelism_threads(1)

warnings.filterwarnings('ignore')

def main():
    """Main function to run the model comparison analysis."""
    
    print("🚀 Time Series Model Comparison with Concept Drift Detection")
    print("="*70)
    
    # Get CSV file path from user
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"📁 Using file: {csv_path}")
    else:
        csv_path = input("📁 Please enter the path of the CSV file to analyze: ").strip()
    
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"❌ Error: File '{csv_path}' not found. Please check the path again.")
        return
    
    try:
        # Prepare data
        print("📊 Preparing data...")
        preparator = DataPreparator()
        df, X, y = preparator.prepare_data(csv_path)
        print(f"✅ Data preparation complete: {len(df)} rows, {len(X.columns)} features")
        
        # Detect concept drift
        print("🔍 Detecting concept drift...")
        detector = ADWINDriftDetector(delta=0.01, min_fold_len=15)
        drift_points = detector.detect(df, 'Close')
        drift_dates_formatted = df.iloc[drift_points]['Date'].dt.strftime('%d/%m/%Y').tolist()
        
        print(f"\n🔍 CONCEPT DRIFT DETECTION RESULTS")
        print("-" * 50)
        print(f"📅 Number of drift points detected: {len(drift_points)}")
        print(f"📍 Drift points (index): {drift_points}")
        print(f"📅 Drift dates: {drift_dates_formatted}")
        
        # Set parameters
        rnn_params = {
            'sequence_length': 15, 
            'units': 32, 
            'dropout_rate': 0.2, 
            'learning_rate': 0.001, 
            'epochs': 50, 
            'batch_size': 32, 
            'verbose': 0
        }
        linear_params = {'fit_intercept': True}
        
        # Compare models
        print("\n🤖 Comparing models...")
        comparator = ModelComparison(rnn_params=rnn_params, linear_params=linear_params)
        results = comparator.compare_models(X, y, drift_points)
        
        # Display results
        comparator.print_summary(results, drift_points, drift_dates_formatted)
        
        # Export results to .txt file
        print("\n💾 Saving results...")
        export_filename = comparator.export_results(results, drift_points, drift_dates_formatted)
        print(f"✅ Results saved successfully: {export_filename}")
        
        # Ask user if they want to export with custom filename
        try:
            custom_filename = input("\nDo you want to save with a custom filename? (Press Enter to skip): ").strip()
            if custom_filename:
                if not custom_filename.endswith('.txt'):
                    custom_filename += '.txt'
                export_filename = comparator.export_results(results, drift_points, drift_dates_formatted, custom_filename)
                print(f"✅ Results saved successfully: {export_filename}")
        except KeyboardInterrupt:
            print("\n⏭️ Skipping additional file save")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{csv_path}' not found. Please check the path again.")
    except Exception as e:
        print(f"❌ Error: An error occurred during analysis: {e}")
        import traceback
        print("📋 Error details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()