import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime
from cross_validation import DriftAdaptiveTimeSeriesCV, BaselineTimeSeriesCV

class ModelComparison:
    """Compares the performance of different models using two CV strategies."""
    def __init__(self, rnn_params: dict = None, linear_params: dict = None):
        self.rnn_params = rnn_params or {}
        self.linear_params = linear_params or {}
        self.models = ['RNN', 'LSTM', 'GRU', 'LINEAR']
        
    def compare_models(self, X: pd.DataFrame, y: pd.Series, drift_points: List[int]) -> Dict:
        results = {}

        for model_type in self.models:
            params = self.linear_params if model_type == 'LINEAR' else self.rnn_params
            print(f"\n{'='*50}")
            print(f"Testing {model_type} Model")
            print(f"{'='*50}")
            # Adaptive CV uses all detected drift points and returns structured data
            drift_cv = DriftAdaptiveTimeSeriesCV(model_type, params)
            drift_results = drift_cv.run(X, y, drift_points)
            
            # Extract metrics for compatibility
            drift_rmse = drift_results['all_folds']['rmse']
            drift_mae = drift_results['all_folds']['mae']
            
            # Baseline CV uses standard 5-fold splitting
            baseline_cv = BaselineTimeSeriesCV(model_type, params, n_splits=5)
            baseline_results = baseline_cv.run(X, y, drift_points)
            
            # Extract metrics from baseline results
            base_rmse = baseline_results['all_folds']['rmse']
            base_mae = baseline_results['all_folds']['mae']
            
            results[model_type] = {
                'adaptive_rmse': drift_rmse,
                'adaptive_mae': drift_mae,
                'baseline_rmse': base_rmse,
                'baseline_mae': base_mae
            }
        
        return results
    
    def print_summary(self, results: Dict, drift_points: List[int] = None, drift_dates: List[str] = None, filename: str = None):
        """Print formatted summary of model comparison results."""
        print("\n" + "="*100)
        print("📊 MODEL COMPARISON SUMMARY")
        print("="*100)
        
        # Display analyzed filename
        if filename:
            print(f"📁 Analyzed file: {filename}")
            print("-" * 50)
        
        # Display drift points information
        if drift_points and drift_dates:
            print(f"\n🔍 CONCEPT DRIFT DETECTION RESULTS")
            print("-" * 50)
            print(f"📅 Number of drift points detected: {len(drift_points)}")
            print(f"📍 Drift points (index): {drift_points}")
            print(f"📅 Drift dates: {drift_dates}")
        
        # Create comparison table
        print(f"\n📈 MODEL PERFORMANCE COMPARISON")
        print("-" * 100)
        print(f"{'Model':<10} {'Strategy':<15} {'Avg RMSE':<12} {'Avg MAE':<12} {'Folds':<8} {'Status':<10}")
        print("-" * 100)
        
        for model_type in self.models:
            if model_type in results:
                # Adaptive CV Results
                if results[model_type]['adaptive_rmse']:
                    avg_rmse = np.mean(results[model_type]['adaptive_rmse'])
                    avg_mae = np.mean(results[model_type]['adaptive_mae'])
                    fold_count = len(results[model_type]['adaptive_rmse'])
                    print(f"{model_type:<10} {'Adaptive CV':<15} {avg_rmse:<12.3f} {avg_mae:<12.3f} {fold_count:<8} {'✅ Valid':<10}")
                else:
                    print(f"{model_type:<10} {'Adaptive CV':<15} {'N/A':<12} {'N/A':<12} {'0':<8} {'❌ No Data':<10}")
                
                # Baseline CV Results
                if results[model_type]['baseline_rmse']:
                    avg_rmse = np.mean(results[model_type]['baseline_rmse'])
                    avg_mae = np.mean(results[model_type]['baseline_mae'])
                    fold_count = len(results[model_type]['baseline_rmse'])
                    print(f"{model_type:<10} {'Baseline CV':<15} {avg_rmse:<12.3f} {avg_mae:<12.3f} {fold_count:<8} {'✅ Valid':<10}")
                else:
                    print(f"{model_type:<10} {'Baseline CV':<15} {'N/A':<12} {'N/A':<12} {'0':<8} {'❌ No Data':<10}")
        
        # Display best performing model
        best_model = self._find_best_model(results)
        if best_model:
            print(f"\n🏆 BEST PERFORMING MODEL")
            print("-" * 50)
            print(f"🥇 Winner: {best_model}")
            if best_model in results:
                best_scores = results[best_model]['adaptive_rmse'] or results[best_model]['baseline_rmse']
                if best_scores:
                    best_avg_rmse = np.mean(best_scores)
                    print(f"📊 Best Average RMSE: {best_avg_rmse:.3f}")
        
        print("\n" + "="*100)
    
    def export_results(self, results: Dict, drift_points: List[int] = None, 
                      drift_dates: List[str] = None, filename: str = None, analyzed_file: str = None) -> str:
        """Export results to a text file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_results_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("📊 MODEL COMPARISON RESULTS\n")
            f.write("="*100 + "\n")
            f.write(f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Display analyzed filename
            if analyzed_file:
                f.write(f"📁 Analyzed file: {analyzed_file}\n")
            
            f.write("\n")
            
            # Display drift points information
            if drift_points and drift_dates:
                f.write("🔍 CONCEPT DRIFT DETECTION RESULTS\n")
                f.write("-" * 50 + "\n")
                f.write(f"📅 Number of drift points detected: {len(drift_points)}\n")
                f.write(f"📍 Drift points (index): {drift_points}\n")
                f.write(f"📅 Drift dates: {drift_dates}\n\n")
            
            # Create comparison table
            f.write("📈 MODEL PERFORMANCE COMPARISON\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Model':<10} {'Strategy':<15} {'Avg RMSE':<12} {'Avg MAE':<12} {'Folds':<8} {'Status':<10}\n")
            f.write("-" * 100 + "\n")
            
            for model_type in self.models:
                if model_type in results:
                    # Adaptive CV Results
                    if results[model_type]['adaptive_rmse']:
                        avg_rmse = np.mean(results[model_type]['adaptive_rmse'])
                        avg_mae = np.mean(results[model_type]['adaptive_mae'])
                        fold_count = len(results[model_type]['adaptive_rmse'])
                        f.write(f"{model_type:<10} {'Adaptive CV':<15} {avg_rmse:<12.3f} {avg_mae:<12.3f} {fold_count:<8} {'Valid':<10}\n")
                    else:
                        f.write(f"{model_type:<10} {'Adaptive CV':<15} {'N/A':<12} {'N/A':<12} {'0':<8} {'No Data':<10}\n")
                    
                    # Baseline CV Results
                    if results[model_type]['baseline_rmse']:
                        avg_rmse = np.mean(results[model_type]['baseline_rmse'])
                        avg_mae = np.mean(results[model_type]['baseline_mae'])
                        fold_count = len(results[model_type]['baseline_rmse'])
                        f.write(f"{model_type:<10} {'Baseline CV':<15} {avg_rmse:<12.3f} {avg_mae:<12.3f} {fold_count:<8} {'Valid':<10}\n")
                    else:
                        f.write(f"{model_type:<10} {'Baseline CV':<15} {'N/A':<12} {'N/A':<12} {'0':<8} {'No Data':<10}\n")
            
            # Display best performing model
            best_model = self._find_best_model(results)
            if best_model:
                f.write(f"\n🏆 BEST PERFORMING MODEL\n")
                f.write("-" * 50 + "\n")
                f.write(f"🥇 Winner: {best_model}\n")
                if best_model in results:
                    best_scores = results[best_model]['adaptive_rmse'] or results[best_model]['baseline_rmse']
                    if best_scores:
                        best_avg_rmse = np.mean(best_scores)
                        f.write(f"📊 Best Average RMSE: {best_avg_rmse:.3f}\n")
            
            # Display detailed results for each fold
            f.write(f"\n📋 DETAILED FOLD RESULTS\n")
            f.write("="*100 + "\n")
            
            for model_type in self.models:
                if model_type in results:
                    f.write(f"\n{model_type} MODEL DETAILS:\n")
                    f.write("-" * 50 + "\n")
                    
                    # Adaptive CV Details
                    if results[model_type]['adaptive_rmse']:
                        f.write("Adaptive CV Results:\n")
                        for i, (rmse, mae) in enumerate(zip(results[model_type]['adaptive_rmse'], 
                                                           results[model_type]['adaptive_mae'])):
                            f.write(f"  Fold {i+1}: RMSE={rmse:.3f}, MAE={mae:.3f}\n")
                    
                    # Baseline CV Details
                    if results[model_type]['baseline_rmse']:
                        f.write("Baseline CV Results:\n")
                        for i, (rmse, mae) in enumerate(zip(results[model_type]['baseline_rmse'], 
                                                           results[model_type]['baseline_mae'])):
                            f.write(f"  Fold {i+1}: RMSE={rmse:.3f}, MAE={mae:.3f}\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("End of Report\n")
        
        return filename
    
    def export_results_csv(self, results: Dict, drift_points: List[int] = None, 
                           drift_dates: List[str] = None, filename: str = None, 
                           analyzed_file: str = None) -> str:
        """Export results to CSV format for easy upload and analysis."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_results_{timestamp}.csv"
        
        # Prepare data for CSV export
        csv_data = []
        
        # Add summary information as first rows
        csv_data.append({
            'Section': 'Metadata',
            'Field': 'Generated Date',
            'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Model': '',
            'Strategy': '',
            'Fold': '',
            'RMSE': '',
            'MAE': ''
        })
        
        if analyzed_file:
            csv_data.append({
                'Section': 'Metadata',
                'Field': 'Analyzed File',
                'Value': analyzed_file,
                'Model': '',
                'Strategy': '',
                'Fold': '',
                'RMSE': '',
                'MAE': ''
            })
        
        if drift_points and drift_dates:
            csv_data.append({
                'Section': 'Drift Detection',
                'Field': 'Number of Drift Points',
                'Value': len(drift_points),
                'Model': '',
                'Strategy': '',
                'Fold': '',
                'RMSE': '',
                'MAE': ''
            })
            csv_data.append({
                'Section': 'Drift Detection',
                'Field': 'Drift Points (Index)',
                'Value': str(drift_points),
                'Model': '',
                'Strategy': '',
                'Fold': '',
                'RMSE': '',
                'MAE': ''
            })
            csv_data.append({
                'Section': 'Drift Detection',
                'Field': 'Drift Dates',
                'Value': str(drift_dates),
                'Model': '',
                'Strategy': '',
                'Fold': '',
                'RMSE': '',
                'MAE': ''
            })
        
        # Add best model information
        best_model = self._find_best_model(results)
        if best_model:
            best_scores = results[best_model]['adaptive_rmse'] or results[best_model]['baseline_rmse']
            if best_scores:
                best_avg_rmse = np.mean(best_scores)
                csv_data.append({
                    'Section': 'Best Model (Accuracy)',
                    'Field': 'Winner',
                    'Value': best_model,
                    'Model': '',
                    'Strategy': '',
                    'Fold': '',
                    'RMSE': best_avg_rmse,
                    'MAE': ''
                })
        
        # Add summary comparison table
        for model_type in self.models:
            if model_type in results:
                # Adaptive CV Results
                if results[model_type]['adaptive_rmse']:
                    avg_rmse = np.mean(results[model_type]['adaptive_rmse'])
                    avg_mae = np.mean(results[model_type]['adaptive_mae'])
                    fold_count = len(results[model_type]['adaptive_rmse'])
                    csv_data.append({
                        'Section': 'Summary',
                        'Field': 'Average Performance',
                        'Value': 'Valid',
                        'Model': model_type,
                        'Strategy': 'Adaptive CV',
                        'Fold': fold_count,
                        'RMSE': avg_rmse,
                        'MAE': avg_mae
                    })
                else:
                    csv_data.append({
                        'Section': 'Summary',
                        'Field': 'Average Performance',
                        'Value': 'No Data',
                        'Model': model_type,
                        'Strategy': 'Adaptive CV',
                        'Fold': 0,
                        'RMSE': np.nan,
                        'MAE': np.nan
                    })
                
                # Baseline CV Results
                if results[model_type]['baseline_rmse']:
                    avg_rmse = np.mean(results[model_type]['baseline_rmse'])
                    avg_mae = np.mean(results[model_type]['baseline_mae'])
                    fold_count = len(results[model_type]['baseline_rmse'])
                    csv_data.append({
                        'Section': 'Summary',
                        'Field': 'Average Performance',
                        'Value': 'Valid',
                        'Model': model_type,
                        'Strategy': 'Baseline CV',
                        'Fold': fold_count,
                        'RMSE': avg_rmse,
                        'MAE': avg_mae
                    })
                else:
                    csv_data.append({
                        'Section': 'Summary',
                        'Field': 'Average Performance',
                        'Value': 'No Data',
                        'Model': model_type,
                        'Strategy': 'Baseline CV',
                        'Fold': 0,
                        'RMSE': np.nan,
                        'MAE': np.nan
                    })
        
        # Add detailed fold results
        for model_type in self.models:
            if model_type in results:
                # Adaptive CV Details
                if results[model_type]['adaptive_rmse']:
                    for i, (rmse, mae) in enumerate(zip(results[model_type]['adaptive_rmse'], 
                                                       results[model_type]['adaptive_mae'])):
                        csv_data.append({
                            'Section': 'Detailed Results',
                            'Field': f'Fold {i+1}',
                            'Value': 'Adaptive CV',
                            'Model': model_type,
                            'Strategy': 'Adaptive CV',
                            'Fold': i+1,
                            'RMSE': rmse,
                            'MAE': mae
                        })
                
                # Baseline CV Details
                if results[model_type]['baseline_rmse']:
                    for i, (rmse, mae) in enumerate(zip(results[model_type]['baseline_rmse'], 
                                                       results[model_type]['baseline_mae'])):
                        csv_data.append({
                            'Section': 'Detailed Results',
                            'Field': f'Fold {i+1}',
                            'Value': 'Baseline CV',
                            'Model': model_type,
                            'Strategy': 'Baseline CV',
                            'Fold': i+1,
                            'RMSE': rmse,
                            'MAE': mae
                        })
        
        # Convert to DataFrame and save
        df_export = pd.DataFrame(csv_data)
        df_export.to_csv(filename, index=False, encoding='utf-8-sig')
        
        return filename
    
    def _find_best_model(self, results: Dict) -> str:
        best_model, best_score = None, float('inf')
        for model_type in self.models:
            if model_type in results:
                scores = results[model_type]['adaptive_rmse'] or results[model_type]['baseline_rmse']
                if scores and np.mean(scores) < best_score:
                    best_score = np.mean(scores)
                    best_model = model_type
        return best_model
