# ต้องมาก่อนการใช้ TensorFlow ทุกอย่าง!
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

# ===== ตั้งค่าความเสถียรและ reproducibility =====
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ปิด multi-threading ของ TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(1)

warnings.filterwarnings('ignore')

def main():
    """Main function to run the model comparison analysis."""
    
    print("🚀 Time Series Model Comparison with Concept Drift Detection")
    print("="*70)
    
    # รับ path ของไฟล์ CSV จากผู้ใช้
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"📁 ใช้ไฟล์: {csv_path}")
    else:
        csv_path = input("📁 กรุณาใส่ path ของไฟล์ CSV ที่ต้องการวิเคราะห์: ").strip()
    
    # ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
    if not Path(csv_path).exists():
        print(f"❌ Error: ไฟล์ '{csv_path}' ไม่พบ กรุณาตรวจสอบ path อีกครั้ง")
        return
    
    try:
        # เตรียมข้อมูล
        print("📊 กำลังเตรียมข้อมูล...")
        preparator = DataPreparator()
        df, X, y = preparator.prepare_data(csv_path)
        print(f"✅ เตรียมข้อมูลเรียบร้อย: {len(df)} แถว, {len(X.columns)} features")
        
        # ตรวจจับ concept drift
        print("🔍 กำลังตรวจจับ concept drift...")
        detector = ADWINDriftDetector(delta=0.01, min_fold_len=15)
        drift_points = detector.detect(df, 'Close')
        drift_dates_formatted = df.iloc[drift_points]['Date'].dt.strftime('%d/%m/%Y').tolist()
        
        print(f"\n🔍 CONCEPT DRIFT DETECTION RESULTS")
        print("-" * 50)
        print(f"📅 จำนวน Drift Points ที่ตรวจพบ: {len(drift_points)}")
        print(f"📍 Drift Points (Index): {drift_points}")
        print(f"📅 Drift Dates: {drift_dates_formatted}")
        
        # ตั้งค่า parameters
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
        
        # เปรียบเทียบโมเดล
        print("\n🤖 กำลังเปรียบเทียบโมเดล...")
        comparator = ModelComparison(rnn_params=rnn_params, linear_params=linear_params)
        results = comparator.compare_models(X, y, drift_points)
        
        # แสดงผลลัพธ์แบบใหม่
        comparator.print_summary(results, drift_points, drift_dates_formatted)
        
        # Export ผลลัพธ์เป็นไฟล์ .txt
        print("\n💾 กำลังบันทึกผลลัพธ์...")
        export_filename = comparator.export_results(results, drift_points, drift_dates_formatted)
        print(f"✅ บันทึกผลลัพธ์เรียบร้อยแล้ว: {export_filename}")
        
        # ถามผู้ใช้ว่าต้องการ export ด้วยชื่อไฟล์ที่กำหนดเองหรือไม่
        try:
            custom_filename = input("\nต้องการบันทึกด้วยชื่อไฟล์ที่กำหนดเองหรือไม่? (กด Enter เพื่อข้าม): ").strip()
            if custom_filename:
                if not custom_filename.endswith('.txt'):
                    custom_filename += '.txt'
                export_filename = comparator.export_results(results, drift_points, drift_dates_formatted, custom_filename)
                print(f"✅ บันทึกผลลัพธ์เรียบร้อยแล้ว: {export_filename}")
        except KeyboardInterrupt:
            print("\n⏭️ ข้ามการบันทึกไฟล์เพิ่มเติม")
        
    except FileNotFoundError:
        print(f"❌ Error: ไม่พบไฟล์ '{csv_path}' กรุณาตรวจสอบ path อีกครั้ง")
    except Exception as e:
        print(f"❌ Error: เกิดข้อผิดพลาดในการวิเคราะห์: {e}")
        import traceback
        print("📋 รายละเอียดข้อผิดพลาด:")
        traceback.print_exc()

if __name__ == "__main__":
    main()