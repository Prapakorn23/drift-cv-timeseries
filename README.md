# Time Series Model Comparison with Concept Drift Detection

โปรแกรมนี้ใช้สำหรับเปรียบเทียบประสิทธิภาพของโมเดลต่างๆ ในการทำนายราคาหุ้น โดยมีการตรวจจับ concept drift และใช้ cross-validation แบบ adaptive

## โครงสร้างไฟล์

- `main.py` - ไฟล์หลักสำหรับรันโปรแกรม
- `data_preparation.py` - คลาสสำหรับเตรียมข้อมูลและ feature engineering
- `models.py` - คลาสโมเดลต่างๆ (RNN, LSTM, GRU, Linear Regression)
- `drift_detection.py` - คลาสสำหรับตรวจจับ concept drift ด้วย ADWIN
- `cross_validation.py` - กลยุทธ์ cross-validation (Adaptive และ Baseline)
- `model_comparison.py` - คลาสสำหรับเปรียบเทียบโมเดล
- `requirements.txt` - รายการ dependencies

## การติดตั้ง

```bash
pip install -r requirements.txt
```

## การใช้งาน

### วิธีที่ 1: ใส่ path ผ่าน command line
```bash
python main.py "path/to/your/data.csv"
```

### วิธีที่ 2: ใส่ path เมื่อโปรแกรมถาม
```bash
python main.py
```
แล้วใส่ path ของไฟล์ CSV เมื่อโปรแกรมถาม

## ฟีเจอร์ใหม่

### 📊 การแสดงผลที่ปรับปรุงแล้ว
- แสดงผลแบบตารางที่อ่านง่าย
- แบ่งส่วนข้อมูลชัดเจนด้วย emoji และเส้นแบ่ง
- แสดงสถานะของแต่ละ fold (Valid/No Data)
- แสดงโมเดลที่ดีที่สุดอย่างเด่นชัด

### 💾 การ Export ผลลัพธ์
- บันทึกผลลัพธ์อัตโนมัติเป็นไฟล์ .txt
- ชื่อไฟล์จะรวม timestamp เพื่อไม่ให้ซ้ำกัน
- สามารถกำหนดชื่อไฟล์เองได้
- ไฟล์ที่ export จะมีรายละเอียดครบถ้วน รวมถึงผลลัพธ์แต่ละ fold

## รูปแบบไฟล์ CSV ที่รองรับ

ไฟล์ CSV ต้องมีคอลัมน์ต่อไปนี้:
- `Date` - วันที่ (รูปแบบ dd/mm/yyyy)
- `Close` - ราคาปิด
- `Volume` - ปริมาณการซื้อขาย
- `High` - ราคาสูงสุด (ถ้ามี)
- `Low` - ราคาต่ำสุด (ถ้ามี)

## Features ที่สร้างขึ้น

- `Return` - อัตราผลตอบแทน
- `Volatility` - ความผันผวน (7-day rolling standard deviation)
- `Volume_Log` - log ของปริมาณการซื้อขาย
- `Return_Volume` - ผลคูณระหว่าง Return และ Volume_Log

## โมเดลที่รองรับ

1. **RNN** - Simple Recurrent Neural Network
2. **LSTM** - Long Short-Term Memory
3. **GRU** - Gated Recurrent Unit
4. **LINEAR** - Linear Regression

## Cross-Validation Strategies

1. **Adaptive CV** - ใช้จุด drift ที่ตรวจพบเพื่อแบ่งข้อมูล
2. **Baseline CV** - แบ่งข้อมูลแบบ 5-fold ตามปกติ

## ตัวอย่างการใช้งาน

```python
from data_preparation import DataPreparator
from drift_detection import ADWINDriftDetector
from model_comparison import ModelComparison

# เตรียมข้อมูล
preparator = DataPreparator()
df, X, y = preparator.prepare_data("data.csv")

# ตรวจจับ drift
detector = ADWINDriftDetector(delta=0.01, min_fold_len=15)
drift_points = detector.detect(df, 'Close')

# เปรียบเทียบโมเดล
comparator = ModelComparison()
results = comparator.compare_models(X, y, drift_points)
comparator.print_summary(results)
```

## หมายเหตุ

- โปรแกรมจะตั้งค่า random seed เพื่อความเสถียรของผลลัพธ์
- TensorFlow จะถูกตั้งค่าให้ใช้ single thread เพื่อความเสถียร
- โปรแกรมจะข้าม fold ที่มีข้อมูลไม่เพียงพอสำหรับการ train/test
