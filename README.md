# Time Series Model Comparison with Concept Drift Detection

โปรแกรมเปรียบเทียบประสิทธิภาพของโมเดลต่างๆ ในการทำนายราคาหุ้น พร้อมระบบตรวจจับการเปลี่ยนแปลงของข้อมูล (Concept Drift) และการ Cross-Validation แบบปรับตัว

---

## 📁 โครงสร้างไฟล์

```
Project/
├── main.py                  # จุดเริ่มต้นของโปรแกรม
├── data_preparation.py      # เตรียมข้อมูลและสร้าง features
├── drift_detection.py       # ตรวจจับ Concept Drift ด้วย ADWIN
├── models.py               # โมเดล RNN, LSTM, GRU, Linear Regression
├── cross_validation.py     # กลยุทธ์ Cross-Validation
├── model_comparison.py     # เปรียบเทียบโมเดลและ export ผลลัพธ์
├── requirements.txt        # รายการ dependencies
└── README.md              # เอกสารนี้
```

---

## 🧩 คำอธิบายแต่ละไฟล์

### 1. `main.py` - จุดเริ่มต้นของโปรแกรม

**หน้าที่:**

- รับ CSV file path จากผู้ใช้ (ผ่าน command line หรือ input)
- เรียกใช้ `DataPreparator` เพื่อเตรียมข้อมูล
- เรียกใช้ `ADWINDriftDetector` เพื่อตรวจจับ drift points
- เรียกใช้ `ModelComparison` เพื่อเปรียบเทียบโมเดล
- แสดงผลและ export ผลลัพธ์เป็นไฟล์ TXT หรือ CSV

**การตั้งค่าสำคัญ:**

- กำหนด random seed = 42 เพื่อความเสถียร
- จำกัด TensorFlow ให้ใช้ single thread
- ปิดการแสดง warnings ทั้งหมด

---

### 2. `data_preparation.py` - เตรียมข้อมูลและสร้าง Features

**คลาส:** `DataPreparator`

**หน้าที่:**

1. โหลดข้อมูลจากไฟล์ CSV
2. แปลงคอลัมน์ `Date` ให้เป็น datetime
3. สร้าง features ใหม่:
   - **Return**: อัตราผลตอบแทน `(Close[t] - Close[t-1]) / Close[t-1]`
   - **Volatility**: ความผันผวน (7-day rolling std)
   - **Volume_Log**: `log(Volume + 1)`
   - **Return_Volume**: `Return × Volume_Log`
4. ลบ missing values
5. แยกข้อมูลเป็น Features (X) และ Target (y)

**Input:** ไฟล์ CSV ที่มี columns: `Date`, `Close`, `Volume`

**Output:**

- `df`: DataFrame ที่สมบูรณ์
- `X`: Features สำหรับการเทรน
- `y`: Target variable (Close price)

---

### 3. `drift_detection.py` - ตรวจจับ Concept Drift

**คลาส:** `ADWINDriftDetector`

**หลักการทำงาน:**

- ใช้ **ADWIN (Adaptive Windowing)** algorithm จาก `river` library
- ตรวจจับจุดที่การกระจายของข้อมูลเปลี่ยนแปลงอย่างมีนัยสำคัญ
- เก็บเฉพาะ drift points ที่มีระยะห่างมากกว่า `min_fold_len`

**Parameters:**

- `delta`: ระดับความเชื่อมั่นในการตรวจจับ (ค่าเล็ก = ตรวจจับไวขึ้น)
- `min_fold_len`: ระยะห่างขั้นต่ำระหว่าง drift points

**Output:** รายการ index ของจุดที่เกิด concept drift

---

### 4. `models.py` - โมเดลสำหรับการทำนาย

**โมเดลที่รองรับ:**

#### 4.1 `RNNRegressor` - เครือข่ายประสาทแบบวนซ้ำ

- รองรับ 3 แบบ: **RNN**, **LSTM**, **GRU**
- ใช้ `SequenceGenerator` เพื่อสร้าง sliding windows
- Parameters:
  - `sequence_length`: ความยาวของ sequence
  - `units`: จำนวน neurons
  - `dropout_rate`: อัตรา dropout
  - `learning_rate`: อัตราการเรียนรู้
  - `epochs`, `batch_size`

#### 4.2 `LinearRegressionModel` - Linear Regression

- โมเดลเชิงเส้นแบบง่าย
- ใช้ `sklearn.linear_model.LinearRegression`

---

### 5. `cross_validation.py` - กลยุทธ์ Cross-Validation

#### 5.1 `DriftAdaptiveTimeSeriesCV`

**หลักการ:**

- แบ่งข้อมูลตาม **drift points** ที่ตรวจจับได้
- แต่ละช่วงระหว่าง drift points จะถูกแบ่งเป็น train (80%) และ test (20%)
- เหมาะสำหรับข้อมูลที่มีการเปลี่ยนแปลงตามเวลา

**ข้อดี:**

- สะท้อนพฤติกรรมของข้อมูลจริง
- ทดสอบความสามารถในการปรับตัวของโมเดล

#### 5.2 `BaselineTimeSeriesCV`

**หลักการ:**

- แบ่งข้อมูลเป็น **5 ส่วนเท่าๆ กัน** (standard 5-fold)
- แต่ละส่วนแบ่งเป็น train (80%) และ test (20%)
- ใช้เป็น baseline สำหรับเปรียบเทียบ

**ทั้ง 2 class คืนค่า:**

- `all_folds`: RMSE และ MAE ของทุก folds
- `before_drift`: metrics ก่อนจุด drift แรก
- `after_drift`: metrics หลังจุด drift แรก

---

### 6. `model_comparison.py` - เปรียบเทียบโมเดล

**คลาส:** `ModelComparison`

**หน้าที่หลัก:**

#### 6.1 `compare_models()`

- เทรนและทดสอบโมเดล 4 แบบ: RNN, LSTM, GRU, LINEAR
- ใช้ทั้ง Adaptive CV และ Baseline CV
- เก็บผลลัพธ์ RMSE และ MAE

#### 6.2 `print_summary()`

- แสดงตารางเปรียบเทียบโมเดล
- แสดง drift points ที่ตรวจจับได้
- แสดงโมเดลที่ดีที่สุด (ตาม RMSE เฉลี่ย)

#### 6.3 `export_results()`

- Export ผลลัพธ์เป็นไฟล์ **.txt**
- รวมข้อมูลทุกอย่างรวมถึงผลละเอียดของแต่ละ fold

#### 6.4 `export_results_csv()`

- Export ผลลัพธ์เป็นไฟล์ **.csv**
- จัดรูปแบบให้เหมาะกับการวิเคราะห์ต่อ (เช่น นำเข้า Excel, Tableau)

---

## ⚙️ การติดตั้ง

### ขั้นตอนที่ 1: ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### ขั้นตอนที่ 2: ตรวจสอบ Python version

- ต้องการ Python 3.8 ขึ้นไป

---

## 🚀 วิธีการใช้งาน

### วิธีที่ 1: ระบุ path ผ่าน command line

```bash
python main.py "path/to/your/data.csv"
```

### วิธีที่ 2: ป้อน path เมื่อโปรแกรมถาม

```bash
python main.py
```

จากนั้นป้อน path ของไฟล์ CSV เมื่อโปรแกรมถาม

### การบันทึกผลลัพธ์

โปรแกรมจะถามว่าต้องการบันทึกผลลัพธ์หรือไม่:

- กด `1` → บันทึกเป็น `.txt`
- กด `2` → บันทึกเป็น `.csv`
- กด `Enter` → ข้ามการบันทึก

---

## � รูปแบบไฟล์ CSV ที่รองรับ

ไฟล์ CSV ต้องมีคอลัมน์ดังต่อไปนี้:

| Column   | คำอธิบาย            | ตัวอย่าง   |
| -------- | ------------------- | ---------- |
| `Date`   | วันที่ (dd/mm/yyyy) | 01/01/2020 |
| `Close`  | ราคาปิด             | 150.25     |
| `Volume` | ปริมาณการซื้อขาย    | 1000000    |
| `High`   | ราคาสูงสุด (ถ้ามี)  | 152.00     |
| `Low`    | ราคาต่ำสุด (ถ้ามี)  | 148.50     |

---

## 📈 Features ที่สร้างขึ้นอัตโนมัติ

| Feature         | สูตร                                   | ความหมาย                              |
| --------------- | -------------------------------------- | ------------------------------------- |
| `Return`        | `(Close[t] - Close[t-1]) / Close[t-1]` | อัตราผลตอบแทน                         |
| `Volatility`    | `rolling_std(Return, window=7)`        | ความผันผวน 7 วัน                      |
| `Volume_Log`    | `log(Volume + 1)`                      | Logarithm ของปริมาณการซื้อขาย         |
| `Return_Volume` | `Return × Volume_Log`                  | ความสัมพันธ์ระหว่าง Return และ Volume |

---

## 🤖 โมเดลที่รองรับ

1. **RNN** - Simple Recurrent Neural Network
2. **LSTM** - Long Short-Term Memory (เหมาะกับข้อมูล time series)
3. **GRU** - Gated Recurrent Unit (เร็วกว่า LSTM แต่ประสิทธิภาพใกล้เคียง)
4. **LINEAR** - Linear Regression (baseline model)

---

## 🔄 กลยุทธ์ Cross-Validation

### 1. Adaptive CV

- แบ่งข้อมูลตาม drift points
- สะท้อนการเปลี่ยนแปลงของข้อมูลตามเวลา
- เหมาะสำหรับข้อมูลที่มี concept drift

### 2. Baseline CV

- แบ่งข้อมูลแบบมาตรฐาน (5-fold)
- ใช้เป็น baseline สำหรับเปรียบเทียบ

---

## 📝 ตัวอย่างการใช้งาน (API)

```python
from data_preparation import DataPreparator
from drift_detection import ADWINDriftDetector
from model_comparison import ModelComparison

# 1. เตรียมข้อมูล
preparator = DataPreparator()
df, X, y = preparator.prepare_data("data.csv")

# 2. ตรวจจับ drift
detector = ADWINDriftDetector(delta=0.01, min_fold_len=15)
drift_points = detector.detect(df, 'Close')

# 3. ตั้งค่า parameters สำหรับโมเดล
rnn_params = {
    'sequence_length': 15,
    'units': 32,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32
}

linear_params = {
    'fit_intercept': True
}

# 4. เปรียบเทียบโมเดล
comparator = ModelComparison(rnn_params=rnn_params, linear_params=linear_params)
results = comparator.compare_models(X, y, drift_points)

# 5. แสดงผลและบันทึก
comparator.print_summary(results, drift_points)
comparator.export_results(results, drift_points, filename="results.txt")
comparator.export_results_csv(results, drift_points, filename="results.csv")
```

---

## 🔍 หลักการทำงานของโปรแกรม

```
┌─────────────────────┐
│  1. โหลดข้อมูล CSV  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. เตรียมข้อมูล    │
│  - สร้าง features   │
│  - ลบ missing       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. ตรวจจับ Drift   │
│  (ADWIN Algorithm)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. แบ่งข้อมูล      │
│  - Adaptive CV      │
│  - Baseline CV      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. เทรนโมเดล       │
│  (RNN/LSTM/GRU/LR)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  6. ประเมินผล       │
│  - คำนวณ RMSE/MAE  │
│  - หาโมเดลที่ดีสุด │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  7. แสดงผลและ       │
│     Export          │
└─────────────────────┘
```

---

## 📊 ตัวอย่างผลลัพธ์

```
🚀 Time Series Model Comparison with Concept Drift Detection
======================================================================

🔍 CONCEPT DRIFT DETECTION RESULTS
--------------------------------------------------
📅 Number of drift points detected: 3
📍 Drift points (index): [50, 120, 180]
📅 Drift dates: ['15/03/2020', '10/07/2020', '05/11/2020']

📈 MODEL PERFORMANCE COMPARISON
--------------------------------------------------
Model      Strategy        Avg RMSE    Avg MAE     Folds   Status
--------------------------------------------------
RNN        Adaptive CV     2.345       1.890       4       ✅ Valid
RNN        Baseline CV     2.567       2.012       5       ✅ Valid
LSTM       Adaptive CV     2.123       1.765       4       ✅ Valid
LSTM       Baseline CV     2.389       1.923       5       ✅ Valid
...

🏆 BEST PERFORMING MODEL
--------------------------------------------------
🥇 Winner: LSTM
📊 Best Average RMSE: 2.123
```

---

## ⚠️ หมายเหตุสำคัญ

1. **Reproducibility:**
   - โปรแกรมตั้ง random seed = 42 เพื่อให้ผลลัพธ์เหมือนกันทุกครั้ง
   - TensorFlow ใช้ single thread เพื่อความเสถียร

2. **Data Requirements:**
   - ต้องมีข้อมูลเพียงพอสำหรับการเทรน (แนะนำ > 100 rows)
   - แต่ละ fold ต้องมีข้อมูลมากกว่า `sequence_length` สำหรับโมเดล RNN

3. **Performance:**
   - การเทรนโมเดล RNN/LSTM/GRU ใช้เวลานานกว่า Linear Regression
   - สามารถปรับ `epochs` และ `batch_size` เพื่อเพิ่มความเร็ว

4. **Missing Data:**
   - โปรแกรมจะลบแถวที่มี NaN ออกอัตโนมัติ
   - ถ้าข้อมูลหายมาก อาจส่งผลต่อประสิทธิภาพ

---

## 📦 Dependencies

- `numpy >= 1.21.0` - การคำนวณทางสถิติ
- `pandas >= 1.3.0` - จัดการข้อมูล
- `tensorflow >= 2.8.0` - โมเดล deep learning
- `scikit-learn >= 1.0.0` - Linear Regression และ metrics
- `scipy >= 1.7.0` - ฟังก์ชันทางสถิติ
- `river >= 0.15.0` - ADWIN algorithm

---

## 🎯 การปรับแต่ง Parameters

### สำหรับ RNN/LSTM/GRU:

```python
rnn_params = {
    'sequence_length': 15,    # ความยาว sequence (เพิ่มถ้าต้องการ pattern ยาวขึ้น)
    'units': 32,              # จำนวน neurons (เพิ่มถ้าต้องการโมเดลซับซ้อนขึ้น)
    'dropout_rate': 0.2,      # อัตรา dropout (เพิ่มถ้า overfitting)
    'learning_rate': 0.001,   # อัตราการเรียนรู้
    'epochs': 50,             # จำนวนรอบการเทรน
    'batch_size': 32,         # ขนาด batch
    'verbose': 0              # 0=ไม่แสดงผล, 1=แสดงผลการเทรน
}
```

### สำหรับ ADWIN:

```python
detector = ADWINDriftDetector(
    delta=0.01,          # ลดค่าถ้าต้องการตรวจจับไวขึ้น
    min_fold_len=15      # ระยะห่างขั้นต่ำระหว่าง drift points
)
```

---

## 📧 การแก้ปัญหา

### ปัญหา: "Not enough data to calculate metrics"

**สาเหตุ:** ข้อมูลในบาง fold น้อยเกินไป
**แก้ไข:** ลด `sequence_length` หรือเพิ่มข้อมูล

### ปัญหา: โมเดล RNN ช้ามาก

**แก้ไข:**

- ลด `epochs` (เช่น จาก 50 → 20)
- ลด `units` (เช่น จาก 32 → 16)
- เพิ่ม `batch_size`

### ปัญหา: ผลลัพธ์แตกต่างกันทุกครั้ง

**แก้ไข:** ตรวจสอบว่ามีการตั้ง random seed ครบทุกที่

---

## 📄 License

This project is for educational purposes.
