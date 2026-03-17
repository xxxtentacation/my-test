import pandas as pd
import numpy as np

# 设置随机种子以便复现
np.random.seed(42)

n = 10000  # 样本数

# 数值变量（模拟生理指标）
data = {
    'age': np.random.normal(50, 12, n).clip(18, 85),
    'height': np.random.normal(170, 10, n).clip(140, 200),
    'weight': np.random.normal(70, 15, n).clip(40, 120),
    'bmi': None,  # 将由 height 和 weight 计算
    'rest_hr': np.random.normal(72, 10, n).clip(40, 120),
    'max_hr': np.random.normal(180, 15, n).clip(120, 220),
    'peakVO2': np.random.normal(28, 6, n).clip(10, 50),  # 主要目标变量
    'vo2_max': np.random.normal(30, 7, n).clip(12, 55),
    'ventilation': np.random.normal(80, 20, n).clip(30, 150),
    'respiratory_rate': np.random.normal(16, 3, n).clip(8, 30),
    'sbp': np.random.normal(130, 15, n).clip(90, 180),
    'dbp': np.random.normal(80, 10, n).clip(60, 110),
    'cholesterol': np.random.normal(200, 30, n).clip(120, 300),
    'glucose': np.random.normal(95, 15, n).clip(70, 150),
    'hemoglobin': np.random.normal(14, 1.5, n).clip(10, 18),
    'o2_saturation': np.random.normal(97, 1.5, n).clip(90, 100),
    'fev1': np.random.normal(3.5, 0.8, n).clip(1.0, 6.0),
    'fvc': np.random.normal(4.2, 0.9, n).clip(1.5, 7.0),
    'constant_col1': [5.0] * n,  # 零方差列（会被剔除）
}

# 计算 BMI
data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)

# 分类变量
genders = np.random.choice(['Male', 'Female'], size=n, p=[0.5, 0.5])
smoking = np.random.choice(['Never', 'Former', 'Current'], size=n, p=[0.6, 0.3, 0.1])
exercise = np.random.choice(['Low', 'Medium', 'High'], size=n, p=[0.3, 0.4, 0.3])

data['gender'] = genders
data['smoking_status'] = smoking
data['exercise_group'] = exercise

# 构建 DataFrame
df = pd.DataFrame(data)

# 引入少量缺失值（约 2%）
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    missing_idx = np.random.choice(df.index, size=int(0.02 * n), replace=False)
    df.loc[missing_idx, col] = np.nan

# 保存到 CSV
output_path = r"E:\Study\python\WorkSpace\PyTorch\test2\peakVO2.csv"
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ 模拟数据已生成并保存至: {output_path}")
print(f"📊 数据形状: {df.shape}")
print(f"🔢 数值变量数: {len(numeric_cols)}")
print(f"🔤 分类变量: gender, smoking_status, exercise_group")
print("💡 注意：包含少量缺失值和一个零方差列（constant_col1），用于测试 PCA 预处理流程。")