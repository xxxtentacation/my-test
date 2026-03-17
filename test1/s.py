import pandas as pd
import numpy as np

# 设置随机种子以保证可复现
np.random.seed(42)

n = 2000  # 样本量

# 生成模拟数据
data = {
    'age': np.random.randint(40, 85, size=n),
    'male': np.random.choice([0, 1], size=n, p=[0.4, 0.6]),
    'bmi': np.random.normal(28, 5, size=n),
    'lvef.metabl': np.clip(np.random.normal(40, 12, size=n), 15, 70),  # LVEF 通常 15–70%
    'resting.systolic.bp': np.random.normal(130, 20, size=n),
    'resting.hr': np.random.normal(75, 15, size=n),
    'hgb': np.random.normal(13.5, 2, size=n),  # 血红蛋白 (g/dL)
    'creatinine': np.random.gamma(1.5, 0.8, size=n),  # 肌酐 (mg/dL)
    'glucose': np.random.normal(110, 30, size=n),
    'cad': np.random.choice([0, 1], size=n, p=[0.6, 0.4]),  # 冠心病
    'diabetes': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    'acei': np.random.choice([0, 1], size=n, p=[0.5, 0.5]),  # ACEI 类药物
    'betablok': np.random.choice([0, 1], size=n, p=[0.6, 0.4]),  # β受体阻滞剂
    'surgery.cabg': np.random.choice([0, 1], size=n, p=[0.85, 0.15]),  # 冠脉搭桥
}

# 生成 peak.vo2（基于 LVEF、年龄、性别等合理相关）
# 理论：年轻、男性、高 LVEF → 更高 peakVO2
peak_vo2 = (
    25
    - 0.1 * (data['age'] - 60)
    + 2.5 * data['male']
    + 0.15 * (data['lvef.metabl'] - 40)
    - 0.1 * (data['bmi'] - 28)
    - 0.5 * data['diabetes']
    + np.random.normal(0, 2, size=n)
)
data['peak.vo2'] = np.clip(peak_vo2, 8, 35)  # 合理范围：8–35 mL/kg/min

# 转为 DataFrame
df = pd.DataFrame(data)

# 引入少量缺失值（约 5%）
for col in ['lvef.metabl', 'hgb', 'peak.vo2', 'creatinine']:
    missing_idx = np.random.choice(df.index, size=int(0.05 * n), replace=False)
    df.loc[missing_idx, col] = np.nan

# 保存为 CSV
df.to_csv("E:\Study\python\WorkSpace\PyTorch\peakVO2.csv", index=False)

print("\n前5行预览:")
print(df.head())
print(f"\n数据形状: {df.shape}")
print("\n各列缺失值数量:")
print(df.isnull().sum())