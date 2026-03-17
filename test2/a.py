# ========== PCA 降维分析 (Python 版本) ==========

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# 1. 设置路径并读取数据
data_path = r"E:\Study\python\WorkSpace\PyTorch\test2\peakVO2.csv"
output_dir = r"E:\Study\python\WorkSpace\PyTorch\test2"

df = pd.read_csv(data_path)
print("=== 数据基本信息 ===")
print(f"数据维度: {df.shape}")
print("列名:")
print(df.columns.tolist())

# 2. 数据预处理：只保留数值型变量
numeric_df = df.select_dtypes(include=[np.number])
print(f"\n数值变量数量: {numeric_df.shape[1]}")
print(f"样本数量: {numeric_df.shape[0]}")

# 删除缺失值行
numeric_df = numeric_df.dropna()
print(f"清理后样本数: {numeric_df.shape[0]}")

# 删除方差为0的列
zero_var_cols = numeric_df.columns[numeric_df.var() == 0].tolist()
if zero_var_cols:
    print(f"\n删除零方差变量: {zero_var_cols}")
    numeric_df = numeric_df.drop(columns=zero_var_cols)

# 3. 标准化数据
print("\n正在标准化数据...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)
scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns, index=numeric_df.index)

# 4. 执行 PCA
print("\n=== 执行主成分分析（PCA） ===")
pca = PCA()
pca_result = pca.fit(scaled_df)

# 5. PCA 结果分析
eigenvalues = pca.explained_variance_
variance_percent = pca.explained_variance_ratio_ * 100
cumulative_variance = np.cumsum(variance_percent)

# 创建方差解释 DataFrame
variance_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(len(eigenvalues))],
    '特征值': eigenvalues,
    '方差解释率': variance_percent,
    '累计方差解释率': cumulative_variance
})

print("\n前10个主成分的方差解释:")
print(variance_df.head(10))

# 6. 确定保留的主成分数量
# 准则1：累计方差 ≥ 80%
n_components_80 = np.argmax(cumulative_variance >= 80) + 1
if n_components_80 == 0:  # 全部都不满足
    n_components_80 = len(eigenvalues)

# 准则2：Kaiser准则（特征值 > 1）
n_components_kaiser = np.sum(eigenvalues > 1)

# 建议保留数量（取较小值，但至少保留2个）
recommended_n = min(n_components_80, n_components_kaiser)
if recommended_n < 2:
    recommended_n = 2

print(f"\n建议保留的主成分数量: {recommended_n}")
print(f"可解释总方差的: {cumulative_variance[recommended_n - 1]:.1f}%")

# 7. 可视化
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False

# 7.1 碎石图 + 累计方差（双Y轴）
fig, ax1 = plt.subplots(figsize=(10, 6))
x_vals = np.arange(1, min(16, len(eigenvalues)+1))
ax1.bar(x_vals, variance_df.loc[:x_vals[-1]-1, '方差解释率'], color='steelblue', alpha=0.7, label='方差解释率')
ax1.set_xlabel('主成分')
ax1.set_ylabel('方差解释率 (%)')
ax1.set_title(f'PCA 碎石图（建议保留前 {recommended_n} 个主成分）')

ax2 = ax1.twinx()
ax2.plot(x_vals, cumulative_variance[x_vals-1] / 5, color='red', marker='o', label='累计方差解释率（右轴）')
ax2.set_ylabel('累计方差解释率 (%)')
ax2.set_ylim(0, 20)  # 因为除以5，对应0–100%

ax1.set_xticks(x_vals)
ax1.set_xticklabels([f'PC{i}' for i in x_vals], rotation=45)
plt.tight_layout()
plt.show()

# 7.2 累计方差图
plt.figure(figsize=(8, 5))
plt.plot(x_vals, cumulative_variance[x_vals-1], marker='o', color='blue')
plt.axhline(80, color='red', linestyle='--', label='80% 阈值')
plt.axvline(recommended_n, color='green', linestyle='--', label=f'建议数量: {recommended_n}')
plt.title('累计方差解释率')
plt.xlabel('主成分数量')
plt.ylabel('累计方差解释率 (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7.3 前两个主成分散点图
pca_scores = pca.transform(scaled_df)
pca_scores_df = pd.DataFrame(pca_scores[:, :2], columns=['PC1', 'PC2'], index=scaled_df.index)

# 尝试找一个分类变量用于着色
categorical_vars = df.select_dtypes(exclude=[np.number]).columns.tolist()
color_col = None
if categorical_vars:
    color_col = categorical_vars[0]
    if len(df.loc[scaled_df.index, color_col].dropna()) == len(pca_scores_df):
        pca_scores_df[color_col] = df.loc[scaled_df.index, color_col].values

plt.figure(figsize=(8, 6))
if color_col and color_col in pca_scores_df.columns:
    sns.scatterplot(data=pca_scores_df, x='PC1', y='PC2', hue=color_col, alpha=0.7, s=30)
else:
    plt.scatter(pca_scores_df['PC1'], pca_scores_df['PC2'], alpha=0.6, s=30, color='blue')
plt.title(f'PCA 样本分布\nPC1 ({variance_percent[0]:.1f}%) vs PC2 ({variance_percent[1]:.1f}%)')
plt.xlabel(f'PC1 ({variance_percent[0]:.1f}%)')
plt.ylabel(f'PC2 ({variance_percent[1]:.1f}%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7.4 变量载荷图（前10个贡献最大的变量）
loadings = pd.DataFrame(
    pca.components_.T[:, :2],
    columns=['PC1', 'PC2'],
    index=numeric_df.columns
)
loadings['贡献度'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)
loadings_top = loadings.nlargest(10, '贡献度')

plt.figure(figsize=(8, 8))
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
for idx, row in loadings_top.iterrows():
    plt.arrow(0, 0, row['PC1'], row['PC2'], head_width=0.02, head_length=0.02, fc='red', ec='red', alpha=0.6)
    plt.text(row['PC1']*1.1, row['PC2']*1.1, idx, fontsize=9, ha='center', va='center')
plt.xlim(loadings_top['PC1'].min()*1.2, loadings_top['PC1'].max()*1.2)
plt.ylim(loadings_top['PC2'].min()*1.2, loadings_top['PC2'].max()*1.2)
plt.title('变量载荷图（前10个重要变量）')
plt.xlabel('PC1 载荷')
plt.ylabel('PC2 载荷')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. 提取降维后的数据
print("\n=== 提取降维后的数据 ===")
pca_reduced = pca_scores[:, :recommended_n]
reduced_df = pd.DataFrame(pca_reduced, columns=[f'PC{i+1}' for i in range(recommended_n)], index=scaled_df.index)

# 添加回分类变量
if categorical_vars:
    reduced_df = pd.concat([reduced_df, df.loc[scaled_df.index, categorical_vars]], axis=1)

# 添加原始 peakVO2（不区分大小写）
peak_cols = [col for col in df.columns if 'peak' in col.lower()]
if peak_cols:
    peak_col = peak_cols[0]
    reduced_df['original_peakVO2'] = df.loc[scaled_df.index, peak_col]
    print("已添加原始 peakVO2 变量")

print(f"原始数据维度: {scaled_df.shape[1]} 个变量")
print(f"降维后维度: {recommended_n} 个主成分")

# 9. 保存结果
os.makedirs(output_dir, exist_ok=True)

# 降维数据
reduced_path = os.path.join(output_dir, "peakVO2_PCA_reduced.csv")
reduced_df.to_csv(reduced_path, index=False)
print(f"\n降维后的数据已保存到: {reduced_path}")

# PCA 载荷
loadings_output = pd.DataFrame(
    pca.components_.T[:, :recommended_n],
    columns=[f'PC{i+1}' for i in range(recommended_n)],
    index=numeric_df.columns
)
loadings_path = os.path.join(output_dir, "peakVO2_PCA_loadings.csv")
loadings_output.to_csv(loadings_path)
print(f"PCA 载荷已保存到: {loadings_path}")

# 方差解释
variance_path = os.path.join(output_dir, "peakVO2_PCA_variance.csv")
variance_df.to_csv(variance_path, index=False)
print(f"方差解释已保存到: {variance_path}")

# 10. 主成分解释性分析
print("\n=== 主成分解释性分析 ===")
for i in range(min(3, recommended_n)):
    pc_loadings = pca.components_[i]
    abs_loadings = np.abs(pc_loadings)
    top_indices = np.argsort(-abs_loadings)[:5]
    top_vars = numeric_df.columns[top_indices]
    top_vals = pc_loadings[top_indices]
    print(f"\nPC{i+1} (解释方差 {variance_percent[i]:.1f}%):")
    for var, val in zip(top_vars, top_vals):
        print(f"  {var}: {val:.3f}")

# 11. 结果总结
compression_rate = (1 - recommended_n / numeric_df.shape[1]) * 100
print("\n" + "="*60)
print("PCA 降维处理完成！")
print("="*60)
print("总结:")
print(f"✓ 原始变量数: {numeric_df.shape[1]}")
print(f"✓ 保留主成分数: {recommended_n}")
print(f"✓ 方差解释率: {cumulative_variance[recommended_n - 1]:.1f}%")
print(f"✓ 维度压缩率: {compression_rate:.1f}%")
print("\n输出文件:")
print(f"1. 降维数据: {reduced_path}")
print(f"2. PCA 载荷: {loadings_path}")
print(f"3. 方差解释: {variance_path}")
print("="*60)

# 12. 显示前几个样本的降维结果
print("\n降维后前5个样本的数据:")
print(reduced_df.head())