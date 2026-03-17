import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer  # <<< 新增：用于处理缺失值
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = "/test1/peakVO2.csv"
data = pd.read_csv(file_path)

print("=== 数据基本信息 ===")
print(f"数据形状: {data.shape}")
print("列名:", list(data.columns))
print("\n" + "="*50)
print("1. 分类任务：逻辑回归（预测低 LVEF）")
print("="*50)

# 查找 LVEF 列（不区分大小写）
lvef_col = None
for col in data.columns:
    if 'lvef' in col.lower():
        lvef_col = col
        break
if lvef_col is None:
    raise ValueError("未找到包含 'lvef' 的列，请检查数据")

# 清洗数据：去除 LVEF 缺失值
df_class = data.dropna(subset=[lvef_col]).copy()
df_class['low_lvef'] = (df_class[lvef_col] < 50).astype(int)

print(f"使用 LVEF 列: {lvef_col}")
print("低 LVEF 样本分布:", df_class['low_lvef'].value_counts())

# 选择数值型特征
numeric_cols = df_class.select_dtypes(include=[np.number]).columns.tolist()
predictors = [col for col in numeric_cols if col not in [lvef_col, 'low_lvef']]

# 计算与 LVEF 的相关性，选 top 3
corrs = df_class[predictors].corrwith(df_class[lvef_col]).abs().sort_values(ascending=False)
top3 = corrs.head(3).index.tolist()
print(f"选择的预测变量（相关性最高）: {top3}")

# 准备 X 和 y
X = df_class[top3]
y = df_class['low_lvef']

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=456, stratify=y
)

# ===== 新增：缺失值填充 + 标准化 =====
imputer = SimpleImputer(strategy='mean')  # 用均值填充
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
# ===================================

# 训练模型
log_reg = LogisticRegression(max_iter=1000, random_state=456)
log_reg.fit(X_train_scaled, y_train)

# 预测
y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# 评估
print("=== 模型评估 ===")
print(classification_report(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.3f}")

# ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC 曲线（LVEF 分类）')
plt.legend()
plt.grid(True)
plt.show()

# 特征重要性（优势比）
coef = log_reg.coef_[0]
odds_ratios = np.exp(coef)
importance_df = pd.DataFrame({'变量': top3, '系数': coef, '优势比 (OR)': odds_ratios})
print("=== 特征重要性（优势比）===")
print(importance_df.round(3))

# ==============================================================================
print("\n" + "="*50)
print("2. 回归任务：线性回归（预测 LVEF 值）")
print("="*50)

df_reg = data.dropna(subset=[lvef_col]).copy()

# 选择数值特征
numeric_cols = df_reg.select_dtypes(include=[np.number]).columns.tolist()
predictors = [col for col in numeric_cols if col != lvef_col]

# 相关性选 top 5
corrs = df_reg[predictors].corrwith(df_reg[lvef_col]).abs().sort_values(ascending=False)
top5 = corrs.head(5).index.tolist()
print(f"选择的预测变量: {top5}")

X = df_reg[top5]
y = df_reg[lvef_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=456)

# ===== 新增：缺失值填充（回归任务）=====
imputer_reg = SimpleImputer(strategy='mean')
X_train_imputed = imputer_reg.fit_transform(X_train)
X_test_imputed = imputer_reg.transform(X_test)
# =======================================

# 拟合线性回归（使用 statsmodels 获取详细统计）
X_train_sm = sm.add_constant(X_train_imputed)
ols_model = sm.OLS(y_train, X_train_sm).fit()
print("=== 线性回归结果摘要 ===")
print(ols_model.summary())

# 预测
X_test_sm = sm.add_constant(X_test_imputed)
y_pred_reg = ols_model.predict(X_test_sm)

# 性能指标
mse = mean_squared_error(y_test, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)
print(f"=== 回归性能 ===")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")

# 可视化：实际 vs 预测
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred_reg, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际 LVEF')
plt.ylabel('预测 LVEF')
plt.title(f'LVEF 预测效果 (R² = {r2:.3f})')
plt.grid(True)
plt.show()

# ==============================================================================
print("\n" + "="*50)
print("3. 聚类任务：K-means")
print("="*50)

# 提取数值列并去缺失（聚类必须无缺失）
numeric_data = data.select_dtypes(include=[np.number])
numeric_data_clean = numeric_data.dropna()  # 聚类任务直接删除含 NaN 的行

print(f"用于聚类的样本数: {numeric_data_clean.shape[0]}")
print(f"变量数: {numeric_data_clean.shape[1]}")

# 标准化
scaler_cluster = StandardScaler()
scaled_features = scaler_cluster.fit_transform(numeric_data_clean)

# 确定最佳 K：肘部法 + 轮廓系数
inertias = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=25, random_state=456)
    kmeans.fit(scaled_features)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(scaled_features, kmeans.labels_))

# 绘图
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(range(2, 11), inertias, 'bo-')
ax[0].set_title('肘部法则 (Inertia)')
ax[0].set_xlabel('K')
ax[0].set_ylabel('Inertia')
ax[1].plot(range(2, 11), sil_scores, 'go-')
ax[1].set_title('轮廓系数')
ax[1].set_xlabel('K')
ax[1].set_ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

best_k = K_range[np.argmax(sil_scores)]
print(f"选择最佳聚类数 K = {best_k}（轮廓系数最高）")

# 执行 K-means
kmeans_final = KMeans(n_clusters=best_k, n_init=25, random_state=456)
clusters = kmeans_final.fit_predict(scaled_features)

# PCA 可视化
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
df_pca['cluster'] = clusters

# 若有 LVEF，用其作为点大小
if lvef_col in data.columns:
    original_index = numeric_data_clean.index
    df_pca['lvef'] = data.loc[original_index, lvef_col].values
    sizes = df_pca['lvef']
else:
    sizes = 50

plt.figure(figsize=(7, 5))
scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['cluster'], cmap='tab10', s=sizes, alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title(f'K-means 聚类 (K={best_k}) —— 点大小表示 LVEF')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
plt.grid(True)
plt.show()

# 聚类中心热图
centers = kmeans_final.cluster_centers_
centers_df = pd.DataFrame(centers, columns=numeric_data_clean.columns)
centers_df.index.name = 'cluster'
plt.figure(figsize=(10, 4))
sns.heatmap(centers_df.T, cmap='coolwarm', center=0, annot=False)
plt.title('聚类中心（标准化后）')
plt.xlabel('聚类编号')
plt.ylabel('变量')
plt.show()

# 各聚类特征摘要
print("=== 各聚类特征摘要（与总体均值对比）===")
overall_mean = numeric_data_clean.mean()
for i in range(best_k):
    cluster_mask = (clusters == i)
    cluster_mean = numeric_data_clean.iloc[cluster_mask].mean()
    diff = (cluster_mean - overall_mean).abs().sort_values(ascending=False)
    top_vars = diff.head(3).index.tolist()
    print(f"聚类 {i} (n={cluster_mask.sum()}):")
    for var in top_vars:
        print(f"  {var}: 聚类均值={cluster_mean[var]:.2f}, 总体={overall_mean[var]:.2f}")

# 保存带聚类标签的结果
result_df = data.loc[numeric_data_clean.index].copy()
result_df['cluster'] = clusters
result_df.to_csv("E:\Study\python\WorkSpace\PyTorch\lvef_clusters_python.csv", index=False)
print("聚类结果已保存至 'lvef_clusters_python.csv'")

# 聚类质量
avg_sil = max(sil_scores)
print(f"=== 聚类质量 ===")
print(f"平均轮廓宽度: {avg_sil:.3f} → {'合理结构' if avg_sil > 0.5 else '弱/无结构'}")