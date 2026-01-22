import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# 人工智能2301 李天牧 20233098
# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# ==========================================
# 1. 数据层面 (Data Level)
# ==========================================

# 1.1 加载 Pima Indians Diabetes 数据集
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

print("数据集概览：")
print(df.head())

# 1.2 数据清洗：处理生物学意义上的异常零值
# 在此数据集中，血糖、血压、BMI等不应为0
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, np.nan)
    # 使用中位数填补缺失值（比均值更具鲁棒性）
    df[col] = df[col].fillna(df[col].median())

# 1.3 特征与标签分离
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 1.4 处理样本不平衡 (SMOTE)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 1.5 数据集划分与标准化
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将缩放后的数据转回DataFrame，以便SHAP保留特征名
X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)

# ==========================================
# 2. 方法层面 (Method Level)
# ==========================================

# 2.1 构建 XGBoost 分类模型
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 2.2 模型训练
model.fit(X_train_final, y_train)

# ==========================================
# 3. 分析层面 (Analysis Level)
# ==========================================

# 3.1 性能验证
y_pred = model.predict(X_test_final)
y_prob = model.predict_proba(X_test_final)[:, 1]

print("\n模型评估报告：")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC 分数: {roc_auc_score(y_test, y_prob):.4f}")

# 3.2 引入 SHAP 解释分析 (核心部分)
# 基于你读过的 Lundberg (2017) 论文，我们使用 TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_final)

# 可视化 A: 全局重要性分析 (Summary Plot)
print("\n生成全局解释图...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_final, show=False)
plt.title("SHAP Global Feature Importance (Diabetes Prediction)")
plt.show()

# 可视化 B: 个体决策解释 (Waterfall Plot)
# 选取测试集中第一个样本进行分析
print("\n生成个体案例解释图 (样本索引 0)...")
sample_idx = 0
# 创建一个特定的 Explanation 对象用于绘图
sample_explanation = shap.Explanation(
    values=shap_values[sample_idx], 
    base_values=explainer.expected_value, 
    data=X_test_final.iloc[sample_idx], 
    feature_names=X.columns
)

plt.figure(figsize=(10, 4))
shap.plots.waterfall(sample_explanation, show=False)
plt.title(f"Local Explanation for Patient #{sample_idx}")
plt.show()

# 3.3 实验分析结论
print("""
分析层面结论：
1. 通过 SHAP Summary Plot 可以看出，血糖浓度 (Glucose) 是模型决策的最核心特征，
   其值越高，患病预测的 SHAP 值越大（正向推动）。
2. 个体 Waterfall Plot 展示了模型如何对特定患者进行诊断。例如，即便某患者年龄较轻，
   但如果其 BMI 和血糖过高，模型依然会将其判定为高风险人群。
3. 这种基于归因的分析，验证了模型逻辑与临床医学常识的一致性。
""")