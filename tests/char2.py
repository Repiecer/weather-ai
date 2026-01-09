from sklearn.metrics import (confusion_matrix, classification_report, 
                           f1_score, roc_auc_score, accuracy_score)
import pandas as pd

# 创建示例数据
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred_proba = [0.2, 0.8, 0.4, 0.3, 0.9, 0.1, 0.7, 0.6, 0.55, 0.85, 0.25, 0.35, 0.75, 0.4]

print("=== 模型综合评估 ===\n")

# 1. 基础指标
accuracy = accuracy_score(y_true, y_pred)
print(f"准确率: {accuracy:.3f}")

# 2. 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("\n混淆矩阵:")
print(cm)

# 3. 详细分类报告
print("\n分类报告:")
print(classification_report(y_true, y_pred))

# 4. F1分数
f1 = f1_score(y_true, y_pred)
print(f"F1分数: {f1:.3f}")

# 5. ROC-AUC（需要概率预测）
auc = roc_auc_score(y_true, y_pred_proba)
print(f"ROC-AUC分数: {auc:.3f}")

# 6. 计算所有指标
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

metrics_df = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
}, index=['Class 0', 'Class 1'])

print("\n详细指标表格:")
print(metrics_df)