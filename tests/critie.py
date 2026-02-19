import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# 创建模拟数据
X, y = make_classification(n_samples=100, n_features=20, 
                           n_informative=5, n_redundant=5,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("=== 正则化对比实验 ===\n")

# 1. 带强正则化（简单模型，可能欠拟合）
model_strong_reg = LogisticRegression(penalty='l2', C=0.01, max_iter=1000)
model_strong_reg.fit(X_train, y_train)
train_acc1 = accuracy_score(y_train, model_strong_reg.predict(X_train))
test_acc1 = accuracy_score(y_test, model_strong_reg.predict(X_test))
print(f"强正则化(C=0.01): 训练集准确率={train_acc1:.3f}, 测试集准确率={test_acc1:.3f}")
print(f"非零系数数量: {(np.abs(model_strong_reg.coef_) > 1e-5).sum()}")

# 2. 适中正则化（默认，平衡）
model_default = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
model_default.fit(X_train, y_train)
train_acc2 = accuracy_score(y_train, model_default.predict(X_train))
test_acc2 = accuracy_score(y_test, model_default.predict(X_test))
print(f"\n默认正则化(C=1.0): 训练集准确率={train_acc2:.3f}, 测试集准确率={test_acc2:.3f}")
print(f"非零系数数量: {(np.abs(model_default.coef_) > 1e-5).sum()}")

# 3. 弱正则化（接近无正则化，可能过拟合）
model_weak_reg = LogisticRegression(penalty='l2', C=1000, max_iter=1000)
model_weak_reg.fit(X_train, y_train)
train_acc3 = accuracy_score(y_train, model_weak_reg.predict(X_train))
test_acc3 = accuracy_score(y_test, model_weak_reg.predict(X_test))
print(f"\n弱正则化(C=1000): 训练集准确率={train_acc3:.3f}, 测试集准确率={test_acc3:.3f}")
print(f"非零系数数量: {(np.abs(model_weak_reg.coef_) > 1e-5).sum()}")

# 4. 无正则化（理论上，实际用极大C近似）
model_no_reg = LogisticRegression(penalty='l2', C=1e10, max_iter=1000)
model_no_reg.fit(X_train, y_train)
train_acc4 = accuracy_score(y_train, model_no_reg.predict(X_train))
test_acc4 = accuracy_score(y_test, model_no_reg.predict(X_test))
print(f"\n无正则化(C=1e10): 训练集准确率={train_acc4:.3f}, 测试集准确率={test_acc4:.3f}")
print(f"系数绝对值之和: {np.abs(model_no_reg.coef_).sum():.2f} (通常很大)")