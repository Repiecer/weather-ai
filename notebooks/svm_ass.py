import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# 创建可线性分离的数据
X, y = make_blobs(n_samples=100, centers=2, 
                  random_state=42, cluster_std=1.5)

# 训练SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# 可视化
plt.figure(figsize=(10, 6))

# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=50, alpha=0.6)

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格来评估模型
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

# 绘制决策边界和间隔
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], 
           alpha=0.5, linestyles=['--', '-', '--'])

# 标记支持向量
ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], 
           s=100, linewidth=1, facecolors='none', edgecolors='r')

plt.title('SVM分类器: 决策边界和支持向量')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.grid(True, alpha=0.3)
plt.show()

print(f"支持向量数量: {len(svm.support_vectors_)}")
print(f"支持向量索引: {svm.support_}")