# -*- coding: UTF-8 -*-
'''
這段程式碼展示了硬邊界與軟邊界 SVM 的區別，並且強調了支持向量的概念。
硬邊界 SVM 會試圖將所有資料點正確分類，可能會導致過擬合；而軟邊界 SVM 則允許一些錯誤分類，以獲得更好的泛化能力。
這在實際應用中是非常重要的，因為資料通常會有雜訊或重疊的
'''

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定繪圖使用的字型為微軟正黑體
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# 產生可線性分類但含少量雜訊的資料（容易看到 margin 差異）
X, y = make_classification(n_samples=100,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_clusters_per_class=1,
                           class_sep=2.0,
                           flip_y=0.1,  # 加入 10% 標記錯誤（雜訊）
                           random_state=42)
y = 2 * y - 1  # 調整標籤為 {-1, +1}

# 建立兩個 SVM 模型
clf_hard = SVC(kernel='linear', C=1e6)
clf_soft = SVC(kernel='linear', C=0.1)

# 分別訓練
clf_hard.fit(X, y)
clf_soft.fit(X, y)

# 建立網格
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                     np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))

# 繪圖
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
titles = ['Hard Margin SVM（C=1e6）', 'Soft Margin SVM（C=0.1）']
clfs = [clf_hard, clf_soft]

for ax, clf, title in zip(axes, clfs, titles):
    # 計算決策邊界
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # 畫分類區域（背景）
    ax.contourf(xx, yy, Z > 0, alpha=0.3, cmap='bwr')
    # 畫邊界線與 margin 線（Z = ±1）
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')

    # 資料點
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    # 支持向量（綠色空心圈）
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=120, facecolors='none', edgecolors='green', linewidths=1.5,
               label='支持向量')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("特徵 1")
    ax.set_ylabel("特徵 2")
    ax.legend()
    ax.set_aspect('equal')

plt.suptitle("Hard Margin vs. Soft Margin SVM 分類邊界比較", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()