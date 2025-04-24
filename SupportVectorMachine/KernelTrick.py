# -*- coding: UTF-8 -*-
'''
這段程式碼展示了 SVM 使用不同核函數的分類效果，並且可以幫助理解核函數的作用與特性。 
透過這些圖形，我們可以觀察到不同核函數對於資料的分類邊界是如何影響的。
例如，線性核函數適合線性可分的資料，而 RBF 核函數則能夠處理更複雜的非線性資料。
多項式核函數則可以調整多項式的次數來適應不同的資料分佈。
Sigmoid 核函數則類似於神經網路中的激活函數，適合某些特定的應用場景。
這些圖形可以幫助我們更好地理解 SVM 的工作原理，並選擇適合的核函數來解決特定的分類問題。
'''

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定繪圖使用的字型為微軟正黑體
from sklearn.svm import SVC
from sklearn.datasets import make_moons

# 建立非線性可分的資料集（如月牙形）
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 設定要比較的四種核函數
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
titles = [
    '線性核函數（Linear）',
    '多項式核函數（Polynomial, degree=3）',
    'RBF 高斯核函數（γ=1.0）',
    'Sigmoid 核函數（tanh 型）'
]

# 建立網格點座標，用來畫決策邊界
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                     np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))

# 設定畫布大小
plt.figure(figsize=(12, 6))
plt.suptitle("SVM 四種核函數分類邊界比較", fontsize=16)

# 繪製四個子圖
for i, kernel in enumerate(kernels, 1):
    # 根據不同核函數初始化模型
    if kernel == 'poly':
        clf = SVC(kernel=kernel, degree=3, coef0=1, C=1.0)
    elif kernel == 'sigmoid':
        clf = SVC(kernel=kernel, coef0=0.5, C=1.0)
    else:
        clf = SVC(kernel=kernel, gamma=1.0, C=1.0)

    # 訓練模型
    clf.fit(X, y)

    # 預測網格點上的分類決策值
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # 畫子圖（2x2 格局）
    plt.subplot(2, 2, i)
    # 畫出分類區域（Z > 0 一邊為紅色，一邊為藍色）
    plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap='bwr')
    # 畫出決策邊界與 margin 線
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'])

    # 畫出資料點
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title(titles[i - 1], fontsize=13)
    plt.xlabel("特徵 1", fontsize=11)
    plt.ylabel("特徵 2", fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().set_aspect('equal')

# 自動調整子圖間距，保留標題空間
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


