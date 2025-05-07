# -*- coding: utf-8 -*-
"""
📘 本程式說明：
這段程式碼示範如何使用 PCA（主成分分析）來降維 iris 資料集，
包含資料標準化、計算協方差矩陣、特徵值分解、主成分選擇與降維視覺化。
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定中文字型為微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 設定負號可正常顯示
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ===========================================
# Step 1. 讀取 iris 資料集並設定欄位名稱
# ===========================================

df = pd.read_csv('PCA&LDA/iris.data')
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
print("這是前五筆資料:\n", df.head())

# ===========================================
# Step 2. 分離特徵與標籤
# X: 特徵資料，y: 類別標籤
# ===========================================

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# 標籤與特徵對應詞典（供圖示與說明用）
label_dict = {1: 'Iris-Setosa', 2: 'Iris-Versicolor', 3: 'Iris-Virginica'}
feature_dict = {
    0: 'sepal length [cm]',
    1: 'sepal width [cm]',
    2: 'petal length [cm]',
    3: 'petal width [cm]'
}

# ===========================================
# Step 3. 顯示每個特徵欄位的直方圖（依類別）
# ===========================================

plt.figure(figsize=(8, 6))
for cnt in range(4):
    plt.subplot(2, 2, cnt + 1)
    for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
        plt.hist(X[y == lab, cnt],
                 label=lab,
                 bins=10,
                 alpha=0.3)
    plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)
plt.tight_layout()
plt.show()

# ===========================================
# Step 4. 特徵標準化（Z-score）
# ===========================================

X_std = StandardScaler().fit_transform(X)

# ===========================================
# Step 5. 計算協方差矩陣（手動與 NumPy）
# ===========================================

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot(X_std - mean_vec) / (X_std.shape[0] - 1)
print("手動計算的協方差矩陣:\n", cov_mat)

cov_np = np.cov(X_std.T)
print("NumPy 計算的協方差矩陣:\n", cov_np)

# ===========================================
# Step 6. 特徵值與特徵向量計算
# ===========================================

eig_vals, eig_vecs = np.linalg.eig(cov_np)
print("特徵向量（每欄對應一個特徵向量）:\n", eig_vecs)
print("特徵值:\n", eig_vals)

# ===========================================
# Step 7. 特徵值與特徵向量配對並排序
# ===========================================

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

print("特徵值排序（由大到小）:")
for val, vec in eig_pairs:
    print("特徵值:", val)

# ===========================================
# Step 8. 解釋變異量與累積變異量計算
# ===========================================

tot = sum(eig_vals)
var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

print("每個主成分解釋的變異百分比:\n", var_exp)
print("累積解釋變異百分比:\n", cum_var_exp)

# ===========================================
# Step 9. 視覺化：變異量與累積變異
# ===========================================

plt.figure(figsize=(6, 4))
plt.bar(range(4), var_exp, alpha=0.5, align='center',
        label='單一主成分變異量比例')
plt.step(range(4), cum_var_exp, where='mid',
         label='累積解釋變異量比例')
plt.ylabel('解釋變異百分比')
plt.xlabel('主成分')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# ===========================================
# Step 10. 建立投影矩陣 W（選前2個主成分）
# ===========================================

matrix_w = np.hstack((
    eig_pairs[0][1].reshape(4, 1),
    eig_pairs[1][1].reshape(4, 1)
))
print("前兩個主成分組成的投影矩陣 W:\n", matrix_w)

# ===========================================
# Step 11. 將標準化資料投影到主成分空間
# ===========================================

Y = X_std.dot(matrix_w)
print("投影後的資料 Y（在主成分空間的表示）:\n", Y)

# ===========================================
# Step 12. 原始特徵空間中前兩維的散佈圖
# ===========================================

plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                    ('blue', 'red', 'green')):
    plt.scatter(X[y == lab, 0],
                X[y == lab, 1],
                label=lab,
                c=col)
plt.xlabel('sepal_len')
plt.ylabel('sepal_wid')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# ===========================================
# Step 13. PCA 降維後（2 維主成分）的資料視覺化
# ===========================================

plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                    ('blue', 'red', 'green')):
    plt.scatter(Y[y == lab, 0],
                Y[y == lab, 1],
                label=lab,
                c=col)
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()
