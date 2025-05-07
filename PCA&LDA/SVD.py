# -*- coding: utf-8 -*-
"""
📘 本程式說明：
這是一個使用 PCA（主成分分析）結合 SVD（奇異值分解）來示範降維與資料重建的簡單範例。
資料是手動定義的 2 維 × 5 筆資料矩陣，程式會：
1. 對資料做 SVD 分解。
2. 取前 1 個主成分，將資料降維至 1 維。
3. 再根據該主成分還原回原本的 2 維空間。
4. 繪製原始資料與還原資料點的對應關係。

🧪 目的：
- 幫助理解 SVD 在 PCA 中的作用。
- 顯示資料投影與還原的幾何視覺差異。
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定字型為微軟正黑體（支援中文顯示）
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號（例如 -1）

# ==============================
# Step 1. 建立原始資料矩陣（已中心化）
# ==============================
# 每欄為一筆樣本，總共 5 筆樣本，每筆有 2 個特徵
X = np.array([
    [-1, -1, 0, 2, 0],
    [-2,  0, 0, 1, 1]
])

# ==============================
# Step 2. 對資料進行 SVD 分解
# ==============================
# 這裡我們使用 np.linalg.svd 來取得 U、S、VT 三個矩陣
# U: 左奇異向量（表示投影後的方向）
# S: 奇異值（可視為特徵值平方根）
# VT: 右奇異向量（表示主成分方向）
U, S, VT = np.linalg.svd(X, full_matrices=False)

# ==============================
# Step 3. 取前 K 個主成分（K=1）
# ==============================
K = 1
U_k = U[:, :K]             # 取前 K 個主成分方向（2×1）
S_k = np.diag(S[:K])       # 取前 K 個奇異值並形成對角矩陣（1×1）
VT_k = VT[:K, :]           # 取前 K 個右奇異向量（1×5）

# ==============================
# Step 4. 資料投影（降維處理）
# ==============================
# 將資料投影到主成分軸（1 維表示）
# 結果是每個樣本在主成分方向上的值
X_proj = np.dot(U_k, S_k)  # X_proj shape: (2,1)

# ==============================
# Step 5. 從降維後的資料還原回原空間
# ==============================
# 利用 X_proj × VT_k 還原到原始 2 維空間
X_recon = np.dot(U_k, np.dot(S_k, VT_k))  # shape: (2,5)

# ==============================
# Step 6. 資料視覺化：原始 vs 還原資料
# ==============================
# 圖中藍點為原始資料，紅點為降維還原後資料
# 中間用虛線連接，顯示還原誤差方向

plt.figure(figsize=(8, 6))

# 原始資料點（藍色）
plt.scatter(X[0], X[1], color='blue', label='原始資料點')

# 還原資料點（紅色）
plt.scatter(X_recon[0], X_recon[1], color='red', label='降維再還原資料點')

# 每個樣本畫一條虛線，連接原始點與還原點
for i in range(X.shape[1]):
    plt.plot(
        [X[0, i], X_recon[0, i]],
        [X[1, i], X_recon[1, i]],
        'k--',
        linewidth=0.5
    )

plt.xlabel('X軸')
plt.ylabel('Y軸')
plt.title('PCA via SVD：降維後還原視覺化')
plt.legend()
plt.grid(True)
plt.axis('equal')  # 保持 x、y 比例一致
plt.tight_layout()
plt.show()
