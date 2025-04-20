import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei') # 設定字型為微軟正黑體（可顯示中文）
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# ====== 定義可視化函式 plot_dbscan() ======
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    """
    用來繪製 DBSCAN 聚類結果的自定義視覺化函式

    參數說明：
    - dbscan：訓練好的 DBSCAN 模型物件
    - X：輸入的 2D 資料集
    - size：核心點畫出來的大小
    - show_xlabels, show_ylabels：是否顯示座標軸標籤
    """
    
    # 建立一個與 labels_ 等長的布林陣列，用來標記哪些是核心點
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True  # 核心樣本索引位置設為 True

    # 離群點標記（DBSCAN 會將離群點標記為 -1）
    anomalies_mask = dbscan.labels_ == -1

    # 非核心點（非核心 + 非離群 = 邊界點）
    non_core_mask = ~(core_mask | anomalies_mask)

    # 將三種資料類型切出來
    cores = dbscan.components_      # 所有核心點的座標
    anomalies = X[anomalies_mask]   # 離群點
    non_cores = X[non_core_mask]    # 邊界點

    # 畫核心點（大圓圈）
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    
    # 在核心點中畫小星星加強標記
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])

    # 畫離群點（紅色叉叉）
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)

    # 畫邊界點（小點）
    plt.scatter(non_cores[:, 0], non_cores[:, 1],
                c=dbscan.labels_[non_core_mask], marker=".")

    # 設定座標軸顯示
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')

    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')

    # 顯示 DBSCAN 使用的參數
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)



# ===== 主程式區塊（從這裡開始執行） =====
if __name__ == "__main__":

    # 產生「月牙型」的二維資料集（非線性資料）
    X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

    # 顯示資料分布（尚未分群）
    plt.plot(X[:, 0], X[:, 1], 'b.')  # 用藍點畫出所有資料
    plt.title("原始資料分佈")
    plt.show()

    # 建立兩個 DBSCAN 模型，使用不同的 ε（鄰域半徑）
    dbscan = DBSCAN(eps=0.05, min_samples=5)   # 半徑小，預期找出很多雜訊
    dbscan2 = DBSCAN(eps=0.2, min_samples=5)   # 半徑大，分群效果較穩定

    # 套用 DBSCAN 模型進行訓練
    dbscan.fit(X)
    dbscan2.fit(X)

    # 建立視覺化圖像窗口（共兩張圖）
    plt.figure(figsize=(9, 3.2))

    # 第一張圖：ε = 0.05 的分群結果（預期分群數多、雜訊多）
    plt.subplot(121)
    plot_dbscan(dbscan, X, size=100)

    # 第二張圖：ε = 0.2 的分群結果（預期穩定分成兩群）
    plt.subplot(122)
    plot_dbscan(dbscan2, X, size=600, show_ylabels=False)

    plt.suptitle("DBSCAN 不同 ε 參數下的分群效果", fontsize=16)
    plt.tight_layout()
    plt.show()