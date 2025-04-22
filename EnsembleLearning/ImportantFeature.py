# -*- coding: UTF-8 -*-
"""
此程式主要目的是：
1. 使用隨機森林（Random Forest）模型來評估特徵重要性（feature importance）。
2. 分別以 iris 資料集與 MNIST 手寫數字影像資料集為例，示範如何使用隨機森林找出關鍵特徵。
3. 最後將 MNIST 特徵重要性視覺化成一張 28x28 圖像，用以觀察哪些像素對分類最具影響力。
"""

import numpy as np
np.random.seed(42)  # 設定隨機種子，確保每次執行結果一致
import matplotlib
# matplotlib.use('Agg')  # 若在無圖形介面（如伺服器）執行，可開啟此行避免 GUI 錯誤
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定中文字型為微軟正黑體，支援圖形中文字顯示
plt.rcParams['axes.labelsize'] = 14  # 坐標標籤字體大小
plt.rcParams['xtick.labelsize'] = 12  # x軸刻度字體大小
plt.rcParams['ytick.labelsize'] = 12  # y軸刻度字體大小
import warnings
warnings.filterwarnings('ignore')  # 關閉警告訊息

# 匯入 Scikit-learn 模型與資料集模組
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris  # 花卉資料集
from sklearn.datasets import fetch_openml  # 可從 openml.org 載入 MNIST 等資料

# 定義一個繪圖函式來顯示 MNIST 的特徵重要性
def plot_digit(data):
    """
    將一維特徵資料（長度為 784）重塑為 28x28 的影像格式並顯示。

    輸入：
    - data: 一維 NumPy 陣列，長度需為 784，對應於 MNIST 的像素特徵重要性

    輸出：
    - 無回傳值，於目前圖表中以「熱力圖」方式呈現特徵重要性
    """
    image = data.reshape(28, 28)  # 轉換成 28x28 的影像
    plt.imshow(image, cmap=matplotlib.cm.hot)  # 使用熱力圖配色顯示重要程度
    plt.axis('off')  # 不顯示軸線


if __name__ == "__main__":

    # --- Part 1: 使用 iris 資料集示範特徵重要性 ---

    iris = load_iris()  # 載入花卉資料集，共 150 筆、4 個特徵
    rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)  # 建立 500 棵樹的隨機森林
    rf_clf.fit(iris['data'], iris['target'])  # 進行模型訓練

    # 印出每個特徵的名稱與其對分類的重要性
    for name, score in zip(iris['feature_names'], rf_clf.feature_importances_):
        print(name, score)

    # --- Part 2: 使用 MNIST 資料集示範像素層級的重要性視覺化 ---

    # 載入 MNIST 手寫數字資料集（含 70000 筆 28x28 灰階圖像）
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    # 使用隨機森林訓練整個 MNIST 資料集（注意：這會比較花時間）
    rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rf_clf.fit(mnist['data'], mnist['target'])

    # 印出總共幾個特徵（應為 784）
    print(f"總共有多少特徵：{rf_clf.feature_importances_.shape[0]}")

    # --- 特徵重要性視覺化 ---

    plot_digit(rf_clf.feature_importances_)  # 將 feature_importances 當作圖片顯示

    # 加入 colorbar 解釋特徵重要性（最小值為不重要，最大值為很重要）
    char = plt.colorbar(ticks=[
        rf_clf.feature_importances_.min(),
        rf_clf.feature_importances_.max()
    ])
    char.ax.set_yticklabels(['Not important', 'Very important'])

    plt.title('Feature Importances') 
    plt.show() 
