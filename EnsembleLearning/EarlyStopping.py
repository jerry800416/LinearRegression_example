# -*- coding: utf-8 -*-
"""
此程式主要功能：
展示 Gradient Boosting Regressor（GBRT）在非線性資料上的應用，
包括兩種 Early Stopping 策略：
1. 使用 `staged_predict()` 對驗證集進行逐步預測，找出最佳樹數（最佳模型）
2. 使用 `warm_start=True` 動態增長模型，若驗證誤差連續上升五次則停止訓練
同時也視覺化驗證誤差曲線與最終最佳模型之預測結果
"""

import numpy as np
np.random.seed(42)  # 設定隨機種子，確保結果可重現

import matplotlib
# matplotlib.use('Agg')  # 若在無 GUI 圖形介面下執行，可開啟此行以避免錯誤
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定中文字型以支援中文顯示
plt.rcParams['axes.unicode_minus'] = False  # 設定負號可以正確顯示
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

import warnings
warnings.filterwarnings('ignore')  # 關閉警告訊息

# 匯入必要模組
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    """
    繪製回歸預測圖形（模型預測線與原始資料點）
    
    輸入：
    - regressors: list，一或多個已訓練好的回歸器
    - X: 原始特徵資料（形狀為 [n_samples, 1]）
    - y: 原始目標值（形狀為 [n_samples]）
    - axes: 圖形繪製範圍 [x_min, x_max, y_min, y_max]
    - label: 預測曲線的圖例文字
    - style: 預測曲線的樣式（例如 'r-' 表示紅色實線）
    - data_style: 原始資料點的樣式（例如 'b.' 表示藍色點）
    - data_label: 原始資料點的圖例文字
    
    輸出：
    - 將模型預測結果與資料點畫在圖上
    """
    x1 = np.linspace(axes[0], axes[1], 500)  # 建立等距預測座標點
    y_pred = sum(reg.predict(x1.reshape(-1, 1)) for reg in regressors)  # 合併所有回歸器預測值
    plt.plot(X[:, 0], y, data_style, label=data_label)  # 畫出原始資料點
    plt.plot(x1, y_pred, style, linewidth=2, label=label)  # 畫出預測線
    if label or data_label:
        plt.legend(loc="upper center", fontsize=14)
    plt.axis(axes)  # 設定繪圖座標軸範圍


if __name__ == "__main__":

    # --- 建立一維隨機資料 ---
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0]**2 + 0.05 * np.random.randn(100)

    # 分割訓練集與驗證集
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

    # 建立 Gradient Boosting 模型，固定最多使用 120 棵樹
    gbrt = GradientBoostingRegressor(
        max_depth=2,         # 每棵樹的最大深度為 2
        n_estimators=120,    # 最多建構 120 棵樹
        random_state=42
    )
    gbrt.fit(X_train, y_train)  # 訓練模型

    # 使用 staged_predict 產生逐步預測（每棵樹加上去後都預測一次）
    errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]

    # 找出驗證誤差最小所對應的棵樹數
    bst_n_estimators = np.argmin(errors)

    # 建立最佳樹數的模型（early stopping 結果）
    gbrt_best = GradientBoostingRegressor(
        max_depth=2,
        n_estimators=bst_n_estimators,
        random_state=42
    )
    gbrt_best.fit(X_train, y_train)  # 重新訓練最佳模型

    min_error = np.min(errors)  # 最小驗證誤差

    # --- 視覺化驗證誤差曲線與最佳模型預測結果 ---
    plt.figure(figsize=(11, 4))

    # 左圖：每棵樹累加後在驗證集的 MSE
    plt.subplot(121)
    plt.plot(errors, 'b.-')  # 畫出每階段錯誤
    plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], 'k--')  # 垂直線：最佳棵數
    plt.plot([0, 120], [min_error, min_error], 'k--')  # 水平線：最小誤差
    plt.axis([0, 120, 0, 0.01])
    plt.title('驗證誤差（Validation Error）')

    # 右圖：最佳模型的預測結果（含資料點）
    plt.subplot(122)
    plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title('最佳模型預測（共 %d 棵樹）' % bst_n_estimators)
    plt.show()


    # --- Warm Start 方式實現 Early Stopping ---
    gbrt = GradientBoostingRegressor(
        max_depth=2,
        random_state=42,
        warm_start=True  # 允許模型在已訓練狀態下繼續增長
    )

    error_going_up = 0         # 追蹤驗證誤差是否連續上升
    min_val_error = float('inf')  # 初始最小誤差設為無限大

    # 迭代從 1 棵樹開始逐步增加，直到驗證誤差連續 5 次上升為止
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)

        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0  # 若誤差下降，重設計數器
        else:
            error_going_up += 1
            if error_going_up == 5:
                break  # 若誤差連續上升 5 次，則停止訓練

    # 印出訓練停止時的總樹數
    print("Warm Start 模型提早停止時的樹數：", gbrt.n_estimators)
