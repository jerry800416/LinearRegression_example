# -*- coding: UTF-8 -*-
"""
這隻程式的主要作用是：
比較「單棵決策樹」、「Bagging 模型（含 OOB）」與「隨機森林」在非線性資料集（make_moons）上的分類效果與決策邊界差異。

功能包含：
- 產生非線性二分類資料集 make_moons
- 建立並訓練 Decision Tree、BaggingClassifier（含 OOB）與 Random Forest
- 計算準確率與 OOB 分數
- 視覺化四種模型的決策邊界（Decision Boundary）

此程式使用 matplotlib 繪圖，若於非 notebook 環境執行可能需手動切換後端。
"""

import numpy as np
np.random.seed(42)  # 設定隨機種子以確保實驗可重現
import os
import matplotlib
# matplotlib.use('Agg')  # 若在無圖形介面環境中執行，請取消註解以避免 GUI 衝突
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定繪圖使用的字型為微軟正黑體
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')  # 關閉警告訊息

# scikit-learn 的常用模組
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    """
    繪製分類模型的決策邊界

    輸入參數：
    - clf: 已訓練的分類器，如 DecisionTreeClassifier、BaggingClassifier、RandomForestClassifier
    - X: 特徵資料（形狀為 [n_samples, 2]）
    - y: 標籤資料（形狀為 [n_samples]）
    - axes: 決策邊界的顯示範圍 [x_min, x_max, y_min, y_max]
    - alpha: 背景填色透明度
    - contour: 是否畫出邊界線輪廓

    輸出：
    - 於當前 matplotlib 圖形繪製模型分類區域與原始資料分佈
    """
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]  # 組成測試用的網格座標
    y_pred = clf.predict(X_new).reshape(x1.shape)  # 對整個座標空間預測分類

    # 分類區域背景顏色（根據預測結果）
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, cmap=custom_cmap, alpha=0.3)

    # 若需要畫出邊界線輪廓
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)

    # 原始訓練資料點
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'yo', alpha=0.6)
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'bs', alpha=0.6)

    # 圖形設置
    plt.axis(axes)
    plt.xlabel('x1')
    plt.xlabel('x2')


if __name__ == "__main__":

    # 生成帶雜訊的非線性半月形資料
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 訓練單棵決策樹
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    print(f"單棵決策樹的準確率：{accuracy_score(y_test, y_pred_tree)}")

    # 建立 Bagging 模型（使用 500 棵隨機抽樣的決策樹）
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=500,
        max_samples=100,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
    print(f"Bagging 模型的準確率：{accuracy_score(y_test, y_pred)}")

    # Bagging + OOB 模型：使用未被抽樣進每棵樹的樣本做驗證（Out-of-Bag）
    bag_clf_oob = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=500,
        max_samples=100,
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        oob_score=True
    )
    bag_clf_oob.fit(X_train, y_train)
    y_pred_oob = bag_clf_oob.predict(X_test)
    print(f"Bagging 模型的 OOB 準確率：{bag_clf_oob.oob_score_}")

    # 隨機森林模型（也是一種特殊的 Bagging）
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    print(f"隨機森林的準確率：{accuracy_score(y_test, y_pred_rf)}")

    # 四種模型的決策邊界視覺化
    plt.figure(figsize=(12, 5))
    plt.subplot(141)
    plot_decision_boundary(tree_clf, X, y)
    plt.title('Decision Tree')
    plt.subplot(142)
    plot_decision_boundary(bag_clf, X, y)
    plt.title('Bagging')
    plt.subplot(143)
    plot_decision_boundary(bag_clf_oob, X, y)
    plt.title('Bagging (OOB)')
    plt.subplot(144)
    plot_decision_boundary(rf_clf, X, y)
    plt.title('Random Forest')
    plt.suptitle('Bagging vs Forest')
    plt.show()
