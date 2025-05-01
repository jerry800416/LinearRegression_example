# -*- coding: utf-8 -*-
"""
此程式用於展示集成學習中兩種經典方法：
1. AdaBoost（二分類）：使用 DecisionTree 作為弱分類器，觀察分類邊界與準確率
2. Gradient Boosting（迴歸）：使用三棵回歸樹依序逼近殘差，並視覺化整體預測過程
"""

import numpy as np
np.random.seed(42)  # 設定隨機種子，確保每次執行結果一致
import matplotlib
# matplotlib.use('Agg')  # 若在無圖形介面環境執行，可取消註解避免錯誤
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定中文字型
plt.rcParams['axes.unicode_minus'] = False  # 設定負號可以正確顯示
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score,mean_squared_error
from matplotlib.colors import ListedColormap
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    """
    繪製二維分類模型的決策邊界
    輸入：
    - clf: 分類器
    - X, y: 特徵與標籤資料
    - axes: 顯示邊界範圍 [x_min, x_max, y_min, y_max]
    - alpha: 背景透明度
    - contour: 是否畫邊界線
    """
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff'])
    plt.contourf(x1, x2, y_pred, cmap=custom_cmap, alpha=0.3)
    if contour:
        plt.contour(x1, x2, y_pred, colors='k', linewidths=0.5)
    plt.plot(X[y==0, 0], X[y==0, 1], 'yo', alpha=0.6)
    plt.plot(X[y==1, 0], X[y==1, 1], 'bs', alpha=0.6)
    plt.axis(axes)
    plt.xlabel('特徵 x1')
    plt.ylabel('特徵 x2')


def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    """
    繪製回歸模型的預測結果與資料點
    輸入：
    - regressors: 一個或多個回歸器
    - X, y: 原始資料
    - axes: [x_min, x_max, y_min, y_max]
    - label, style: 預測線的圖例與樣式
    - data_style, data_label: 原始資料點的樣式與圖例
    """
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(reg.predict(x1.reshape(-1, 1)) for reg in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=14)
    plt.axis(axes)


# --- AdaBoost 二分類 ---
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)
print(f"AdaBoost 準確率：{accuracy_score(y_test, y_pred)}")
plt.figure()
plot_decision_boundary(ada_clf, X, y)
plt.title('AdaBoost 分類邊界')
plt.show()

# --- Gradient Boosting 迴歸器 ---
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0]**2 + 0.05 * np.random.randn(100)

# 手動建立三棵樹逐步擬合殘差
reg1 = DecisionTreeRegressor(max_depth=2)
reg1.fit(X, y)
y2 = y - reg1.predict(X)
reg2 = DecisionTreeRegressor(max_depth=2)
reg2.fit(X, y2)
y3 = y2 - reg2.predict(X)
reg3 = DecisionTreeRegressor(max_depth=2)
reg3.fit(X, y3)

# 預測值
X_new = np.array([[0.8]])
pred = sum(tree.predict(X_new) for tree in (reg1, reg2, reg3))
print(f"手動 Gradient Boosting 預測值：{pred[0]:.4f}")

plt.figure(figsize=(11, 11))
plt.subplot(321)
plot_predictions([reg1], X, y, [-0.5, 0.5, -0.1, 0.8], label="第1棵樹預測", style="g-", data_label="訓練資料")
plt.title("殘差與第1棵樹預測")
plt.subplot(322)
plot_predictions([reg1], X, y, [-0.5, 0.5, -0.1, 0.8], label="第1棵樹", data_label="訓練資料")
plt.title("集成預測 (1 棵樹)")
plt.subplot(323)
plot_predictions([reg2], X, y2, [-0.5, 0.5, -0.5, 0.5], label="第2棵樹", style="g-", data_style="k+", data_label="殘差")
plt.subplot(324)
plot_predictions([reg1, reg2], X, y, [-0.5, 0.5, -0.1, 0.8], label="第1+2棵樹")
plt.title("集成預測 (2 棵樹)")
plt.subplot(325)
plot_predictions([reg3], X, y3, [-0.5, 0.5, -0.5, 0.5], label="第3棵樹", style="g-", data_style="k+")
plt.xlabel("x1")
plt.subplot(326)
plot_predictions([reg1, reg2, reg3], X, y, [-0.5, 0.5, -0.1, 0.8], label="第1+2+3棵樹")
plt.xlabel("x1")
plt.suptitle("Gradient Boosting 回歸視覺化流程", fontsize=16)
plt.tight_layout()
plt.show()


X_train,X_val,y_train,y_val = train_test_split(X,y,random_state=49)
gbrt = GradientBoostingRegressor(max_depth = 2,
                          n_estimators = 120,
                          random_state = 42
)
gbrt.fit(X_train,y_train)

errors = [mean_squared_error(y_val,y_pred) for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth = 2,
                          n_estimators = bst_n_estimators,
                          random_state = 42
)
gbrt_best.fit(X_train,y_train)
min_error = np.min(errors)
min_error
plt.figure(figsize = (11,4))
plt.subplot(121)
plt.plot(errors,'b.-')
plt.plot([bst_n_estimators,bst_n_estimators],[0,min_error],'k--')
plt.plot([0,120],[min_error,min_error],'k--')
plt.axis([0,120,0,0.01])
plt.title('Val Error')
plt.subplot(122)
plot_predictions([gbrt_best],X,y,axes=[-0.5,0.5,-0.1,0.8])
plt.title('Best Model(%d trees)'%bst_n_estimators)
plt.show()

gbrt = GradientBoostingRegressor(max_depth = 2,
                             random_state = 42,
                                 warm_start =True
)
error_going_up = 0
min_val_error = float('inf')

for n_estimators in range(1,120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train,y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val,y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up +=1
        if error_going_up == 5:
            break
    
print (gbrt.n_estimators)