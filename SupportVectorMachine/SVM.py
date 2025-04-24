import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定繪圖使用的字型為微軟正黑體
from matplotlib.widgets import Slider
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# 建立資料集
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
y = 2 * y - 1  # 轉成 {-1, +1}

# 建立網格
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                     np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))

# 初始 C 值
initial_C = 1.0

# 建立初始模型
clf = SVC(kernel='linear', C=initial_C)
clf.fit(X, y)

# 初始圖形繪製
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # 空出下方放 Slider

# 計算決策函數
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 畫分類區域與邊界
contourf = ax.contourf(xx, yy, Z > 0, alpha=0.3, cmap='bwr')
contour = ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'])

# 畫資料點與支持向量
scatter_pts = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
sv = ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=100, edgecolors='green', facecolors='none', linewidths=1.5, label='支持向量')

ax.set_title(f"SVM 分類（C = {initial_C:.2f}）")
ax.set_xlabel("特徵 1")
ax.set_ylabel("特徵 2")
ax.legend()

# 新增滑動條（位置：[left, bottom, width, height]）
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'C 值', valmin=0.01, valmax=1000.0, valinit=initial_C, valstep=0.01)

# 定義更新函數
def update(val):
    C = slider.val
    clf = SVC(kernel='linear', C=C)
    clf.fit(X, y)

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # 更新圖形內容
    for coll in contour.collections:
        coll.remove()
    new_contour = ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'])

    # 更新分類區域
    for coll in contourf.collections:
        coll.remove()
    new_contourf = ax.contourf(xx, yy, Z > 0, alpha=0.3, cmap='bwr')

    # 更新支持向量
    sv.set_offsets(clf.support_vectors_)

    ax.set_title(f"SVM 分類（C = {C:.2f}）")
    fig.canvas.draw_idle()

# 連結滑動條與更新函數
slider.on_changed(update)

plt.show()
