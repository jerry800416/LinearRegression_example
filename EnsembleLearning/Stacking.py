# -*- coding: utf-8 -*-
"""
此程式展示了一種簡易的集成學習方法：**Stacking（堆疊集成）**。
使用 4 種基本分類器（隨機森林、極端樹、線性 SVM、多層感知機），
先各自對 MNIST 驗證集預測，然後將這些預測作為新特徵輸入另一個隨機森林（稱為 Blender）來進行最終分類。
特點：使用驗證集預測作為次階段模型訓練資料，並透過 OOB 檢驗集成器表現。
"""

# 匯入必要的函式與套件
import numpy as np
np.random.seed(42)  # 固定隨機種子，確保可重現性

from sklearn.datasets import fetch_openml  # 從 openml 載入 MNIST 資料集
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

# 載入 MNIST 資料集（784 維像素特徵，目標為 0~9）
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# 將資料拆分為訓練集 + 驗證集（50000）與測試集（10000）
X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)

# 從訓練集再切出 10000 筆作為驗證集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)

# 定義四種基礎分類器（弱學習器）
random_forest_clf = RandomForestClassifier(random_state=42)         # 隨機森林
extra_trees_clf = ExtraTreesClassifier(random_state=42)             # 極端隨機樹（更激進的隨機性）
svm_clf = LinearSVC(random_state=42)                                 # 線性支援向量機
mlp_clf = MLPClassifier(random_state=42)                             # 多層感知機（神經網路）

# 將所有基本模型收集到一個列表中
estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]

# 訓練所有基礎分類器（每個模型都使用相同的訓練集）
for estimator in estimators:
    print("Training the", estimator)  # 印出目前正在訓練的模型
    estimator.fit(X_train, y_train)

# 初始化一個矩陣來儲存每個模型對驗證集的預測結果
# 形狀為 [驗證集樣本數, 模型數量] => 每一欄為一個模型的預測
X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)

# 對驗證集進行預測，並將每個模型的預測結果存到對應的欄位中
for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

# 建立最終的 Blender 模型（使用隨機森林，並啟用 OOB 檢驗）
rnd_forest_blender = RandomForestClassifier(
    n_estimators=200,        # 訓練 200 棵樹
    oob_score=True,          # 啟用 OOB 評估
    random_state=42
)

# 將「驗證集上各模型的預測結果」作為輸入特徵，對照真實標籤訓練 Blender
rnd_forest_blender.fit(X_val_predictions, y_val)

# 印出 Blender 的 OOB 分數，代表集成器在驗證集預測表現
print("Blender OOB 分數：", rnd_forest_blender.oob_score_)
