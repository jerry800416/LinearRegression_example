# -*- coding: utf-8 -*-
"""
這段程式主要功能：
1. 建立一個具有 3 個隱藏狀態（代表盒子）、2 個觀察狀態（紅球、白球）的 HMM 模型。
2. 使用已知參數執行維特比演算法，輸出最可能的隱藏狀態序列。
3. 使用 EM（Baum-Welch）演算法根據多筆觀察序列估計模型參數。
"""

import numpy as np
from hmmlearn import hmm

# 隱藏狀態名稱（3個盒子）
states = ["box 1", "box 2", "box 3"]
n_states = len(states)  # 狀態數量 = 3

# 觀察狀態名稱（2種球）
observations = ["red", "white"]
n_observations = len(observations)  # 觀察符號數 = 2

# =========================
# HMM 模型參數設定
# =========================

# 初始狀態機率（π）
start_probability = np.array([0.2, 0.4, 0.4])

# 狀態轉移機率矩陣（A）
transition_probability = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

# 發射機率矩陣（B）：每個狀態下觀察 red 和 white 的機率
emission_probability = np.array([
    [0.5, 0.5],  # box 1
    [0.4, 0.6],  # box 2
    [0.7, 0.3]   # box 3
])

# =========================
# 使用已知參數建構 HMM 模型，並執行維特比演算法
# =========================

# 建立 Multinomial HMM 模型（離散觀察值）
# model = hmm.MultinomialHMM(n_components=n_states) # 若版本低於0.3.0，則使用此行
model = hmm.CategoricalHMM(n_components=n_states)
# 設定模型的參數
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# ➤ 維特比解碼
# 定義觀察序列 O = [紅, 白, 紅]，其中紅=0、白=1
seen = np.array([[0, 1, 0]]).T  # 觀察序列長度為3，需轉置成列向量

# 使用維特比演算法找出最可能的隱藏狀態序列
# 輸入：觀察序列 seen
# 輸出：log 機率、最佳狀態序列 box
logprob, box = model.decode(seen, algorithm="viterbi")
print("維特比推論得到的隱藏狀態序列:", np.array(states)[box])

# ➤ 使用 predict() 方法驗證是否一致（預設也是維特比）
box2 = model.predict(seen)
print("predict() 得到的狀態序列:", np.array(states)[box2])

# ➤ 模型對觀察序列的總機率（對數機率）
print("log 機率（P(O|λ) 的 log）:", model.score(seen))

# =========================
# 使用 EM 演算法重新估計模型參數（Baum-Welch）
# =========================

# 建立一個新模型，啟用 EM 訓練，設定最大迭代次數與收斂條件
# model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
model2 = hmm.CategoricalHMM(n_components=n_states, n_iter=20, tol=0.01)
# 訓練資料：3 組長度為 4 的觀察序列（數值型，每個值為紅=0、白=1）
X2 = np.array([[0, 1, 0, 1],    # 紅白紅白
               [0, 0, 0, 1],    # 紅紅紅白
               [1, 0, 1, 1]])   # 白紅白白

# 使用 Baum-Welch（EM）對模型進行參數學習
# 輸入：觀察序列集合 X2
# 輸出：模型參數會自動更新
model2.fit(X2)

# 印出 EM 訓練後學到的參數
print("EM 訓練後的起始機率 π:")
print(model2.startprob_)
print('-----------------------------------------')

print("EM 訓練後的狀態轉移矩陣 A:")
print(model2.transmat_)
print('-----------------------------------------')

print("EM 訓練後的發射機率矩陣 B:")
print(model2.emissionprob_)
print('-----------------------------------------')

# 使用訓練後的模型計算 log-likelihood 分數
print("模型對 X2 的 log 機率（P(X2|λ)）:", model2.score(X2))
