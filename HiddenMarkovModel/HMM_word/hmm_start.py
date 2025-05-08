# -*- coding: utf-8 -*-
"""
這段程式的主要功能是：
使用從資料中計算得出的 HMM 參數（初始機率、轉移機率、發射機率），
建立一個 `MultinomialHMM` 模型，並針對輸入的一段中文詞句（如：「我要吃饭谢天谢地」），
預測對應的隱藏狀態序列（BMES），實現中文分詞的解碼操作。
"""

import numpy as np
import warnings
from hmmlearn.hmm import CategoricalHMM as mhmm
import get_hmm_param as pa  # 外部模組，提供 startprob、transmat、emissionprob 等函數

warnings.filterwarnings("ignore")  # 忽略 hmmlearn 的警告訊息

# ============================
# 取得 HMM 所需參數
# ============================

# 初始狀態機率向量（π），順序為 B、M、E、S
startprob = np.array(pa.get_startprob())
print("這是初始狀態機率 startprob:", startprob)

# 狀態轉移機率矩陣（A），4x4 對應 B、M、E、S 各狀態間的轉移
transmat = np.array(pa.get_transmat())
print("這是狀態轉移機率矩陣 transmat:\n", transmat)

# 發射機率矩陣（B），4xN 對應各狀態產生各字元的機率
emissionprob = np.array(pa.get_emissionprob())
print("這是發射機率矩陣 emissionprob (未歸一化):\n", emissionprob)
# 👉 修正：對每一列進行歸一化，讓每一列機率總和為 1
row_sums = emissionprob.sum(axis=1, keepdims=True)
emissionprob = emissionprob / row_sums  # 對每一列做 normalize
print("這是修正後（每列總和為 1）的 emissionprob:\n", emissionprob)
# ============================
# 建立 HMM 模型
# ============================

# 建立一個有 4 個隱藏狀態的 MultinomialHMM 模型（對應 B、M、E、S）
mul_hmm = mhmm(n_components=4)

# 設定模型的初始機率、轉移機率、發射機率
mul_hmm.startprob_ = startprob
mul_hmm.transmat_ = transmat
mul_hmm.emissionprob_ = emissionprob

# ============================
# 測試輸入詞句並進行分詞解碼
# ============================

# 測試詞句：phase（例如：「我要吃飯謝天謝地」）
phase = u"我要吃飯謝天謝地"

# 將中文詞句轉換為對應的觀察符號索引序列
# 輸入：字串，輸出：對應的字元索引 list[int]
X = np.array(pa.get_array_from_phase(phase))

# reshape 為 (T, 1)，hmmlearn 的 MultinomialHMM 需要這種格式
X = X.reshape(len(phase), 1)
print("這是轉換後的觀察序列 X:\n", X)

# 使用 HMM 預測輸入字串的隱藏狀態序列（BMES）
Y = mul_hmm.predict(X)
print("這是預測出來的隱藏狀態序列 Y:\n", Y)

# 狀態對照表：B（詞首）、M（詞中）、E（詞尾）、S（單字詞） → 對應索引 {0,1,2,3}