# -*- coding: utf-8 -*-
"""
這段程式主要用於建立基於 BMES（Begin, Middle, End, Single）標註的隱馬可夫模型（HMM）初始機率、轉移機率與發射機率矩陣。
資料來源為一組已標註 BMES 的字詞結構，用於訓練分詞模型或詞性標註模型。
"""

from data import data  # 外部提供的資料格式應為：每筆為 {"詞": "BMES標籤序列"}
import json
import logging

def prints(s):
    """列印除錯資訊（可在上線時禁用）"""
    print(s)

# ======================
# 初始機率：P(state_0)
# ======================
def get_startprob():
    """
    功能：計算 HMM 模型中初始狀態的機率分佈
    輸出：list[float]，順序為 [P(B), P(M), P(E), P(S)]
    """
    c = 0
    c_map = {"B": 0, "M": 0, "E": 0, "S": 0}
    
    for v in data:
        for key in v:
            value = v[key]
        c += 1
        prints("第1個狀態是: " + value[0])
        c_map[value[0]] += 1
        prints("目前各狀態出現次數統計: " + str(c_map))
    
    res = []
    for i in "BMES":
        res.append(c_map[i] / float(c))  # 機率 = 該狀態次數 / 總樣本數
    return res

# ======================
# 轉移機率：P(state_t | state_t-1)
# ======================
def get_transmat():
    """
    功能：計算 HMM 中的狀態轉移機率矩陣
    輸出：list[list[float]]，大小為 4x4 對應 BMES → BMES
    """
    c = 0
    c_map = {}  # 記錄狀態對轉移次數，例如 BM、BE、MM 等
    
    for v in data:
        for key in v:
            value = v[key]
        
        for v_i in range(len(value) - 1):
            couple = value[v_i:v_i + 2]  # e.g., "BM"
            c_map[couple] = c_map.get(couple, 0) + 1
            c += 1
    
    prints("所有狀態轉移次數統計（c_map）為: " + str(c_map))
    
    res = []
    for i in "BMES":
        col = []
        col_count = sum(c_map.get(i + j, 0) for j in "BMES")
        
        for j in "BMES":
            prob = c_map.get(i + j, 0) / float(col_count) if col_count != 0 else 0.0
            col.append(prob)
        res.append(col)
    return res

# ======================
# 字集合與索引對應
# ======================
def get_words():
    """
    功能：提供所有可見的觀察字元（觀察空間）
    輸出：str，組合的字串
    """
    return u"我要吃飯天氣不錯謝天地"

def get_word_map():
    """
    功能：為每個觀察字元建立對應索引
    輸出：dict{字: index}
    """
    words = get_words()
    res = {words[i]: i for i in range(len(words))}
    return res

def get_array_from_phase(phase):
    """
    功能：將輸入的詞（字串）轉換成對應的索引陣列
    輸入：phase (str)
    輸出：list[int]，對應觀察字元的 index 序列
    """
    word_map = get_word_map()
    return [word_map[key] for key in phase]

# ======================
# 發射機率：P(obs_t | state_t)
# ======================
def get_emissionprob():
    """
    功能：計算 HMM 中每個隱藏狀態對應發射到某個觀察符號的機率
    輸出：list[list[float]]，大小為 4xN，對應 BMES 各狀態對應每個字的機率
    """
    c = 0
    c_map = {}  # 統計例如 "B我", "S谢", "E天" 出現次數
    
    for v in data:
        for key in v:
            word = key
            value = v[key]
        
        for v_i in range(len(value)):
            couple = value[v_i] + word[v_i]
            prints("當前發射對（狀態+字）為: " + couple)
            c_map[couple] = c_map.get(couple, 0) + 1
            c += 1
    
    prints("所有發射次數統計（c_map）為: " + str(c_map))
    
    words = get_words()
    res = []
    for i in "BMES":
        col = []
        for j in words:
            prob = c_map.get(i + j, 0) / float(c) if c != 0 else 0.0
            col.append(prob)
        res.append(col)
    return res




if __name__ == "__main__":

    prints("這是初始機率向量 startprob:\n" + str(get_startprob()))
    print("這是轉移機率矩陣 transmat:\n", get_transmat())
    prints("這是發射機率矩陣 emissionprob:\n" + str(get_emissionprob()))
    prints("這是字元對應索引表 word_map:\n" + str(get_word_map()))
