"""Prepares the dataset for training
   此模組主要負責將原始資料進行預處理，以便能夠進行後續的模型訓練。預處理包含資料正規化、特徵擴充（正弦與多項式轉換），以及最後加入偏置項（全1列）。
"""

import numpy as np  # 匯入 NumPy 套件，用於矩陣與數值運算
from .normalize import normalize  # 匯入 normalize 函式，用於數據標準化
from .generate_sinusoids import generate_sinusoids  # 匯入產生正弦函數特徵的函式
from .generate_polynomials import generate_polynomials  # 匯入產生多項式特徵的函式


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """
    將輸入資料做前處理操作，包含以下步驟：
     1. 依需求對資料進行正規化（Normalization）。
     2. 根據設定的 sinusoid_degree 與 polynomial_degree，擴充正弦與多項式特徵。
     3. 最後在資料矩陣最前方加入一列全 1（對應模型中的偏置項）。

    參數:
    - data: 原始輸入資料，應為 NumPy 陣列，每一行代表一筆樣本，每一列代表一個特徵。
    - polynomial_degree: 多項式轉換的最高次數，若設定為 0 表示不進行多項式特徵擴充。
    - sinusoid_degree: 正弦函數轉換的次數，若設定為 0 表示不進行正弦特徵擴充。
    - normalize_data: 布林值，若為 True 則對資料執行正規化處理，否則保留原始資料。

    回傳值:
    - data_processed: 預處理後的資料矩陣，包含原始特徵、（擴充後的）正弦、多項式特徵，並在最前方加上一列全1作為偏置項。
    - features_mean: 原始資料各特徵的平均值（若有做正規化，否則為 0）。
    - features_deviation: 原始資料各特徵的標準差（若有做正規化，否則為 0）。
    """

    # 計算樣本總數: 從資料的第 0 維（列數）取得樣本數
    num_examples = data.shape[0]

    # 複製原始資料以免影響後續處理，data_processed 變數將承載預處理後的資料
    data_processed = np.copy(data)

    # 初始化與正規化相關的變數：
    features_mean = 0         # 用來儲存各特徵的平均值
    features_deviation = 0    # 用來儲存各特徵的標準差
    data_normalized = data_processed  # 初始時預設未正規化的資料即為原始資料

    # 若設定需要正規化，則呼叫 normalize() 函式進行標準化處理
    if normalize_data:
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)
        
        # 更新 data_processed 為正規化後的資料
        data_processed = data_normalized

    # ------------------------------
    # 特徵轉換：正弦函數擴充
    # ------------------------------
    # 若 sinusoid_degree 大於 0，表示希望增加正弦函數特徵
    if sinusoid_degree > 0:
        # 使用正弦函數轉換，傳入正規化後的資料與指定的正弦轉換次數
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        # 將原有資料與新增的正弦特徵沿水平方向（axis=1）合併
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # ------------------------------
    # 特徵轉換：多項式擴充
    # ------------------------------
    # 若 polynomial_degree 大於 0，表示希望擴充多項式特徵（例如 x^2, x^3, ...）
    if polynomial_degree > 0:
        # 呼叫多項式特徵生成函式，傳入正規化後的資料與多項式次數（normalize_data 參數也傳入，以便根據需要調整）
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        # 將多項式特徵與現有資料合併，仍然沿水平方向做連接
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # ------------------------------
    # 加入偏置項：在資料矩陣的最前方加上一列全1
    # ------------------------------
    # 為了在線性模型中包含截距項（偏置），在每筆資料前都加入一個恆定為 1 的特徵
    # np.ones((num_examples, 1)) 建立一個形狀為 (樣本數, 1) 的全1矩陣，
    # 然後利用 np.hstack() 以水平方式將此矩陣與 data_processed 合併
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    # 回傳預處理後的資料與特徵統計資訊
    return data_processed, features_mean, features_deviation
