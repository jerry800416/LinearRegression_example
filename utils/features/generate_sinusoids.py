import numpy as np  # 匯入 NumPy 套件，用於數值運算與矩陣操作


def generate_sinusoids(dataset, sinusoid_degree):
    """
    生成正弦函數特徵

    此函數根據輸入的資料集 (dataset)，對每個數據應用多個正弦函數轉換，
    生成額外的特徵以豐富原始特徵，進而幫助模型捕捉非線性關係。
    
    具體操作:
    - 對 dataset 中的每個元素，分別計算 sin(1*x), sin(2*x), ... , sin(sinusoid_degree*x)
      其中 x 為 dataset 中的原始數據，degree 代表正弦函數中的倍數，從 1 到 sinusoid_degree。
    
    參數:
    - dataset: 原始輸入資料，通常為 NumPy 陣列，形狀為 (樣本數, 特徵數)。
      每個數據點的每個特徵都會被應用正弦轉換。
    - sinusoid_degree: 整數，指定生成的正弦函數特徵的個數（正弦函數的倍數）。
      例如，sinusoid_degree = 3 則會生成三個特徵：sin(1*x), sin(2*x) 與 sin(3*x)。
    
    回傳:
    - sinusoids: 生成的正弦函數特徵矩陣，形狀為 (樣本數, 原始特徵數 × sinusoid_degree)，
      即將原始特徵依照設定倍數生成對應的正弦特徵後合併成新矩陣。
    """

    # 取得輸入資料集中的樣本數（即矩陣的行數）
    num_examples = dataset.shape[0]

    # 初始化一個空矩陣，用於儲存生成的正弦特徵
    # np.empty((num_examples, 0)) 表示矩陣有 num_examples 行，但初始時沒有任何列
    sinusoids = np.empty((num_examples, 0))

    # 利用 for 迴圈，從 1 到 sinusoid_degree 逐一計算不同倍數的正弦特徵
    for degree in range(1, sinusoid_degree + 1):
        # 對 dataset 中的每個元素，乘以當前 degree 後再取正弦值
        # 例如，若 degree 為 2，則計算 sin(2 * dataset)
        sinusoid_features = np.sin(degree * dataset)
        
        # 將此次計算的正弦特徵與先前已計算的正弦特徵矩陣進行水平合併（沿 axis=1）
        # 這樣最終 sinusoids 矩陣中會包含所有 degree 從 1 到 sinusoid_degree 的正弦特徵
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)
        
    # 回傳生成好的正弦函數特徵矩陣
    return sinusoids
