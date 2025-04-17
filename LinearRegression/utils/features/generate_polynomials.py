"""Add polynomial features to the features set
   此模組負責產生多項式特徵，用於擴充原有的特徵組合。
   例：對於兩個原始特徵 x1 與 x2，會生成 x1, x2, x1^2, x1*x2, x2^2 等組合特徵，
   以便用於非線性模型的建模。
"""

import numpy as np                           # 匯入 numpy 作數值運算、矩陣操作
from .normalize import normalize             # 從同一目錄下引入 normalize 函式，用於資料正規化

def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """
    依據指定的多項式次數，為輸入的資料生成多項式特徵
    
    參數:
    - dataset: 輸入的資料矩陣，形狀為 (樣本數, 特徵數)
    - polynomial_degree: 所要生成的最高多項式次數，例如 2 表示產生一階與二階特徵
    - normalize_data: 布林值，若為 True，則在產生多項式特徵後對結果進行正規化處理
    
    回傳:
    - polynomials: 經過多項式變換（以及可能的正規化）後產生的特徵矩陣
      此矩陣的每一列對應一個樣本，每一行代表一個生成的多項式特徵
    
    變換方法示例:
       若原始特徵包含 x1, x2，則會產生以下組合：
       x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3, ... 等
    """

    # 將原始資料沿著特徵方向（axis=1）平均分成兩部分
    # 此處假設資料中至少包含兩個特徵，或希望以兩組特徵進行多項式組合
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]    # 取第一部分的特徵
    dataset_2 = features_split[1]    # 取第二部分的特徵

    # 取得 dataset_1 與 dataset_2 的形狀：(樣本數, 特徵數)
    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    # 驗證兩部分樣本數是否一致，若不一致則無法對應生成多項式特徵
    if num_examples_1 != num_examples_2:
        raise ValueError('Can not generate polynomials for two sets with different number of rows')

    # 若兩部分皆無特徵，則無法生成多項式特徵，拋出錯誤
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Can not generate polynomials for two sets with no columns')

    # 若其中一部分無特徵，則將該部分替換為另一部分，以確保兩者皆有數據用於組合
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    # 取兩部分較小的特徵數量作為基準，避免因兩部分特徵數目不一而出現問題
    # 此處 num_features 代表將取前 num_features 列的資料進行多項式組合
    num_features = num_features_1 if num_features_1 < num_features_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    # 初始化一個空的矩陣，準備儲存生成的多項式特徵，行數等於樣本數，列數初始為 0
    polynomials = np.empty((num_examples_1, 0))

    # 利用兩層迴圈依次生成多項式特徵：
    # 外層 for 迴圈 i 代表多項式的總次數，從 1 到 polynomial_degree（包含）；
    # 內層 for 迴圈 j 用於分配次數到 dataset_2，其餘次數則分配給 dataset_1，
    # 使得 i = (次數分配給 dataset_1) + (次數分配給 dataset_2)。
    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            # 計算多項式特徵：dataset_1^(i - j) * dataset_2^(j)
            # 這裡可以生成如：x1^i (j=0)，x1^(i-1)*x2 (j=1)，...，x2^i (j=i)
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            # 將新生成的 polynomial_feature 與目前的 polynomials 矩陣進行水平連接（axis=1）
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    # 如果 normalize_data 為 True，則對生成的多項式特徵進行正規化處理
    # normalize 函式通常回傳一個包含正規化後資料與其他統計值的元組，此處只取第一個元素作為正規化後的矩陣
    if normalize_data:
        polynomials = normalize(polynomials)[0]

    # 回傳最終生成的多項式特徵矩陣
    return polynomials
