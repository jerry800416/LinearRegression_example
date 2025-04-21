import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei') # 設定字型為微軟正黑體（可顯示中文）

# 匯入自定義的 KMeans 類別（你之前定義好的演算法）
from k_means import KMeans

# 載入 Iris 資料集（CSV 檔案）
data = pd.read_csv('./Kmeans/data/iris.csv')

# 指定三種類別名稱（對應標籤）
iris_types = ['SETOSA','VERSICOLOR','VIRGINICA']

# 指定要用來視覺化的兩個特徵欄位
x_axis = 'petal_length'
y_axis = 'petal_width'

# 建立圖形窗口（分成兩個子圖）
plt.figure(figsize=(12,5))

# 子圖1：真實標籤的分布情況（原始類別）
plt.subplot(1,2,1)
for iris_type in iris_types:
    # 篩選對應類別的資料點並畫散點圖
    plt.scatter(
        data[x_axis][data['class'] == iris_type],
        data[y_axis][data['class'] == iris_type],
        label=iris_type
    )
plt.title('label known')  # 標題：已知標籤
plt.legend()

# 子圖2：不知道標籤的樣貌（只看分布）
plt.subplot(1,2,2)
plt.scatter(data[x_axis][:], data[y_axis][:])  # 全部資料畫出來
plt.title('label unknown')  # 標題：未知標籤
plt.show()


# 取得資料總筆數
num_examples = data.shape[0]

# 從 DataFrame 擷取訓練用的兩個特徵欄位（轉成 numpy 陣列）
x_train = data[[x_axis, y_axis]].values.reshape(num_examples, 2)

# 指定要分的群數（K 值）與最大疊代次數
num_clusters = 3
max_iteritions = 50

# 建立並訓練 KMeans 模型
k_means = KMeans(x_train, num_clusters)
centroids, closest_centroids_ids = k_means.train(max_iteritions)

# ========== 分群結果對比視覺化 ==========
plt.figure(figsize=(12,5))

# 子圖1：真實標籤視覺化（與前面相同）
plt.subplot(1,2,1)
for iris_type in iris_types:
    plt.scatter(
        data[x_axis][data['class'] == iris_type],
        data[y_axis][data['class'] == iris_type],
        label=iris_type
    )
plt.title('label known')  # 顯示真實標籤
plt.legend()

# 子圖2：KMeans 分群結果視覺化
plt.subplot(1,2,2)
for centroid_id, centroid in enumerate(centroids):
    # 找出所有屬於當前群中心的資料點（依分群結果）
    current_examples_index = (closest_centroids_ids == centroid_id).flatten()

    # 將這些點畫出來，每個群組不同顏色
    plt.scatter(
        data[x_axis][current_examples_index],
        data[y_axis][current_examples_index],
        label=centroid_id
    )

# 畫出中心點位置（用黑色叉叉標記）
for centroid_id, centroid in enumerate(centroids):
    plt.scatter(
        centroid[0], centroid[1],
        c='black', marker='x'
    )

plt.legend()
plt.title('label kmeans')  # 顯示 K-means 分群結果
plt.show()
