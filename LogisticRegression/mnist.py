import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei') # 設定字型為微軟正黑體（可顯示中文）
import matplotlib.image as mpimg
import math


from logistic_regression import LogisticRegression  # 引入自定義的邏輯回歸類別（支援多分類與特徵處理）

# 讀取手寫數字資料（MNIST子集）
data = pd.read_csv('./LogisticRegression/data/mnist-demo.csv')

# ========== 顯示前 25 張圖片 ==========

numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))  # 決定顯示的圖像格數
plt.figure(figsize=(10, 10))  # 設定圖形大小

for plot_index in range(numbers_to_display):
    digit = data[plot_index:plot_index + 1].values      # 取出一筆資料
    digit_label = digit[0][0]                            # 標籤為資料的第 0 欄
    digit_pixels = digit[0][1:]                          # 圖像像素從第 1 欄開始

    image_size = int(math.sqrt(digit_pixels.shape[0]))   # MNIST 圖像為 28x28（784維）

    frame = digit_pixels.reshape((image_size, image_size))  # 將向量轉回二維矩陣
    plt.subplot(num_cells, num_cells, plot_index + 1)       # 建立子圖
    plt.imshow(frame, cmap='Greys')                         # 使用灰階顯示
    plt.title(digit_label)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

plt.subplots_adjust(hspace=0.5, wspace=0.5)  # 子圖間距調整
plt.show()

# ========== 分割訓練集與測試集（80%訓練，20%測試） ==========
pd_train_data = data.sample(frac=0.8, random_state=42)
pd_test_data = data.drop(pd_train_data.index)

train_data = pd_train_data.values
test_data = pd_test_data.values

# ========== 建立訓練資料與標籤 ==========
num_training_examples = 6000
x_train = train_data[:num_training_examples, 1:]   # 所有訓練樣本的圖像像素（去掉標籤欄）
y_train = train_data[:num_training_examples, [0]]  # 對應的標籤

x_test = test_data[:, 1:]  # 測試資料（像素）
y_test = test_data[:, [0]]  # 測試資料（標籤）

# ========== 訓練參數設定 ==========
max_iterations = 10000       # 最大迭代次數
polynomial_degree = 0        # 不進行多項式特徵擴展
sinusoid_degree = 0          # 不使用正弦特徵
normalize_data = True        # 正規化資料（對於影像像素建議開啟）

# ========== 初始化與訓練邏輯回歸模型 ==========
logistic_regression = LogisticRegression(
    x_train,
    y_train,
    polynomial_degree,
    sinusoid_degree,
    normalize_data
)

# 開始訓練模型，回傳每個數字（類別）對應的 θ 向量與損失曲線
thetas, costs = logistic_regression.train(max_iterations)

# ========== 顯示每個數字對應的參數權重圖像 ==========
numbers_to_display = 9
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10, 10))

# 將 θ（每一個數字的分類器參數）轉回圖像形式，可視為類別特徵圖
for plot_index in range(numbers_to_display):
    digit_pixels = thetas[plot_index][1:]  # 去除偏置項（第 0 維）

    image_size = int(math.sqrt(digit_pixels.shape[0]))  # 計算圖像大小
    frame = digit_pixels.reshape((image_size, image_size))

    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap='Greys')  # θ 權重圖以灰階呈現
    plt.title(plot_index)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

# ========== 損失函數收斂曲線繪圖 ==========
labels = logistic_regression.unique_labels
for index, label in enumerate(labels):
    plt.plot(range(len(costs[index])), costs[index], label=labels[index])

plt.xlabel('Gradient Steps')
plt.ylabel('Cost')
plt.title('每個數字的損失收斂曲線')
plt.legend()
plt.show()

# ========== 評估模型準確率（訓練集與測試集） ==========
y_train_predictions = logistic_regression.predict(x_train)
y_test_predictions = logistic_regression.predict(x_test)

train_precision = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
test_precision = np.sum(y_test_predictions == y_test) / y_test.shape[0] * 100

print('Training Precision: {:5.4f}%'.format(train_precision))
print('Test Precision: {:5.4f}%'.format(test_precision))

# ========== 顯示測試集中預測結果的圖像 ==========
numbers_to_display = 64
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(15, 15))

# 顯示預測結果，正確為綠色，錯誤為紅色
for plot_index in range(numbers_to_display):
    digit_label = y_test[plot_index, 0]         # 真實標籤
    digit_pixels = x_test[plot_index, :]        # 像素資料
    predicted_label = y_test_predictions[plot_index][0]  # 預測標籤

    image_size = int(math.sqrt(digit_pixels.shape[0]))
    frame = digit_pixels.reshape((image_size, image_size))

    color_map = 'Greens' if predicted_label == digit_label else 'Reds'

    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap=color_map)
    plt.title(predicted_label)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
