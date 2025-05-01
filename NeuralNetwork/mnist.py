# -*- coding: UTF-8 -*-
'''
本程式碼目的：
利用自定義的 MultilayerPerceptron 類別，對 MNIST 手寫數字資料集進行分類任務，
包含資料載入、資料視覺化、模型訓練、測試預測與可視化預測結果
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定字體為微軟正黑體（支援中文顯示）
plt.rcParams['axes.unicode_minus'] = False  # 設定負號可以正確顯示
import math
from multilayer_perceptron import MultilayerPerceptron  # 匯入自定義的多層感知器類別

# 載入 MNIST 資料（CSV 格式）
data = pd.read_csv('./NeuralNetwork/data/mnist-demo.csv')

# =============================
# Step 1: 顯示前 25 筆手寫數字圖像
# =============================
numbers_to_display = 25  # 要顯示的筆數
num_cells = math.ceil(math.sqrt(numbers_to_display))  # 計算要建立的網格大小
plt.figure(figsize=(10, 10))
for plot_index in range(numbers_to_display):
    digit = data[plot_index:plot_index+1].values  # 取出一筆資料
    digit_label = digit[0][0]  # 第一欄是 label
    digit_pixels = digit[0][1:]  # 其餘欄是像素值
    image_size = int(math.sqrt(digit_pixels.shape[0]))  # 計算圖像尺寸 (28x28)
    frame = digit_pixels.reshape((image_size, image_size))  # 還原為圖像
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap='Greys')
    plt.title(digit_label)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# =============================
# Step 2: 資料分割為訓練集與測試集
# =============================
train_data = data.sample(frac=0.8)  # 隨機抽取 80% 作為訓練集
test_data = data.drop(train_data.index)  # 剩下作為測試集

train_data = train_data.values  # 轉成 NumPy array
test_data = test_data.values

# 取前 5000 筆作為訓練樣本
num_training_examples = 5000
x_train = train_data[:num_training_examples, 1:]  # 特徵
y_train = train_data[:num_training_examples, [0]]  # 標籤

x_test = test_data[:, 1:]
y_test = test_data[:, [0]]

# =============================
# Step 3: 建立並訓練多層感知器模型
# =============================
layers = [784, 25, 10]  # 每層神經元數量：輸入784 → 隱藏層25 → 輸出10
normalize_data = True  # 是否對輸入特徵標準化
max_iterations = 500  # 訓練次數
alpha = 0.1  # 學習率

# 初始化多層感知器模型
# 輸入：
#   x_train: 特徵資料
#   y_train: 對應標籤
#   layers: 網絡結構設定
#   normalize_data: 是否正規化
# 輸出：
#   建立 MLP 模型實例
multilayer_perceptron = MultilayerPerceptron(x_train, y_train, layers, normalize_data)

# 訓練模型
# train(): 執行梯度下降訓練
# 輸入：最大訓練次數、學習率
# 輸出：最終權重 thetas, 每一步的損失 costs
(thetas, costs) = multilayer_perceptron.train(max_iterations, alpha)

# 顯示訓練過程的成本變化（收斂曲線）
plt.plot(range(len(costs)), costs)
plt.xlabel('Gradient steps')
plt.ylabel('Costs')
plt.title('訓練成本隨梯度步數變化')
plt.show()

# =============================
# Step 4: 預測與準確率計算
# =============================

# 預測訓練與測試資料
# 輸入：x_train, x_test
# 輸出：預測類別
y_train_predictions = multilayer_perceptron.predict(x_train)
y_test_predictions = multilayer_perceptron.predict(x_test)

# 計算準確率（正確預測數 / 總樣本數）
train_p = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
test_p = np.sum(y_test_predictions == y_test) / y_test.shape[0] * 100
print('訓練集準確率：', train_p)
print('測試集準確率：', test_p)

# =============================
# Step 5: 顯示測試預測結果（前 64 張）
# 正確：綠色；錯誤：紅色
# =============================
numbers_to_display = 64
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(15, 15))

for plot_index in range(numbers_to_display):
    digit_label = y_test[plot_index, 0]  # 真實標籤
    digit_pixels = x_test[plot_index, :]  # 像素資料
    predicted_label = y_test_predictions[plot_index][0]  # 預測值

    image_size = int(math.sqrt(digit_pixels.shape[0]))
    frame = digit_pixels.reshape((image_size, image_size))

    color_map = 'Greens' if predicted_label == digit_label else 'Reds'  # 正確綠色，錯誤紅色
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap=color_map)
    plt.title(f'預測: {predicted_label}')
    plt.tick_params(axis='both', which='both', bottom=False, left=False,
                    labelbottom=False, labelleft=False)

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
