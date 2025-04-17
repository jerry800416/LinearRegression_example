# -*- coding: utf-8 -*-
import numpy as np              # 用來進行數值運算與矩陣處理
import pandas as pd             # 用來處理與分析資料（DataFrame 格式）
import matplotlib.pyplot as plt # 用來繪製圖表
plt.rc('font', family='Microsoft JhengHei') # 設定字型為微軟正黑體（可顯示中文）

# 匯入自己實作的線性回歸模型類別
from linear_regression import LinearRegression

# ------------------------------
# 讀取資料與前處理
# ------------------------------
# 讀取 CSV 檔案，檔案中包含 x 與 y 兩個欄位，分別代表輸入與輸出
data = pd.read_csv('./LinearRegression/data/non-linear-regression-x-y.csv')

# 取出 x 與 y 欄位，並轉換成 numpy 陣列，同時利用 reshape 將原本的一維陣列轉換成 (樣本數, 1) 的形式
x = data['x'].values.reshape((data.shape[0], 1))
y = data['y'].values.reshape((data.shape[0], 1))

# 顯示資料前 10 筆（可檢查資料格式與數值）
data.head(10)

# ------------------------------
# 繪製原始資料分佈圖
# ------------------------------
# 使用 plt.plot() 將 x 與 y 的關係做出連續曲線圖，可以直觀地看到資料走勢
plt.plot(x, y)
plt.show()  # 顯示圖表

# ------------------------------
# 設定訓練超參數
# ------------------------------
# 設定梯度下降法所需參數
num_iterations = 50000       # 總迭代次數，較多的迭代次數有助於模型收斂
learning_rate = 0.02         # 學習率，控制每次參數更新的步幅
polynomial_degree = 15       # 多項式轉換的次數（特徵擴充），幫助捕捉非線性關係
sinusoid_degree = 15         # 正弦函數轉換的次數，若數據呈現週期性波動可使用
normalize_data = True        # 是否要對資料進行正規化，能使不同尺度數值平滑收斂

# ------------------------------
# 初始化線性回歸模型
# ------------------------------
# 傳入原始輸入 x 與標籤 y，以及設定的多項式與正弦轉換度數與正規化參數，建立一個 LinearRegression 物件
linear_regression = LinearRegression(x, y, polynomial_degree, sinusoid_degree, normalize_data)

# ------------------------------
# 模型訓練：執行梯度下降法
# ------------------------------
# 調用 train() 方法，傳入學習率與迭代次數，取得訓練後的參數（theta）以及每次迭代的成本值記錄
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

# 輸出初始與最終的損失值，用以觀察模型是否有效下降
print('start loss: {:.2f}'.format(cost_history[0]))
print('end loss: {:.2f}'.format(cost_history[-1]))

# ------------------------------
# 顯示訓練後的模型參數
# ------------------------------
# 將 theta（模型參數向量）轉成一維陣列，再建立一個 DataFrame 用於顯示各參數值
theta_table = pd.DataFrame({'Model Parameters': theta.flatten()})
# 若有需要也可以用 print(theta_table) 來檢查參數內容

# ------------------------------
# 繪製梯度下降法過程的損失函數變化圖
# ------------------------------
# 繪製迭代次數與成本值的關係圖，以便觀察損失函數隨著迭代是否漸漸收斂
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')       # X 軸標示迭代次數
plt.ylabel('Cost')             # Y 軸標示損失函數值
plt.title('Gradient Descent Progress')  # 圖表標題
plt.show()  # 顯示圖表

# ------------------------------
# 利用訓練好的模型進行預測
# ------------------------------
# 設定預測用的數據點數量，透過 np.linspace 產生在 x 範圍內均勻分佈的數值
predictions_num = 1000
x_predictions = np.linspace(x.min(), x.max(), predictions_num).reshape(predictions_num, 1)

# 利用模型的 predict() 方法，計算對應的預測結果 y
y_predictions = linear_regression.predict(x_predictions)

# ------------------------------
# 繪製訓練資料及預測結果圖
# ------------------------------
# 使用散點圖顯示原始訓練數據，再用紅色線條顯示模型預測的結果曲線
plt.scatter(x, y, label='Training Dataset')      # 訓練數據點
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')  # 模型預測線
plt.legend()              # 顯示圖例，說明各線條代表意義
plt.show()                # 顯示最終圖表
