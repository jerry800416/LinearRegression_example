import numpy as np                  # 匯入 NumPy 套件，用於數值運算與矩陣計算
import pandas as pd                 # 匯入 Pandas 套件，用於資料讀取與處理
import matplotlib.pyplot as plt     # 匯入 Matplotlib 套件，用於傳統圖表繪製
plt.rc('font', family='Microsoft JhengHei') # 設定字型為微軟正黑體（可顯示中文）

# 從自訂的 linear_regression 模組中匯入線性回歸模型類別
from linear_regression import LinearRegression

# ------------------------------
# 讀取資料
# ------------------------------
# 從 CSV 檔案中讀取「世界快樂報告 2017」資料集，檔案位於 ./data 資料夾下
data = pd.read_csv('./LinearRegression/data/world-happiness-report-2017.csv')

# ------------------------------
# 資料拆分：建立訓練資料與測試資料
# ------------------------------
# 隨機抽取 80% 的資料作為訓練集 (sample(frac=0.8))，剩下 20% 作為測試集
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

# ------------------------------
# 定義輸入與輸出欄位
# ------------------------------
# 選取 'Economy..GDP.per.Capita.' 欄位作為模型的輸入特徵
input_param_name = 'Economy..GDP.per.Capita.'
# 選取 'Happiness.Score' 欄位作為模型的預測目標（標籤）
output_param_name = 'Happiness.Score'

# ------------------------------
# 提取訓練資料的輸入與輸出
# ------------------------------
# 取出訓練資料中所需的特徵，將資料轉換為 NumPy 陣列；注意雙中括號保持二維陣列格式
x_train = train_data[[input_param_name]].values
# 取出訓練資料對應的目標值，形狀也為 (樣本數, 1)
y_train = train_data[[output_param_name]].values

# ------------------------------
# 提取測試資料的輸入與輸出
# ------------------------------
# 測試資料中只取出單一特徵，注意這裡未使用雙中括號，所以回傳的是一維陣列
x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

# ------------------------------
# 繪製散點圖：展示訓練資料與測試資料的分佈
# ------------------------------
# 利用 plt.scatter() 畫出訓練數據的散點圖
plt.scatter(x_train, y_train, label='Train data')
# 畫出測試數據的散點圖，兩者以不同標籤區隔
plt.scatter(x_test, y_test, label='test data')
# 設定 x 軸標籤，對應輸入特徵名稱
plt.xlabel(input_param_name)
# 設定 y 軸標籤，對應輸出目標名稱
plt.ylabel(output_param_name)
# 設定圖表標題
plt.title('Happy')
# 顯示圖例
plt.legend()
# 顯示圖表
plt.show()

# ------------------------------
# 設定線性回歸模型的訓練超參數
# ------------------------------
num_iterations = 500         # 定義梯度下降法的迭代次數
learning_rate = 0.01         # 定義學習率，控制每次參數更新的步幅

# ------------------------------
# 建立並訓練模型
# ------------------------------
# 初始化 LinearRegression 模型，傳入訓練資料（x_train 與 y_train）
linear_regression = LinearRegression(x_train, y_train)
# 執行訓練程序，傳入學習率與迭代次數，回傳最終參數 theta 以及每一輪的損失歷史 cost_history
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

# 輸出訓練初始與最終的損失值，檢查模型訓練前後損失的變化
print('start loss', cost_history[0])
print('end loss', cost_history[-1])

# ------------------------------
# 繪製梯度下降過程中損失函數的變化圖
# ------------------------------
# 利用 plt.plot() 繪製迭代次數 (x 軸) 與對應損失 (y 軸) 的關係，觀察模型是否收斂
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iter')         # x 軸標示迭代次數
plt.ylabel('cost')         # y 軸標示損失函數值
plt.title('GD')            # 圖表標題 (GD: Gradient Descent)
plt.show()                 # 顯示圖表

# ------------------------------
# 模型預測：建立預測用的輸入數據並預測對應結果
# ------------------------------
predictions_num = 100  # 定義用於生成預測曲線的點數量

# 利用 np.linspace() 產生在訓練資料 x 值範圍內均勻分布的數值，共 predictions_num 個點
# 並用 reshape() 轉換成 (predictions_num, 1) 的格式，符合模型預期輸入形狀
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)

# 利用訓練好的模型對 x_predictions 進行預測，得到對應的 y 值（預測的 Happiness.Score）
y_predictions = linear_regression.predict(x_predictions)

# ------------------------------
# 繪製最終預測結果
# ------------------------------
# 使用散點圖顯示訓練資料與測試資料
plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
# 使用紅色線條畫出模型的預測曲線
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
# 設定 x 軸與 y 軸標籤
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
# 設定圖表標題
plt.title('Happy')
# 顯示圖例
plt.legend()
# 顯示最終圖表
plt.show()
