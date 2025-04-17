import numpy as np                  # 匯入 NumPy 套件，用於數值運算與矩陣操作
import pandas as pd                 # 匯入 Pandas 套件，用於資料讀取與處理
import matplotlib.pyplot as plt     # 匯入 Matplotlib 用於繪製傳統圖表
plt.rc('font', family='Microsoft JhengHei') # 設定字型為微軟正黑體（可顯示中文）
import plotly                      # 匯入 Plotly 套件，用於互動式 3D 圖表繪製
import plotly.graph_objs as go      # 匯入 Plotly 的圖表物件模組

# 設定 Plotly 為離線模式並初始化（適用於 Jupyter Notebook 等環境）
plotly.offline.init_notebook_mode()

# 從自訂的 linear_regression 模組中匯入 LinearRegression 類別
from linear_regression import LinearRegression

# ------------------------------
# 資料讀取與資料集拆分
# ------------------------------

# 讀取世界快樂報告 2017 年的 CSV 資料，此檔案包含多個欄位資料
data = pd.read_csv('./LinearRegression/data/world-happiness-report-2017.csv')

# 隨機從原始資料中抽取 80% 作為訓練資料，剩餘 20% 作為測試資料
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

# 選定兩個作為輸入特徵的欄位名稱
input_param_name_1 = 'Economy..GDP.per.Capita.'
input_param_name_2 = 'Freedom'
# 選定作為預測目標的輸出欄位名稱
output_param_name = 'Happiness.Score'

# 從訓練資料中取出指定的輸入參數欄位，轉換成 NumPy 陣列
# 兩個欄位資料會變成矩陣，形狀為 (訓練樣本數, 2)
x_train = train_data[[input_param_name_1, input_param_name_2]].values
# 取出目標標籤欄位，形狀為 (訓練樣本數, 1)
y_train = train_data[[output_param_name]].values

# 從測試資料中取出相同的輸入與目標欄位，分別轉成 NumPy 陣列
x_test = test_data[[input_param_name_1, input_param_name_2]].values
y_test = test_data[[output_param_name]].values

# ------------------------------
# 使用 Plotly 繪製 3D 資料分布圖（訓練集與測試集）
# ------------------------------

# 定義訓練資料的 3D 散點圖 trace
plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),    # 訓練資料中第一個特徵（經濟/人均 GDP），展平成一維
    y=x_train[:, 1].flatten(),    # 訓練資料中第二個特徵（自由度）
    z=y_train.flatten(),          # 對應的快樂分數
    name='Training Set',          # 圖例名稱
    mode='markers',               # 以散點圖的方式呈現
    marker={
        'size': 10,              # 設定點的大小
        'opacity': 1,            # 設定點的透明度
        'line': {                # 設定點邊緣樣式
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)

# 定義測試資料的 3D 散點圖 trace
plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),
    y=x_test[:, 1].flatten(),
    z=y_test.flatten(),
    name='Test Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)

# 設定圖表整體版面佈局，包括標題與各軸的標籤
plot_layout = go.Layout(
    title='Date Sets',
    scene={
        'xaxis': {'title': input_param_name_1},
        'yaxis': {'title': input_param_name_2},
        'zaxis': {'title': output_param_name}
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}  # 設定圖表邊界
)

# 合併上述兩個 trace 成為一個資料列表
plot_data = [plot_training_trace, plot_test_trace]

# 建立最終的圖表物件
plot_figure = go.Figure(data=plot_data, layout=plot_layout)

# 使用 Plotly 離線模式在網頁中呈現該 3D 圖表
plotly.offline.plot(plot_figure)


# ------------------------------
# 設定線性回歸模型的參數與訓練超參數
# ------------------------------
num_iterations = 500          # 設定梯度下降法的迭代次數
learning_rate = 0.01          # 設定學習率，決定每次參數更新的步幅
polynomial_degree = 0         # 此處不進行多項式特徵擴充
sinusoid_degree = 0           # 此處不進行正弦特徵擴充

# 初始化 LinearRegression 模型，傳入訓練資料及對應參數設定
linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)

# ------------------------------
# 模型訓練
# ------------------------------
# 執行線性回歸的訓練程序，回傳訓練後的參數 theta 與每次迭代的損失歷史
(theta, cost_history) = linear_regression.train(
    learning_rate,
    num_iterations
)

# 輸出訓練開始時與訓練結束時的損失值，以檢查模型訓練收斂情形
print('start loss', cost_history[0])
print('end loss', cost_history[-1])

# ------------------------------
# 繪製梯度下降進展圖
# ------------------------------
# 利用 matplotlib 繪製損失函數隨著迭代次數變化的圖表，以便觀察訓練收斂狀況
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')                   # x 軸標示迭代次數
plt.ylabel('Cost')                         # y 軸標示損失函數值
plt.title('Gradient Descent Progress')     # 圖表標題
plt.show()                                 # 顯示圖表

# ------------------------------
# 進行模型預測：建立 x-y 預測平面
# ------------------------------

# 設定在 x 與 y 軸上取多少點來生成預測平面的網格，這裡使用 10 個點
predictions_num = 10

# 取得 x 軸（第一個輸入特徵）的最小與最大值，用於建立等差數列
x_min = x_train[:, 0].min()
x_max = x_train[:, 0].max()

# 取得 y 軸（第二個輸入特徵）的最小與最大值
y_min = x_train[:, 1].min()
y_max = x_train[:, 1].max()

# 在 x 軸與 y 軸分別產生等差數列，共 predictions_num 個點
x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)

# 初始化預測用的輸入矩陣，預先建立空的陣列，行數為網格點總數（predictions_num * predictions_num），列數為 1
x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))

# 利用雙層迴圈將 x 與 y 軸的所有組合填入預測的資料矩陣中
x_y_index = 0  # 用於追蹤當前填入的位置
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value  # 設定第一個特徵值
        y_predictions[x_y_index] = y_value  # 設定第二個特徵值
        x_y_index += 1                   # 更新索引以存入下一筆資料

# 將兩個特徵合併成最終的預測輸入矩陣，形狀將為 (網格總點數, 2)
# np.hstack() 用來進行水平合併
z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

# ------------------------------
# 繪製預測平面並展示與訓練、測試資料的關係
# ------------------------------

# 定義預測平面在 3D 圖中的 trace，使用散點圖呈現較小尺寸的點以模擬平面效果
plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),        # 預測平面上每個點的 x 值
    y=y_predictions.flatten(),        # 預測平面上每個點的 y 值
    z=z_predictions.flatten(),        # 預測平面上每個點經模型計算出的 z 值（Happiness Score）
    name='Prediction Plane',          # 圖例名稱
    mode='markers',                   # 使用散點圖方式呈現
    marker={
        'size': 1,                   # 點的尺寸較小，形成平面效果
    },
    opacity=0.8,                      # 設定透明度，使預測平面與資料重疊時能看清數據點
    surfaceaxis=2,                    # 指定用哪個軸作為平面方向（通常是 z 軸）
)

# 將所有圖層（訓練資料、測試資料、預測平面）合併成一個列表
plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]

# 重新利用先前設定的版面佈局建立最終圖表物件
plot_figure = go.Figure(data=plot_data, layout=plot_layout)

# 使用 Plotly 離線模式呈現最終 3D 圖表，展示資料與模型預測平面
plotly.offline.plot(plot_figure)
