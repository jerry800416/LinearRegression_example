import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei') # 設定字型為微軟正黑體（可顯示中文）

from logistic_regression import LogisticRegression  # 匯入自定義的邏輯回歸模型（支援特徵擴展）

# 讀取資料集：每筆資料有兩個特徵與一個類別（是否為有效晶片）
data = pd.read_csv('./LogisticRegression/data/microchips-tests.csv')

# 定義二元分類標籤：0 表示無效，1 表示有效
validities = [0, 1]

# 指定要使用的兩個特徵欄位名稱
x_axis = 'param_1'
y_axis = 'param_2'

# 將資料依照標籤分別畫出來（散點圖）
for validity in validities:
    plt.scatter(
        data[x_axis][data['validity'] == validity],
        data[y_axis][data['validity'] == validity],
        label=validity
    )

plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title('Microchips Tests')  # 圖標題
plt.legend()
plt.show()

# ========== 資料前處理 ==========
num_examples = data.shape[0]  # 資料筆數

# 抽取兩個特徵作為訓練資料（x_train: shape 為 num_examples x 2）
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))

# 類別標籤 y（為 0 或 1）
y_train = data['validity'].values.reshape((num_examples, 1))

# ========== 設定訓練參數 ==========
max_iterations = 100000        # 訓練的最大迭代次數（設得較大以求收斂）
regularization_param = 0       # 正則化參數（此版本未使用）
polynomial_degree = 5          # 對原始特徵進行 5 次多項式特徵擴展
sinusoid_degree = 0            # 不進行正弦特徵擴展

# ========== 初始化邏輯回歸模型 ==========
logistic_regression = LogisticRegression(
    x_train,
    y_train,
    polynomial_degree,
    sinusoid_degree
)

# ========== 執行訓練 ==========
thetas, costs = logistic_regression.train(max_iterations)

# 取得每一個 theta 的名稱（用來顯示或分析）
columns = []
for theta_index in range(0, thetas.shape[1]):
    columns.append('Theta ' + str(theta_index))

# 繪製每個類別對應的訓練損失歷程圖（實際上只有一類，因為是二元分類）
labels = logistic_regression.unique_labels
plt.plot(range(len(costs[0])), costs[0], label=labels[0])
plt.plot(range(len(costs[1])), costs[1], label=labels[1])
plt.xlabel('Gradient Steps')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.legend()
plt.show()

# ========== 模型評估 ==========
# 使用訓練資料進行預測
y_train_predictions = logistic_regression.predict(x_train)

# 計算預測精確度（正確率）
precision = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
print('Training Precision: {:5.4f}%'.format(precision))

# ========== 建立決策邊界圖形區域 ==========
samples = 150  # 決策圖解析度（愈高愈平滑）
x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])
y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])

X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)

# 初始化 Z 用來記錄每個區域網格點的預測類別
Z = np.zeros((samples, samples))

# 對每一個 (x, y) 網格點進行預測，儲存在 Z 中
for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data_point = np.array([[x, y]])
        Z[x_index][y_index] = logistic_regression.predict(data_point)[0][0]

# ========== 繪製分類與決策邊界 ==========
positives = (y_train == 1).flatten()  # 有效晶片資料點
negatives = (y_train == 0).flatten()  # 無效晶片資料點

# 原始資料點（依類別分色）
plt.scatter(x_train[negatives, 0], x_train[negatives, 1], label='0')
plt.scatter(x_train[positives, 0], x_train[positives, 1], label='1')

# 畫出預測分類邊界（輪廓線）
plt.contour(X, Y, Z, levels=[0.5])  # 使用 0.5 作為邊界判斷點

plt.xlabel('param_1')
plt.ylabel('param_2')
plt.title('Microchips Tests')
plt.legend()
plt.show()
