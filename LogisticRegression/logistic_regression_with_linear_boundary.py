import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei') # 設定字型為微軟正黑體（可顯示中文）
plt.rcParams['axes.unicode_minus'] = False  # 設定負號可以正確顯示
from logistic_regression import LogisticRegression  # 匯入自定義的邏輯回歸類別

# 讀取資料集（Iris 鳶尾花資料集）
data = pd.read_csv('./LogisticRegression/data/iris.csv')

# 定義三個類別標籤
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

# 指定要用來訓練與繪圖的兩個特徵：花瓣長度與寬度
x_axis = 'petal_length'
y_axis = 'petal_width'

# 根據類別繪製散佈圖（用來視覺化三類資料的分佈情形）
for iris_type in iris_types:
    plt.scatter(
        data[x_axis][data['class'] == iris_type],
        data[y_axis][data['class'] == iris_type],
        label=iris_type
    )
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title("Iris 散佈圖")
plt.legend()
plt.show()

# 資料處理：
num_examples = data.shape[0]  # 資料筆數
# 將特徵轉成 numpy array (只使用兩個特徵)
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
# 將標籤轉成 numpy array，shape 為 (num_examples, 1)
y_train = data['class'].values.reshape((num_examples, 1))

# 訓練參數設定
max_iterations = 1000         # 最多訓練迭代次數
polynomial_degree = 0         # 不使用多項式擴展
sinusoid_degree = 0           # 不使用正弦擴展

# 初始化邏輯回歸模型（自定義的 LogisticRegression 類別）
logistic_regression = LogisticRegression(
    x_train,
    y_train,
    polynomial_degree,
    sinusoid_degree
)

# 訓練模型，取得 θ 參數與每個類別對應的損失值歷程
thetas, cost_histories = logistic_regression.train(max_iterations)

# 取得模型所辨識的所有類別（順序與 θ 對應）
labels = logistic_regression.unique_labels

# 將三個類別對應的損失值曲線畫出來（觀察訓練過程收斂情況）
plt.plot(range(len(cost_histories[0])), cost_histories[0], label=labels[0])
plt.plot(range(len(cost_histories[1])), cost_histories[1], label=labels[1])
plt.plot(range(len(cost_histories[2])), cost_histories[2], label=labels[2])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("損失值收斂曲線")
plt.legend()
plt.show()

# 在訓練資料上進行預測，並計算訓練精確度（百分比）
y_train_predictions = logistic_regression.predict(x_train)
precision = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
print('訓練精確度：', precision)

# 準備畫決策邊界的區域網格
x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])
y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])
samples = 150  # 決策圖的解析度

X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)

# 初始化決策圖用的類別網格（預測結果為三類之一）
Z_SETOSA = np.zeros((samples, samples))
Z_VERSICOLOR = np.zeros((samples, samples))
Z_VIRGINICA = np.zeros((samples, samples))

# 針對每一個網格點預測其類別，並標記到對應的類別平面上
for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data_point = np.array([[x, y]])  # 組成一個輸入點
        prediction = logistic_regression.predict(data_point)[0][0]  # 預測結果是類別名稱
        if prediction == 'SETOSA':
            Z_SETOSA[x_index][y_index] = 1
        elif prediction == 'VERSICOLOR':
            Z_VERSICOLOR[x_index][y_index] = 1
        elif prediction == 'VIRGINICA':
            Z_VIRGINICA[x_index][y_index] = 1

# 再次繪製原始資料點（依據真實類別顯示）
for iris_type in iris_types:
    plt.scatter(
        x_train[(y_train == iris_type).flatten(), 0],
        x_train[(y_train == iris_type).flatten(), 1],
        label=iris_type
    )

# 加入決策邊界圖：畫出每個類別的輪廓線
plt.contour(X, Y, Z_SETOSA, levels=[0.5])
plt.contour(X, Y, Z_VERSICOLOR, levels=[0.5])
plt.contour(X, Y, Z_VIRGINICA, levels=[0.5])

plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title("分類決策邊界")
plt.legend()
plt.show()
