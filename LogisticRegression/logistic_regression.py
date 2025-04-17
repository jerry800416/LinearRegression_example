import numpy as np
from scipy.optimize import minimize
from utils.features import prepare_for_training  # 資料預處理工具（如特徵擴展與標準化）
from utils.hypothesis import sigmoid  # sigmoid 函數實作

class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        """
        初始化邏輯回歸模型：
        1. 對輸入資料進行特徵擴展與標準化等預處理。
        2. 初始化模型參數（θ）為 0。
        """

        # 資料預處理：進行多項式擴展、正弦擴展與標準化
        (data_processed,
         features_mean, 
         features_deviation) = prepare_for_training(
            data, polynomial_degree, sinusoid_degree, normalize_data=False
        )

        self.data = data_processed              # 預處理後的特徵資料
        self.labels = labels                    # 標籤資料（可為多類別）
        self.unique_labels = np.unique(labels)  # 所有唯一的類別值
        self.features_mean = features_mean      # 各特徵的平均值（用於標準化還原）
        self.features_deviation = features_deviation  # 各特徵的標準差
        self.polynomial_degree = polynomial_degree    # 多項式特徵階數
        self.sinusoid_degree = sinusoid_degree        # 正弦特徵階數
        self.normalize_data = normalize_data          # 是否啟用標準化

        num_features = self.data.shape[1]  # 特徵數量
        num_unique_labels = np.unique(labels).shape[0]  # 類別數量

        # 初始化 θ 參數矩陣為全 0（每個類別一組參數）
        self.theta = np.zeros((num_unique_labels, num_features))

    def train(self, max_iterations=1000):
        """
        使用邏輯回歸進行訓練：
        - 對每一個類別進行 One-vs-All（二元分類）
        - 使用 scipy 的 minimize 函數與 Conjugate Gradient 方法優化 θ
        """
        cost_histories = []  # 用於紀錄每一類別的損失歷程
        num_features = self.data.shape[1]

        for label_index, unique_label in enumerate(self.unique_labels):
            # 初始化目前類別的 θ 向量
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features, 1))

            # 將目前類別設定為正類（1），其他為負類（0）
            current_labels = (self.labels == unique_label).astype(float)

            # 執行梯度下降優化
            (current_theta, cost_history) = LogisticRegression.gradient_descent(
                self.data, current_labels, current_initial_theta, max_iterations
            )

            # 儲存最終的 θ
            self.theta[label_index] = current_theta.T
            cost_histories.append(cost_history)

        return self.theta, cost_histories

    @staticmethod
    def gradient_descent(data, labels, current_initial_theta, max_iterations):
        """
        使用 scipy.optimize.minimize 執行參數優化。
        此處選用 Conjugate Gradient (CG) 方法，效果比傳統梯度下降更快。
        """
        cost_history = []  # 用來追蹤損失值變化
        num_features = data.shape[1]

        result = minimize(
            fun=lambda current_theta: LogisticRegression.cost_function(
                data, labels, current_theta.reshape(num_features, 1)
            ),
            x0=current_initial_theta.flatten(),  # 初始 θ 攤平為一維陣列
            method='CG',  # 選用 Conjugate Gradient 演算法
            jac=lambda current_theta: LogisticRegression.gradient_step(
                data, labels, current_theta.reshape(num_features, 1)
            ),
            callback=lambda current_theta: cost_history.append(
                LogisticRegression.cost_function(
                    data, labels, current_theta.reshape(num_features, 1)
                )
            ),
            options={'maxiter': max_iterations}
        )

        # 若優化失敗則拋出錯誤
        if not result.success:
            raise ArithmeticError('無法最小化損失函數：' + result.message)

        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, cost_history

    @staticmethod
    def cost_function(data, labels, theta):
        """
        計算交叉熵（Cross-Entropy）損失函數：
        J(θ) = -1/m * Σ [y log(h(x)) + (1 - y) log(1 - h(x))]
        """
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)

        # 對於 y=1 的樣本計算 log(h(x))
        y_is_set_cost = np.dot(labels[labels == 1].T, np.log(predictions[labels == 1]))

        # 對於 y=0 的樣本計算 log(1 - h(x))
        y_is_not_set_cost = np.dot((1 - labels[labels == 0]).T, np.log(1 - predictions[labels == 0]))

        # 平均化損失
        cost = (-1 / num_examples) * (y_is_set_cost + y_is_not_set_cost)
        return cost

    @staticmethod
    def hypothesis(data, theta):
        """
        預測函數：使用 sigmoid 函數產生機率輸出
        h(x) = sigmoid(θ^T x)
        """
        predictions = sigmoid(np.dot(data, theta))
        return predictions

    @staticmethod
    def gradient_step(data, labels, theta):
        """
        計算損失函數對 θ 的梯度（∇J）：
        ∇J = (1/m) * X^T (h(x) - y)
        """
        num_examples = labels.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        label_diff = predictions - labels  # 機率預測 - 真實標籤
        gradients = (1 / num_examples) * np.dot(data.T, label_diff)
        return gradients.T.flatten()  # 攤平回傳為一維陣列以符合 scipy 最佳化器需求

    def predict(self, data):
        """
        使用訓練好的 θ 預測新資料的類別：
        - 計算每個類別的機率
        - 選擇機率最高的類別作為預測結果
        """
        num_examples = data.shape[0]

        # 對新資料進行同樣的特徵處理
        data_processed = prepare_for_training(
            data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data
        )[0]

        # 計算每個類別的機率（每筆資料會有 num_classes 個機率值）
        prob = LogisticRegression.hypothesis(data_processed, self.theta.T)

        # 選擇最大機率對應的索引作為預測類別
        max_prob_index = np.argmax(prob, axis=1)

        # 將預測結果對應回原本的類別名稱
        class_prediction = np.empty(max_prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            class_prediction[max_prob_index == index] = label

        return class_prediction.reshape((num_examples, 1))
