# -*- coding: UTF-8 -*-
import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient


class MultilayerPerceptron:
    """
    多層感知器（MLP）實作：用於訓練與預測 MNIST 類型的影像分類問題。
    使用前向傳播、反向傳播與梯度下降訓練神經網路。
    """

    def __init__(self, data, labels, layers, normalize_data=False):
        """
        初始化 MLP 類別

        參數：
        - data: 原始輸入資料
        - labels: 標籤（類別編號）
        - layers: 各層神經元數目（如 [784, 25, 10]）
        - normalize_data: 是否對輸入資料正規化

        輸出：無（建構內部參數 self.thetas）
        """
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]
        self.data = data_processed
        self.labels = labels
        self.layers = layers
        self.normalize_data = normalize_data
        self.thetas = MultilayerPerceptron.thetas_init(layers)

    def predict(self, data):
        """
        預測輸入資料的類別

        參數：
        - data: 要預測的輸入資料（N x D）

        回傳：
        - 預測類別（N x 1）
        """
        data_processed = prepare_for_training(data, normalize_data=self.normalize_data)[0]
        num_examples = data_processed.shape[0]
        predictions = MultilayerPerceptron.feedforward_propagation(
            data_processed, self.thetas, self.layers
        )
        return np.argmax(predictions, axis=1).reshape((num_examples, 1))

    def train(self, max_iterations=1000, alpha=0.1):
        """
        執行訓練（反向傳播 + 梯度下降）

        參數：
        - max_iterations: 最大迭代次數
        - alpha: 學習率

        回傳：
        - 訓練後的參數 thetas
        - 每一輪的 cost 歷史紀錄
        """
        unrolled_theta = MultilayerPerceptron.thetas_unroll(self.thetas)
        optimized_theta, cost_history = MultilayerPerceptron.gradient_descent(
            self.data, self.labels, unrolled_theta, self.layers, max_iterations, alpha
        )
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_theta, self.layers)
        return self.thetas, cost_history

    @staticmethod
    def thetas_init(layers):
        """
        初始化每層的 theta 參數（隨機）

        參數：
        - layers: 每層神經元數目

        回傳：
        - thetas: 字典格式的每層權重矩陣
        """
        num_layers = len(layers)
        thetas = {}
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            # 權重初始化，範圍小避免梯度爆炸
            thetas[layer_index] = np.random.rand(out_count, in_count + 1) * 0.05
        return thetas

    @staticmethod
    def thetas_unroll(thetas):
        """
        將 thetas 字典攤平成一個向量（便於優化）

        輸入：
        - thetas: 權重字典

        回傳：
        - unrolled_theta: 向量形式的所有權重
        """
        unrolled_theta = np.array([])
        for theta in thetas.values():
            unrolled_theta = np.hstack((unrolled_theta, theta.flatten()))
        return unrolled_theta

    @staticmethod
    def gradient_descent(data, labels, unrolled_theta, layers, max_iterations, alpha):
        """
        執行梯度下降

        回傳：
        - optimized_theta: 訓練後權重
        - cost_history: 每步的 cost 值
        """
        optimized_theta = unrolled_theta
        cost_history = []
        for _ in range(max_iterations):
            cost = MultilayerPerceptron.cost_function(
                data, labels,
                MultilayerPerceptron.thetas_roll(optimized_theta, layers),
                layers
            )
            cost_history.append(cost)
            theta_gradient = MultilayerPerceptron.gradient_step(data, labels, optimized_theta, layers)
            optimized_theta = optimized_theta - alpha * theta_gradient
        return optimized_theta, cost_history

    @staticmethod
    def gradient_step(data, labels, optimized_theta, layers):
        """
        執行一次梯度計算步驟

        回傳：
        - 梯度向量（與 unrolled_theta 對應）
        """
        theta = MultilayerPerceptron.thetas_roll(optimized_theta, layers)
        gradients = MultilayerPerceptron.back_propagation(data, labels, theta, layers)
        return MultilayerPerceptron.thetas_unroll(gradients)

    @staticmethod
    def back_propagation(data, labels, thetas, layers):
        """
        執行反向傳播演算法計算所有權重的梯度

        回傳：
        - 每層的梯度矩陣（與 thetas 結構相同）
        """
        num_layers = len(layers)
        num_examples, num_features = data.shape
        num_label_types = layers[-1]
        deltas = {l: np.zeros_like(thetas[l]) for l in thetas}

        for example_index in range(num_examples):
            layers_inputs = {}
            layers_activations = {}
            activation = data[example_index, :].reshape((num_features, 1))
            layers_activations[0] = activation

            # 前向傳播
            for layer_index in range(num_layers - 1):
                theta = thetas[layer_index]
                z = np.dot(theta, activation)
                activation = np.vstack(([1], sigmoid(z)))
                layers_inputs[layer_index + 1] = z
                layers_activations[layer_index + 1] = activation

            # one-hot 編碼標籤
            bitwise_label = np.zeros((num_label_types, 1))
            bitwise_label[labels[example_index][0]] = 1

            # 輸出層誤差
            delta = {}
            delta[num_layers - 1] = layers_activations[num_layers - 1][1:, :] - bitwise_label

            # 隱藏層誤差反向傳播
            for layer_index in reversed(range(1, num_layers - 1)):
                theta = thetas[layer_index]
                prev_delta = delta[layer_index + 1]
                z = np.vstack(([1], layers_inputs[layer_index]))
                d = np.dot(theta.T, prev_delta) * sigmoid_gradient(z)
                delta[layer_index] = d[1:, :]  # 移除偏置誤差

            # 累加梯度
            for layer_index in range(num_layers - 1):
                layer_delta = np.dot(delta[layer_index + 1], layers_activations[layer_index].T)
                deltas[layer_index] += layer_delta

        # 平均化
        for layer_index in deltas:
            deltas[layer_index] /= num_examples

        return deltas

    @staticmethod
    def cost_function(data, labels, thetas, layers):
        """
        損失函數計算（交叉熵）

        回傳：
        - cost: 平均損失值
        """
        num_examples = data.shape[0]
        predictions = MultilayerPerceptron.feedforward_propagation(data, thetas, layers)
        num_labels = layers[-1]
        bitwise_labels = np.zeros((num_examples, num_labels))
        for i in range(num_examples):
            bitwise_labels[i, labels[i][0]] = 1
        positive = np.log(predictions[bitwise_labels == 1])
        negative = np.log(1 - predictions[bitwise_labels == 0])
        return - (np.sum(positive) + np.sum(negative)) / num_examples

    @staticmethod
    def feedforward_propagation(data, thetas, layers):
        """
        前向傳播：輸入資料 → 輸出預測

        回傳：
        - 每個樣本的預測機率分布（num_examples x num_classes）
        """
        num_layers = len(layers)
        activation = data
        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]
            z = np.dot(activation, theta.T)
            activation = sigmoid(z)
            activation = np.hstack((np.ones((activation.shape[0], 1)), activation))  # 加偏置
        return activation[:, 1:]  # 去除最後一層偏置

    @staticmethod
    def thetas_roll(unrolled_thetas, layers):
        """
        將一維向量還原為每層的 theta 結構

        回傳：
        - 字典形式的 thetas
        """
        thetas = {}
        shift = 0
        for layer_index in range(len(layers) - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            size = (out_count, in_count + 1)
            volume = size[0] * size[1]
            theta_flat = unrolled_thetas[shift:shift + volume]
            thetas[layer_index] = theta_flat.reshape(size)
            shift += volume
        return thetas
