import numpy as np
from utils.features import prepare_for_training

class LinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        初始化線性回歸模型
        
        參數:
        - data: 原始輸入資料（通常為 NumPy 陣列）
        - labels: 對應的標籤或目標值
        - polynomial_degree: 多項式擴充的次數（預設為 0，不進行多項式轉換）
        - sinusoid_degree: 正弦函數擴充的次數（預設為 0，不做轉換）
        - normalize_data: 是否對資料進行正規化（標準化）
        
        內部流程:
        1. 呼叫 prepare_for_training 函數對原始資料做預處理。
        2. 將預處理後的資料、各特徵的平均值與標準差儲存於物件屬性，以利後續預測或模型更新時使用。
        3. 根據預處理後的資料形狀，取得特徵個數（包含偏置項）。
        4. 初始化參數向量 theta，其維度為 (特徵數, 1)，初始值均設為 0。
        """
        (data_processed,
         features_mean, 
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)
         
        self.data = data_processed                   # 預處理後的資料矩陣（每筆資料已轉換成特徵向量）
        self.labels = labels                         # 標籤向量或目標值
        self.features_mean = features_mean           # 各特徵的平均值（用於數據正規化，預測時也需用到）
        self.features_deviation = features_deviation # 各特徵的標準差（用於數據正規化）
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        
        num_features = self.data.shape[1]            # 特徵數：資料矩陣的列數（通常包含第一列全為 1 作為偏置項）
        self.theta = np.zeros((num_features, 1))       # 參數向量 theta 初始化為 0

    def train(self, alpha, num_iterations=500):
        """
        執行模型訓練：利用梯度下降法進行參數更新
        
        參數:
        - alpha: 學習率，控制每次參數更新的幅度
        - num_iterations: 執行梯度下降法的迭代次數（預設 500 次）
        
        回傳值:
        - theta: 訓練後的參數向量
        - cost_history: 每次迭代時的損失函數值列表，用於檢視訓練過程是否收斂
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        梯度下降迭代過程
        
        參數:
        - alpha: 學習率
        - num_iterations: 總迭代次數
        
        操作:
        - 在每次迭代中，呼叫 gradient_step 進行參數更新。
        - 同時計算當前參數下的損失函數，並記錄到 cost_history 清單中。
        
        回傳值:
        - cost_history: 儲存每次迭代後損失函數的值
        """
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)                               # 執行一次梯度下降的參數更新
            cost_history.append(self.cost_function(self.data, self.labels))  # 計算並記錄當前損失
        return cost_history

    def gradient_step(self, alpha):
        """
        單次梯度下降參數更新步驟（運用矩陣運算）
        
        過程說明:
        1. 取得樣本數 m。
        2. 利用 hypothesis 函數計算目前的預測值（矩陣乘法：data 與 theta）。
        3. 計算預測與真實值之間的差距（delta）。
        4. 利用矩陣運算計算梯度：np.dot(delta.T, data) 計算所有訓練樣本的累積梯度，再取平均（除以 m）。
        5. 利用學習率 alpha 來調整參數 theta 的更新步幅，並更新 theta。
        """
        num_examples = self.data.shape[0]  # m：樣本數
        prediction = LinearRegression.hypothesis(self.data, self.theta)  # 預測值向量
        delta = prediction - self.labels     # 計算誤差向量：預測值與實際標籤的差距
        theta = self.theta                   # 獲取目前的參數
        # 更新參數：gradient = (1/m) * (X^T * delta)，這裡先計算 delta.T * data，然後再轉置
        theta = theta - alpha * (1/num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta                   # 儲存更新後的參數

    def cost_function(self, data, labels):
        """
        計算損失函數（Cost Function），這裡採用 Mean Squared Error (MSE) 損失，外加 1/2 以便於計算導數
        
        參數:
        - data: 輸入資料矩陣
        - labels: 真實標籤向量
        
        運算:
        1. 計算所有樣本的誤差向量（預測值與真實值之差）。
        2. 利用 np.dot(delta.T, delta) 算出誤差平方和。
        3. 乘上 (1/2) 再除以樣本數 m，以獲得平均損失值。
        
        回傳:
        - 該批次資料的損失值（浮點數）
        """
        num_examples = data.shape[0]  # m: 樣本數
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels  # 注意：這裡使用 self.data 與 self.theta
        cost = (1/2) * np.dot(delta.T, delta) / num_examples  # 損失函數計算 (1/2m)*∑(誤差^2)
        return cost[0][0]  # cost 為一維矩陣，返回其數值

    @staticmethod
    def hypothesis(data, theta):
        """
        假設函數：計算預測值
        
        參數:
        - data: 輸入資料矩陣
        - theta: 參數向量
        
        運算:
        - 利用矩陣乘法：預測值 = data dot theta
        
        回傳:
        - 預測結果向量
        """
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        """
        計算給定資料的損失值，通常用於驗證或測試模型
        
        參數:
        - data: 原始測試資料（未必與訓練資料一致）
        - labels: 真實標籤
        
        流程:
        1. 呼叫 prepare_for_training 將輸入資料進行相同的預處理，確保資料格式一致。
        2. 呼叫 cost_function 計算預測後的損失值。
        
        回傳:
        - 指定資料的損失值
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data)[0]
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        利用訓練好的模型進行預測
        
        參數:
        - data: 待預測的原始輸入資料
        
        流程:
        1. 將輸入資料透過 prepare_for_training 做相同的預處理。
        2. 利用 hypothesis 函數與訓練後的參數 theta 得到預測值。
        
        回傳:
        - 預測結果（通常為一個浮點數陣列）
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions


