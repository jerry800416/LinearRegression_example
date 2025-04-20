import numpy as np

class KMeans:
    def __init__(self, data, num_clustres):
        '''
        初始化函式：傳入資料與要分的群數
        '''
        self.data = data
        self.num_clustres = num_clustres
        
    def train(self, max_iterations):
        '''
        訓練函式：執行最大 max_iterations 次疊代以收斂 K-means 分群
        '''
        # 1. 隨機初始化 K 個群中心點（centroids）
        centroids = KMeans.centroids_init(self.data, self.num_clustres)

        # 取得資料筆數
        num_examples = self.data.shape[0]
        
        # 預先建立陣列，用來儲存每一筆資料對應的最近中心點 ID
        closest_centroids_ids = np.empty((num_examples, 1))

        # 2. 進行疊代訓練
        for _ in range(max_iterations):
            # 3. 對每筆資料，找出距離最近的中心點
            closest_centroids_ids = KMeans.centroids_find_closest(self.data, centroids)

            # 4. 根據指派結果，重新計算各中心點的位置（取平均）
            centroids = KMeans.centroids_compute(self.data, closest_centroids_ids, self.num_clustres)

        # 回傳最終的群中心與每筆資料的分群結果
        return centroids, closest_centroids_ids
                
    @staticmethod    
    def centroids_init(data, num_clustres):
        '''
        隨機從資料中選出 K 筆作為初始中心點
        '''
        # 資料總筆數
        num_examples = data.shape[0]

        # 隨機打亂索引順序
        random_ids = np.random.permutation(num_examples)

        # 從打亂後的資料中取前 K 筆作為初始中心點
        centroids = data[random_ids[:num_clustres], :]

        return centroids

    @staticmethod 
    def centroids_find_closest(data, centroids):
        '''
        尋找每筆資料最近的中心點
        '''
        num_examples = data.shape[0]      # 資料總筆數
        num_centroids = centroids.shape[0]  # 中心點數量（K 值）

        # 建立陣列用來儲存每筆資料對應的最近中心點 ID
        closest_centroids_ids = np.zeros((num_examples, 1))

        # 遍歷每筆資料
        for example_index in range(num_examples):
            distance = np.zeros((num_centroids, 1))  # 儲存該筆資料到所有中心點的距離

            for centroid_index in range(num_centroids):
                # 計算資料點與該中心點的差值（向量相減）
                distance_diff = data[example_index, :] - centroids[centroid_index, :]

                # 計算平方距離（不開根號也可比較大小）
                distance[centroid_index] = np.sum(distance_diff**2)

            # 將該資料點指派給距離最近的中心點
            closest_centroids_ids[example_index] = np.argmin(distance)

        return closest_centroids_ids

    @staticmethod
    def centroids_compute(data, closest_centroids_ids, num_clustres):
        '''
        根據每筆資料被指派的群組，重新計算每個中心點的位置
        '''
        num_features = data.shape[1]  # 特徵維度
        centroids = np.zeros((num_clustres, num_features))  # 初始化中心點陣列

        for centroid_id in range(num_clustres):
            # 找出被分配到當前 centroid_id 的所有資料點索引
            closest_ids = closest_centroids_ids == centroid_id

            # 計算這些點的平均值作為新的中心點位置
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(), :], axis=0)

        return centroids
