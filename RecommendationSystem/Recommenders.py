# 感謝 Siraj Raval 提供的推薦系統模組
# 原始碼來源：https://github.com/llSourcell/recommender_live

import numpy as np
import pandas

# =============================
# 基於熱門度的推薦系統類別
# =============================
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None  # 訓練資料集（DataFrame）
        self.user_id = None     # 使用者欄位名稱（字串）
        self.item_id = None     # 項目欄位名稱（字串）
        self.popularity_recommendations = None  # 儲存推薦結果的 DataFrame
        
    # 建立推薦模型
    # 輸入：
    # - train_data：訓練資料（DataFrame，含 user_id 與 item_id 欄位）
    # - user_id：欄位名稱，表示使用者
    # - item_id：欄位名稱，表示項目（如歌曲）
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # 根據每個 item_id 被不同 user_id 點擊（互動）次數作為推薦分數
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns={user_id: 'score'}, inplace=True)

        # 根據分數進行排序（熱門度）
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending=[False, True])

        # 產生排名欄位
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=False, method='first')

        # 取前 10 名最熱門項目作為推薦清單
        self.popularity_recommendations = train_data_sort.head(10)

    # 根據使用者 ID 提供推薦清單
    # 輸入：
    # - user_id：目標使用者
    # 輸出：DataFrame，顯示該使用者的推薦列表
    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations.copy()
        user_recommendations['user_id'] = user_id

        # 將 user_id 欄位放到最前面
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations

# =============================
# 基於項目相似度的推薦系統類別（Item-based CF）
# =============================
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None  # 共現矩陣
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None

    # 取得使用者互動過的項目清單
    # 輸入：user（使用者 ID）
    # 輸出：該使用者互動過的 item_id 清單
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        return user_items

    # 取得與指定項目互動過的使用者集合
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
        return item_users

    # 取得所有訓練集中出現過的項目清單
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
        return all_items

    # 建立共現矩陣（使用 Jaccard 相似度）
    # 輸入：
    # - user_songs：使用者互動過的歌曲清單
    # - all_songs：訓練集所有歌曲
    # 輸出：共現矩陣 (user_songs × all_songs)
    def construct_cooccurence_matrix(self, user_songs, all_songs):
        user_songs_users = []
        for song in user_songs:
            user_songs_users.append(self.get_item_users(song))

        # 初始化共現矩陣
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        for i in range(len(all_songs)):
            users_i = self.get_item_users(all_songs[i])
            for j in range(len(user_songs)):
                users_j = user_songs_users[j]
                intersection = users_i.intersection(users_j)
                if len(intersection) != 0:
                    union = users_i.union(users_j)
                    cooccurence_matrix[j, i] = float(len(intersection)) / float(len(union))
                else:
                    cooccurence_matrix[j, i] = 0
        return cooccurence_matrix

    # 產生前 10 筆推薦項目
    # 輸入：
    # - user：使用者 ID
    # - cooccurence_matrix：共現矩陣
    # - all_songs：訓練集中所有歌曲
    # - user_songs：該使用者已聽過的歌曲
    # 輸出：推薦結果 DataFrame
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))

        # 計算每首歌的相似度平均分數（橫向平均）
        user_sim_scores = cooccurence_matrix.sum(axis=0) / float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        # 根據相似度排序
        sort_index = sorted(((score, idx) for idx, score in enumerate(user_sim_scores)), reverse=True)

        # 建立推薦 DataFrame
        columns = ['user_id', 'song', 'score', 'rank']
        df = pandas.DataFrame(columns=columns)

        # 只保留未聽過的前 10 首
        rank = 1
        for score, idx in sort_index:
            song = all_songs[idx]
            if ~np.isnan(score) and song not in user_songs and rank <= 10:
                df.loc[len(df)] = [user, song, score, rank]
                rank += 1

        if df.shape[0] == 0:
            print("該使用者無足夠資料進行推薦。")
            return -1
        else:
            return df

    # 建立模型（儲存資料與欄位）
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    # 對指定使用者進行推薦
    # 輸入：user（使用者 ID）
    # 輸出：推薦結果 DataFrame
    def recommend(self, user):
        user_songs = self.get_user_items(user)
        print("使用者互動過的歌曲數量：%d" % len(user_songs))

        all_songs = self.get_all_items_train_data()
        print("訓練集歌曲總數：%d" % len(all_songs))

        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        return self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

    # 對指定歌曲清單尋找相似歌曲
    # 輸入：item_list（歌曲清單）
    # 輸出：相似項目推薦 DataFrame
    def get_similar_items(self, item_list):
        user_songs = item_list
        all_songs = self.get_all_items_train_data()
        print("訓練集歌曲總數：%d" % len(all_songs))

        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        return self.generate_top_recommendations("", cooccurence_matrix, all_songs, user_songs)
