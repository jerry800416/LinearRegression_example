# -*- coding: utf-8 -*-
"""
本程式主要實作 3 種推薦演算法：
1. 排行榜式推薦：適用於新用戶（解決冷啟動問題）
2. 基於項目相似度的推薦：根據使用者喜好推薦相似歌曲
3. 基於 SVD 矩陣分解的推薦：利用潛在特徵推估推薦分數
"""

from sklearn.model_selection import train_test_split
import Recommenders as Recommenders
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
import math as mt
import os


def create_popularity_recommendation(train_data, user_id, item_id):
    '''
    函數名稱：create_popularity_recommendation
    功能：根據某一欄位（如歌曲、歌手、專輯）的總播放次數，建立排行榜推薦結果
    輸入：
        - train_data: 訓練資料（DataFrame）
        - user_id: 使用者欄位名稱（字串）
        - item_id: 欲統計的項目欄位（字串，如 'title'、'artist_name' 等）
    輸出：
        - popularity_recommendations: 前 20 名推薦項目（DataFrame）
    '''
    train_data_grouped = train_data.groupby([item_id]).agg({user_id: 'count'}).reset_index()
    train_data_grouped.rename(columns={user_id: 'score'}, inplace=True)
    train_data_sort = train_data_grouped.sort_values(['score', item_id], ascending=[False, True])
    train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=False, method='first')
    popularity_recommendations = train_data_sort.head(20)
    return popularity_recommendations


def compute_svd(urm, K):
    '''
    函數名稱：compute_svd
    功能：對使用者-物品稀疏矩陣進行 SVD 分解
    輸入：
        - urm: 使用者-物品稀疏矩陣 (scipy coo_matrix)
        - K: 分解維度（整數）
    輸出：
        - U, S, Vt: 分解後的三個矩陣（均為 scipy csc_matrix）
    '''
    U, s, Vt = svds(urm, K)
    S = np.diag(np.sqrt(s)).astype(np.float32)
    return csc_matrix(U), csc_matrix(S), csc_matrix(Vt)


def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test):
    '''
    函數名稱：compute_estimated_matrix
    功能：對指定的使用者進行推薦分數預測，並擷取推薦前 250 筆
    輸入：
        - urm: 原始使用者-物品稀疏矩陣
        - U, S, Vt: SVD 分解後的矩陣
        - uTest: 欲預測的使用者 index（整數列表）
        - K: 分解維度（整數）
        - test: 是否為測試（布林，未使用）
    輸出：
        - recomendRatings: 每位使用者對應前 250 筆推薦項目索引（np.ndarray）
    '''
    rightTerm = S @ Vt
    max_recommendation = 250
    MAX_UID, MAX_PID = urm.shape
    estimatedRatings = np.zeros((MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros((MAX_UID, max_recommendation), dtype=np.int32)
    
    for userTest in uTest:
        prod = U[userTest, :] @ rightTerm
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    
    return recomendRatings




if __name__ == "__main__":

    data_home = './RecommendationSystem/'

    # 讀取推薦資料集
    merged_path = os.path.join(data_home, 'triplet_dataset_sub_song_merged.csv')
    if os.path.exists(merged_path):
        triplet_dataset_sub_song_merged = pd.read_csv(merged_path)
    else:
        raise FileNotFoundError("錯誤：找不到 triplet_dataset_sub_song_merged.csv，請先執行 Preprocess.py")

    # 讀取歌曲與使用者播放次數
    song_count_df = pd.read_csv(os.path.join(data_home, 'song_playcount_df.csv'))
    play_count_df = pd.read_csv(os.path.join(data_home, 'user_playcount_df.csv'))

    # 顯示部分訓練資料與播放比例資訊
    total_play_count = sum(song_count_df.play_count)
    print("前 10 萬名使用者佔總播放量比例（%）:", (play_count_df.head(100000).play_count.sum() / total_play_count) * 100)
    play_count_subset = play_count_df.head(100000)

    '''
    排行榜推薦
    '''
    recommendations = create_popularity_recommendation(triplet_dataset_sub_song_merged, 'user', 'title')
    print("排行榜推薦前 20 筆（依歌曲）:\n", recommendations)


    '''
    相似度推薦
    '''
    # 相似度推薦資料準備
    song_count_subset = song_count_df.head(5000)
    song_subset = list(song_count_subset.song)
    triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[
        triplet_dataset_sub_song_merged.song.isin(song_subset)
    ]
    print("過濾後的相似度推薦資料（前幾筆）:\n", triplet_dataset_sub_song_merged_sub.head())

    # 相似度推薦訓練與推薦
    train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_sub, test_size=0.3, random_state=0)
    is_model = Recommenders.item_similarity_recommender_py()
    is_model.create(train_data, 'user', 'title')
    user_id = list(train_data.user)[7]
    print("正在為使用者：", user_id, "執行基於項目相似度的推薦")
    is_model.recommend(user_id)


    '''
    SVD 矩陣分解的推薦
    '''
    # 基於矩陣分解的推薦前處理
    user_total_df = triplet_dataset_sub_song_merged.groupby('user')['listen_count'].sum().reset_index()
    user_total_df.rename(columns={'listen_count': 'total_listen_count'}, inplace=True)
    triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song_merged, user_total_df)
    triplet_dataset_sub_song_merged['fractional_play_count'] = (
        triplet_dataset_sub_song_merged['listen_count'] / triplet_dataset_sub_song_merged['total_listen_count']
    )

    # 建立使用者與歌曲編碼表
    small_set = triplet_dataset_sub_song_merged.copy()
    user_codes = small_set.user.drop_duplicates().reset_index().rename(columns={'index': 'user_index'})
    song_codes = small_set.song.drop_duplicates().reset_index().rename(columns={'index': 'song_index'})
    user_codes['us_index_value'] = list(user_codes.index)
    song_codes['so_index_value'] = list(song_codes.index)
    small_set = pd.merge(small_set, song_codes, how='left')
    small_set = pd.merge(small_set, user_codes, how='left')

    # 建立稀疏矩陣
    mat_candidate = small_set[['us_index_value', 'so_index_value', 'fractional_play_count']]
    data_array = mat_candidate.fractional_play_count.values
    row_array = mat_candidate.us_index_value.values
    col_array = mat_candidate.so_index_value.values
    data_sparse = coo_matrix((data_array, (row_array, col_array)), dtype=float)

    print("稀疏矩陣建立成功，維度為:", data_sparse.shape)

    # 執行 SVD 分解與推薦
    K = 50
    urm = data_sparse
    MAX_UID, MAX_PID = urm.shape
    U, S, Vt = compute_svd(urm, K)
    
    # 多名用戶示範推薦
    uTest = [4, 5, 6, 7, 8, 873, 23, 27513]
    uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, test=True)
    for user in uTest:
        print(f"使用者 {user} 的前 10 筆 SVD 推薦：")
        for rank, i in enumerate(uTest_recommended_items[user, :10], 1):
            song_row = small_set[small_set.so_index_value == i].drop_duplicates('so_index_value')
            title = song_row['title'].values[0]
            artist = song_row['artist_name'].values[0]
            print(f"推薦 {rank}: {title} by {artist}")