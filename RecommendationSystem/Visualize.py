# -*- coding: utf-8 -*-
"""
本程式用來對推薦系統的預處理資料進行統計與視覺化分析，包含：
顯示最受歡迎的歌曲前 20 名
顯示播放總量最高的歌手前 20 名
顯示最受歡迎的專輯前 20 名
繪製使用者播放不同歌曲數的分布直方圖
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定繪圖使用的字型為微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 設定負號可以正確顯示


def show_popular_songs(data):
    '''
    Function: show_popular_songs
    用途：顯示最受歡迎的歌曲前 20 名
    輸入：triplet_dataset_sub_song_merged（已處理合併的播放資料）
    輸出：直條圖顯示歌曲與播放量
    '''
    popular_songs = data[['title','listen_count']].groupby('title').sum().reset_index()
    popular_songs_top_20 = popular_songs.sort_values('listen_count', ascending=False).head(20)

    objects = list(popular_songs_top_20['title'])
    y_pos = np.arange(len(objects))
    performance = list(popular_songs_top_20['listen_count'])

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('播放次數')
    plt.title('最受歡迎的歌曲前 20 名')
    plt.tight_layout()
    plt.show()


def show_popular_artists(data):
    '''
    Function: show_popular_artists
    用途：顯示播放總量最高的歌手前 20 名
    輸入：triplet_dataset_sub_song_merged（已處理合併的播放資料）
    輸出：直條圖顯示歌手與總播放量
    '''
    popular_artist = data[['artist_name','listen_count']].groupby('artist_name').sum().reset_index()
    popular_artist_top_20 = popular_artist.sort_values('listen_count', ascending=False).head(20)

    objects = list(popular_artist_top_20['artist_name'])
    y_pos = np.arange(len(objects))
    performance = list(popular_artist_top_20['listen_count'])

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('播放次數')
    plt.title('最受歡迎的歌手前 20 名')
    plt.tight_layout()
    plt.show()


def show_popular_releases(data):
    '''
    Function: show_popular_releases
    用途：顯示最受歡迎的專輯前 20 名
    輸入：data → 合併後的資料表（DataFrame）
    輸出：直條圖顯示專輯播放總量
    '''
    popular_release = data[['release','listen_count']].groupby('release').sum().reset_index()
    popular_release_top_20 = popular_release.sort_values('listen_count', ascending=False).head(20)

    objects = list(popular_release_top_20['release'])
    y_pos = np.arange(len(objects))
    performance = list(popular_release_top_20['listen_count'])

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('播放次數')
    plt.title('最受歡迎的專輯前 20 名')
    plt.tight_layout()
    plt.show()


def show_user_distribution(data):
    '''
    Function: show_user_distribution
    用途：繪製使用者播放不同歌曲數的分布直方圖
    輸入：triplet_dataset_sub_song_merged（已處理合併的播放資料）
    輸出：直方圖
    '''
    user_song_count = data[['user', 'title']].groupby('user').count().reset_index()
    user_song_count = user_song_count.sort_values(by='title', ascending=False)

    print("使用者每人播放歌曲數的描述統計如下：")
    print(user_song_count['title'].describe())

    x = user_song_count['title']
    n, bins, patches = plt.hist(x, bins=50, facecolor='green', alpha=0.75)

    plt.xlabel('使用者播放過的不同歌曲數')
    plt.ylabel('使用者人數')
    plt.title('使用者播放歌曲數分布直方圖')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # 設定資料目錄路徑
    data_home = './RecommendationSystem/'

    # 嘗試載入事先處理好的合併資料集
    if os.path.exists(f'{data_home}triplet_dataset_sub_song_merged.csv'):
        triplet_dataset_sub_song_merged = pd.read_csv(f'{data_home}triplet_dataset_sub_song_merged.csv')
    else:
        raise FileNotFoundError("錯誤：資料檔案不存在，請先執行 Preprocess.py 來產生 triplet_dataset_sub_song_merged.csv")

    # 顯示最受歡迎的歌曲前 20 名
    show_popular_songs(triplet_dataset_sub_song_merged)
    # 顯示播放總量最高的歌手前 20 名
    show_popular_artists(triplet_dataset_sub_song_merged)
    # 顯示最受歡迎的專輯前 20 名
    show_popular_releases(triplet_dataset_sub_song_merged)
    # 繪製使用者播放不同歌曲數的分布直方圖
    show_user_distribution(triplet_dataset_sub_song_merged)
