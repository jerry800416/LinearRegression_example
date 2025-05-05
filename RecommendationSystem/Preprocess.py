# -*- coding: utf-8 -*-
"""
本程式碼為 Million Song Dataset (MSD) 的資料前處理階段，
包含以下任務：
1. 載入原始三元組資料（user, song, play_count）
2. 統計每位使用者與每首歌曲的總播放量，並各自儲存為 CSV
3. 篩選前 10 萬名活躍用戶與前 3 萬首熱門歌曲
4. 擷取包含上述子集的資料列，另存為 subset 資料集
5. 將歌曲中對應的 metadata（歌手、曲名、專輯等）合併
6. 移除冗餘欄位，輸出乾淨合併後的資料集
"""
import pandas as pd
import numpy as np
import os
import sqlite3
import requests
import zipfile
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")



def download_file_with_progress(url, save_path):
    """
    使用 requests + tqdm 顯示進度條下載檔案
    """
    if os.path.exists(save_path):
        print(f"檔案已存在：{save_path}")
        return

    try:
        print(f"開始下載：{url}")
        with requests.get(url, stream=True, verify=False) as r:
            r.raise_for_status()  # 如果 HTTP 錯誤會 raise Exception
            total = int(r.headers.get('content-length', 0))
            with open(save_path, 'wb') as f, tqdm(
                desc=save_path,
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                ncols=80
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        print(f"已下載到：{save_path}")
    except Exception as e:
        print(f"下載失敗：{e}")


def compute_user_play_counts(filepath):
    """
    統計每位使用者的播放總次數
    輸入: 使用者-歌曲-播放量的檔案路徑
    輸出: DataFrame 包含 'user' 與其對應的 'play_count'
    """
    output_dict = {}
    with open(filepath) as f:
        for line in f:
            user = line.split('\t')[0]
            play_count = int(line.split('\t')[2])
            if user in output_dict:
                play_count += output_dict[user]
            output_dict[user] = play_count
    user_play_df = pd.DataFrame([{'user': k, 'play_count': v} for k, v in output_dict.items()])
    user_play_df = user_play_df.sort_values(by='play_count', ascending=False)
    return user_play_df


def compute_song_play_counts(filepath):
    """
    統計每首歌曲的播放總次數
    輸入: 使用者-歌曲-播放量的檔案路徑
    輸出: DataFrame 包含 'song' 與其對應的 'play_count'
    """
    output_dict = {}
    with open(filepath) as f:
        for line in f:
            song = line.split('\t')[1]
            play_count = int(line.split('\t')[2])
            if song in output_dict:
                play_count += output_dict[song]
            output_dict[song] = play_count
    song_play_df = pd.DataFrame([{'song': k, 'play_count': v} for k, v in output_dict.items()])
    song_play_df = song_play_df.sort_values(by='play_count', ascending=False)
    return song_play_df



if __name__ == "__main__":


    # ===== 0. 下載資料集 =====
    # 下載 track_metadata.db
    download_file_with_progress(
        "https://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/track_metadata.db",
        "RecommendationSystem/track_metadata.db"
    )

    # 下載並解壓 train_triplets.txt.zip
    zip_path = "RecommendationSystem/train_triplets.txt.zip"
    txt_path = "RecommendationSystem/train_triplets.txt"

    download_file_with_progress(
        "http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip",
        zip_path
    )

    if not os.path.exists(txt_path) and os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("RecommendationSystem/")
            print("train_triplets.txt 已成功解壓")
        except Exception as e:
            print(f"🚫 解壓縮錯誤：{e}")
    else:
        print("train_triplets.txt 已存在，略過解壓")

    # ===== 1. 讀取原始使用者-歌曲-播放量資料 =====
    
    # 設定資料目錄路徑
    data_home = './RecommendationSystem/'

    # 讀取三欄格式的使用者播放資料：user_id, song_id, play_count
    triplet_dataset = pd.read_csv(f'{data_home}train_triplets.txt',
                                sep='\t', header=None,
                                names=['user', 'song', 'play_count'])
    
    print("原始資料筆數與欄位數:", triplet_dataset.shape)
    print("資料結構摘要:")
    print(triplet_dataset.info())
    print("前 10 筆原始播放紀錄:")
    print(triplet_dataset.head(10))

    # ===== 2. 統計每位使用者的總播放次數 =====
    # 如果 user_playcount_df.csv 已存在，則不再重複計算
    if os.path.exists(f'{data_home}user_playcount_df.csv'):
        play_count_df = pd.read_csv(f'{data_home}user_playcount_df.csv')
    else:
        play_count_df = compute_user_play_counts(f'{data_home}train_triplets.txt')
        play_count_df.to_csv(f'{data_home}user_playcount_df.csv', index=False)

    print("播放次數最多的前 10 名使用者:")
    print(play_count_df.head(10))

    # ===== 3. 統計每首歌曲的總播放次數 =====
    # 如果 song_playcount_df.csv 已存在，則不再重複計算
    if os.path.exists(f'{data_home}song_playcount_df.csv'):
        song_count_df = pd.read_csv(f'{data_home}song_playcount_df.csv')
    else:
        song_count_df = compute_song_play_counts(f'{data_home}train_triplets.txt')
        song_count_df.to_csv(f'{data_home}song_playcount_df.csv', index=False)

    print("播放次數最多的前 10 首歌曲:")
    print(song_count_df.head(10))

    # ===== 4. 擷取前 10 萬活躍用戶與前 3 萬熱門歌曲作為子集 =====

    total_play_count = song_count_df['play_count'].sum()
    
    print("前 10 萬名使用者佔總播放比例 (%):",(play_count_df.head(100000)['play_count'].sum() / total_play_count) * 100)
    
    print("前 3 萬首歌曲佔總播放比例 (%):",(song_count_df.head(30000)['play_count'].sum() / total_play_count) * 100)

    user_subset = list(play_count_df.head(100000)['user'])
    song_subset = list(song_count_df.head(30000)['song'])

    # ===== 5. 過濾原始資料，只保留 subset 中的使用者與歌曲 =====
    triplet_dataset = pd.read_csv(f'{data_home}train_triplets.txt', sep='\t',
                                header=None, names=['user', 'song', 'play_count'])
    triplet_dataset_sub = triplet_dataset[triplet_dataset['user'].isin(user_subset)]
    triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub['song'].isin(song_subset)]
    # 如果 triplet_dataset_sub_song.csv 已存在，則不再重複計算
    if not os.path.exists(f'{data_home}triplet_dataset_sub_song.csv'):
        triplet_dataset_sub_song.to_csv(f'{data_home}triplet_dataset_sub_song.csv', index=False)

    print("過濾後的播放紀錄筆數與欄位數:", triplet_dataset_sub_song.shape)
    print("過濾後的播放資料前 10 筆:")
    print(triplet_dataset_sub_song.head(10))

    # ===== 6. 合併歌曲的 metadata 資料 =====
    conn = sqlite3.connect(f'{data_home}track_metadata.db')
    track_metadata_df = pd.read_sql('SELECT * FROM songs', con=conn)
    track_metadata_df_sub = track_metadata_df[track_metadata_df['song_id'].isin(song_subset)]
    # 如果 track_metadata_df_sub.csv 已存在，則不再重複計算
    if not os.path.exists(f'{data_home}track_metadata_df_sub.csv'):
        track_metadata_df_sub.to_csv(f'{data_home}track_metadata_df_sub.csv', index=False)
    
    print("歌曲 metadata 子集筆數與欄位數:", track_metadata_df_sub.shape)

    # ===== 7. 清理與合併資料 =====
    track_metadata_df_sub.drop(columns=['track_id', 'artist_mbid'], inplace=True)
    track_metadata_df_sub = track_metadata_df_sub.drop_duplicates(subset=['song_id'])

    print("播放資料集 (合併前) 前幾筆:")
    print(triplet_dataset_sub_song.head())
    print("歌曲中繼資料集 (合併前) 前幾筆:")
    print(track_metadata_df_sub.head())

    # 合併播放資料與 metadata
    triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song,track_metadata_df_sub,how='left', left_on='song', right_on='song_id')
    triplet_dataset_sub_song_merged.rename(columns={'play_count': 'listen_count'}, inplace=True)

    # 去除不必要欄位
    cols_to_drop = ['song_id', 'artist_id', 'duration',
                    'artist_familiarity', 'artist_hotttnesss',
                    'track_7digitalid', 'shs_perf', 'shs_work']
    triplet_dataset_sub_song_merged.drop(columns=cols_to_drop, inplace=True)

    # 如果 triplet_dataset_sub_song_merged.csv 已存在，則不再重複存檔
    if not os.path.exists(f'{data_home}triplet_dataset_sub_song_merged.csv'):
        triplet_dataset_sub_song_merged.to_csv(f'{data_home}triplet_dataset_sub_song_merged.csv', index=False)

    # 最終處理後資料
    print("整合後的資料前 10 筆:")
    print(triplet_dataset_sub_song_merged.head(10))