# -*- coding: UTF-8 -*-
# 匯入必要的函式庫
import pandas as pd  # 用於處理表格資料
from mlxtend.frequent_patterns import apriori, association_rules  # 用於找出頻繁項目集與產生關聯規則

# ---------- 步驟一：讀取 MovieLens 資料集 ----------
# 假設你已下載 'ml-latest-small/movies.csv'，這個檔案包含 movieId、title、genres

movies = pd.read_csv('AssociationRule/movies.csv')  # 載入電影資料

# ---------- 步驟二：將 genres 欄位轉換為 One-Hot 編碼 ----------
# genres 欄位是一個以 '|' 分隔的字串，例如 "Action|Adventure|Sci-Fi"
# 我們先移除原始 genres 欄位，改用 get_dummies 分割為多個類別欄位（每種類型一欄）

movies_ohe = movies.drop('genres', axis=1).join(movies['genres'].str.get_dummies(sep='|'))

# 為了方便操作，設定 movieId 與 title 為索引
movies_ohe.set_index(['movieId', 'title'], inplace=True)

# 若需要觀察轉換後的資料：
# pd.options.display.max_columns = 100
# print(movies_ohe.head())

# ---------- 步驟三：使用 Apriori 找出頻繁項目集 ----------
# 這裡的「項目」是指電影的 genre 標籤（如：Comedy、Action、Drama 等）
# min_support=0.025 表示該類型的組合至少要出現在 2.5% 的電影中才會被視為頻繁

frequent_itemsets_movies = apriori(movies_ohe, use_colnames=True, min_support=0.025)

# ---------- 步驟四：建立關聯規則，並以提升度（Lift）為衡量指標 ----------
# metric='lift' 表示我們關心的關聯性強度
# min_threshold=1.25 表示提升度要大於 1.25 才會被納入（大於 1 代表正向關聯）

rules_movies = association_rules(frequent_itemsets_movies, metric='lift', min_threshold=1.25)

# ---------- 步驟五：篩選提升度較高的關聯規則 ----------
# 這裡我們只顯示 lift > 4 的規則，並根據 lift 值從高到低排序

strong_rules = rules_movies[rules_movies['lift'] > 4].sort_values(by='lift', ascending=False)

# 顯示結果（若想簡化只看關鍵欄位）
print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# ---------- 步驟六：補充應用：找出屬於兒童電影但非動畫的資料 ----------
# 這一行不是關聯規則的一部分，而是針對特定條件做的資料篩選
# 可用來探索規則的實際對應資料範例

children_not_animation = movies[
    (movies['genres'].str.contains('Children')) & 
    (~movies['genres'].str.contains('Animation'))
]

print("\n🔹 非動畫類的兒童電影（Children 且非 Animation）：")
print(children_not_animation.head())