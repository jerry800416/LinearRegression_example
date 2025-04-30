# -*- coding: UTF-8 -*-
# 匯入必要的函式庫
import pandas as pd  # 用於處理表格資料
from mlxtend.preprocessing import TransactionEncoder  # 用來將交易資料轉換為 One-Hot 編碼
from mlxtend.frequent_patterns import apriori, association_rules  # Apriori 與規則挖掘工具

# 建立交易資料集，每一筆交易為一個商品清單（即一筆顧客購物紀錄）
dataset = [
    ['牛奶', '麵包', '奶油'],
    ['牛奶', '麵包'],
    ['牛奶'],
    ['麵包', '奶油'],
    ['牛奶', '麵包', '奶油']
]

# ---------- 步驟一：資料轉換為 One-Hot 編碼形式 ----------
# 每個商品將被轉換為一個欄位，若某筆交易中包含該商品，則該欄位為 True，否則為 False

te = TransactionEncoder()                   # 建立 TransactionEncoder 物件
te_ary = te.fit(dataset).transform(dataset) # 將交易資料轉為布林矩陣（True/False）
df = pd.DataFrame(te_ary, columns=te.columns_)  # 建立 pandas DataFrame 表格格式

print("🔹 One-Hot 編碼後的交易資料：")
print(df)

# ---------- 步驟二：找出頻繁項目集 ----------
# 使用 Apriori 演算法，找出同時出現次數超過 min_support 的項目組合
# 例如 min_support=0.5 表示：至少出現在 50% 以上的交易中

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

print("\n🔹 頻繁項目集（support ≥ 0.5）：")
print(frequent_itemsets)

# ---------- 步驟三：根據頻繁項目集產生關聯規則 ----------
# 並計算規則的三個重要指標：support, confidence, lift
# min_threshold=0.5 表示信賴度至少要有 50% 才被考慮

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# 選擇我們有興趣的欄位顯示：前項(antecedents)、後項(consequents)、三大指標
print("\n🔹 關聯規則分析結果：")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
