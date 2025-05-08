# -*- coding: utf-8 -*-
"""
這段程式碼使用 Gaussian HMM 對金價時間序列資料進行建模，並根據模型的隱藏狀態進行短期預測。
輸入資料來自 CSV（包含日期、收盤價與成交量），模型將觀察變動與成交量並預測未來價格趨勢。
"""

import datetime
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt, cm
plt.rc('font', family='Microsoft JhengHei') # 設定字型為微軟正黑體（可顯示中文）
plt.rcParams['axes.unicode_minus'] = False  # 設定負號可以正確顯示
from matplotlib.dates import YearLocator, MonthLocator
from hmmlearn.hmm import GaussianHMM

# 設定標的名稱與資料時間區間
ticker = "gold"
start_date = datetime.date(2010, 1, 1)
end_date = datetime.date.today() - datetime.timedelta(days=15)

# ========================
# 讀取資料並預處理
# ========================

# 讀取資料（假設已包含欄位：date, open, high, low, close, volume, Adj Close）
data = pd.read_csv('HiddenMarkovModel/data2.csv', header=0)
df = pd.DataFrame(data)

# 將日期欄轉為 datetime 物件
df['date'] = pd.to_datetime(df['date'])
print("原始資料前五筆:\n", df.head())

# 移除不需要的欄位
df.reset_index(inplace=True, drop=False)
df.drop(['index', 'open', 'low', 'high', 'Adj Close'], axis=1, inplace=True)
print("移除多餘欄位後:\n", df.head())

# 將日期轉為 ordinal（可計算用的整數）
df['date'] = df['date'].apply(datetime.datetime.toordinal)
print("將日期轉為 ordinal:\n", df.head())

# 轉成 numpy 結構便於後續處理
df = list(df.itertuples(index=False, name=None))
dates = np.array([q[0] for q in df], dtype=int)
close_v = np.array([q[1] for q in df])
volume = np.array([q[2] for q in df])[1:]

# 收盤價變動
diff = np.diff(close_v)
print("收盤價變動差分（前10筆）:", diff[:10])

dates = dates[1:]
print("對應日期（前10筆）:", dates[:10])

close_v = close_v[1:]
print("剩餘收盤價 shape:", close_v.shape)

# 建立觀測特徵：差分 + 成交量
X = np.column_stack([diff, volume])

# ========================
# 模型訓練與隱藏狀態解碼
# ========================

print("開始訓練 Gaussian HMM 並解碼 ...", end="")
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(X)
hidden_states = model.predict(X)
print("完成！")

# 顯示轉移機率矩陣
print("隱藏狀態轉移機率矩陣 A:")
print(model.transmat_)

# 顯示每個狀態的均值與變異數（變異數為對角線）
print("每個狀態的平均與變異數:")
params = pd.DataFrame(columns=('State', 'Means', 'Variance'))
for i in range(model.n_components):
    params.loc[i] = [f"狀態 {i}", model.means_[i], np.diag(model.covars_[i])]
print(params.head())

# ========================
# 繪製不同隱藏狀態下的資料點分布
# ========================

fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True, figsize=(15, 15))
colours = cm.rainbow(np.linspace(0, 1, model.n_components))

for i, (ax, colour) in enumerate(zip(axs, colours)):
    mask = hidden_states == i
    ax.plot_date(dates[mask], close_v[mask], ".", c=colour)
    ax.set_title(f"{i} 號隱藏狀態")

    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.grid(True)

plt.show()

# ========================
# 預測未來 N 天的價格與成交量
# ========================

# 使用轉移機率 × 狀態均值，計算每個狀態的期望「變動 + 成交量」
expected_returns_and_volumes = np.dot(model.transmat_, model.means_)
expected_returns = expected_returns_and_volumes[:, 0]
expected_volumes = expected_returns_and_volumes[:, 1]

params = pd.concat([
    pd.Series(expected_returns),
    pd.Series(expected_volumes)
], axis=1)
params.columns = ['Returns', 'Volume']
print("每個狀態對應的預期變動與成交量:\n", params)

# 預測未來 N 天
lastN = 7
start_date = datetime.date.today() - datetime.timedelta(days=lastN * 2)
end_date = datetime.date.today()

dates = np.array([q[0] for q in df], dtype=int)

predicted_prices = []
predicted_dates = []
predicted_volumes = []
actual_volumes = []

for idx in range(lastN):
    state = hidden_states[-lastN + idx]
    current_price = df[-lastN + idx][1]
    volume = df[-lastN + idx][2]
    actual_volumes.append(volume)
    
    current_date = datetime.date.fromordinal(dates[-lastN + idx])
    predicted_date = current_date + datetime.timedelta(days=1)
    
    predicted_dates.append(predicted_date)
    predicted_prices.append(current_price + expected_returns[state])
    predicted_volumes.append(np.round(expected_volumes[state]))

# ========================
# 繪圖：實際與預測價格變化
# ========================

plt.figure(figsize=(15, 5), dpi=100)
plt.title(ticker + "：實際與預測價格", fontsize=14)
plt.plot(predicted_dates, close_v[-lastN:], label='實際')
plt.plot(predicted_dates, predicted_prices, label='預測')
plt.legend()
plt.show()

# ========================
# 繪圖：實際與預測成交量變化
# ========================

plt.figure(figsize=(15, 5), dpi=100)
plt.title(ticker + "：實際與預測成交量", fontsize=14)
plt.plot(predicted_dates, actual_volumes, label='實際')
plt.plot(predicted_dates, predicted_volumes, label='預測')
plt.legend()
plt.show()
