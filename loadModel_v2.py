import datetime
import time
import warnings
from datetime import datetime
from datetime import timedelta
from pickle import load

import ccxt
import numpy as np
import pandas as pd
import pyupbit
from tensorflow import keras
from tqdm import tqdm

warnings.filterwarnings('ignore')
np.random.seed(42)

def make_dataset(data, window_size=20, rangeY = 2):
    feature_list = []
    label_list = []
    buy = []
    sell = []
    feature = [
        'hour', 'minute',
        'open', 'low', 'high', 'close', 'volume',
        'cci', 'mfi10', 'fast_k', 'fast_d',
        'slow_k', 'slow_d', 'rsi', 'ma5',
        'ma10', 'ma15', 'ma20',

        'b_open', 'b_low', 'b_high', 'b_close', 'b_volume',
        'b_cci', 'b_mfi10', 'b_fast_k', 'b_fast_d',
        'b_slow_k', 'b_slow_d', 'b_rsi', 'b_ma5',
        'b_ma10', 'b_ma15', 'b_ma20'
    ]
    upbit_feature = ['open', 'low', 'high', 'close', 'ma5', 'ma10', 'ma15', 'ma20']
    binance_feature = ['b_open', 'b_low', 'b_high', 'b_close', 'b_ma5', 'b_ma10', 'b_ma15', 'b_ma20']

    for i in tqdm(range(1, len(data) - (window_size + rangeY))):
        try:
            x = data[i-1:i+window_size][feature]
            if x.isnull().values.any():
                print(str(i) + " isNan")
                pass

            for j in range(1, window_size + 1):
                date = datetime.strptime(str(data.index[i - 1])[:16], "%Y-%m-%d %H:%M") + timedelta(minutes=j)
                t = data.loc[str(date)]
            for j in range(1, rangeY):
                date = datetime.strptime(str(data.index[i + window_size])[:16], "%Y-%m-%d %H:%M") + timedelta(minutes=j)
                t = data.loc[str(date)]
            if True in np.array(x['close'][-1] * 1.007 < data[i + window_size:i + (window_size + rangeY)]['close']):
                y = 2
            elif True in np.array(
                    x['close'][-1] * 0.995 >= data[i + window_size:i + (window_size + rangeY)]['close'][-1]):
                y = 1
            else:
                y = 0
            buy.append(x['close'][-1])
            sell.append(data[i+window_size:i+(window_size + rangeY)]['close'][-1])
            x[upbit_feature] = (x[upbit_feature] / x[upbit_feature].shift(1) - 1) * 100
            x[binance_feature] = (x[binance_feature] / x[binance_feature].shift(1) - 1) * 100
            label_list.append(y)
            feature_list.append(x[1:].values)
        except Exception as e:
            print(e)
            pass
    return np.array(feature_list), np.array(label_list), np.array(buy), np.array(sell)


model = keras.models.load_model("crypto_model2")

day_count = 400
date = None
coin = "ETC"
interval = "minute1"
dfs = []
for i in tqdm(range(day_count // 200 + 1)):
    try:
        if i < day_count // 200:
            df = pyupbit.get_ohlcv("KRW-"+coin, to=date, interval=interval)
            date = df.index[0]
        elif day_count % 200 != 0:
            df = pyupbit.get_ohlcv("KRW-"+coin, to=date, interval=interval, count=day_count % 200)
        else:
            break
        dfs.append(df)
        time.sleep(0.1)
    except Exception:
        break
df = pd.concat(dfs).sort_index()

binance = ccxt.binance()
binance_date = int((datetime.strptime(str(df.index[0])[:16], "%Y-%m-%d %H:%M")).timestamp() * 1000)
bdfs = []

for i in tqdm(range(day_count // 1000 + 1)):
    ohlcvs = None
    if i < day_count // 1000:
        ohlcvs = binance.fetch_ohlcv(coin + '/USDT', timeframe="1m", since=binance_date, limit=1000)
    elif day_count % 1000 != 0:
        ohlcvs = binance.fetch_ohlcv(coin + '/USDT', timeframe="1m", since=binance_date, limit=day_count % 1000)
    else:
        break
    bdf = pd.DataFrame(ohlcvs, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    bdf['time'] = bdf['datetime']
    bdf['datetime'] = pd.to_datetime(bdf['datetime'], unit="ms")
    bdf.set_index('datetime', inplace=True)
    bdfs.append(bdf)
    binance_date = bdf['time'][-1] + 1
    time.sleep(0.2)
bdf = pd.concat(bdfs).sort_index()

df['b_high'] = None
df['b_low'] = None
df['b_open'] = None
df['b_close'] = None
df['b_volume'] = None

start = 0
for i in tqdm(range(len(df))):
    try:
        upbitDate = datetime.strptime(str(df.index[i])[:16], "%Y-%m-%d %H:%M") - timedelta(hours=9)
        binance = bdf.loc[str(upbitDate)]
        df['b_high'][i] = binance['high']
        df['b_low'][i] = binance['low']
        df['b_close'][i] = binance['close']
        df['b_open'][i] = binance['open']
        df['b_volume'][i] = binance['volume']
    except Exception as e:
        print(e)
        pass

df = df.dropna()

df['tp'] = (df['high'] + df['low'] + df['close']) / 3

df['b_tp'] = (df['b_high'] + df['b_low'] + df['b_close']) / 3

df['sma'] = df['tp'].rolling(20).mean()
df['mad'] = df['tp'].rolling(20).apply(lambda x: pd.Series(x).mad())
df['cci'] = (df['tp'] - df['sma']) / (0.015 * df['mad'])

df['b_sma'] = df['b_tp'].rolling(20).mean()
df['b_mad'] = df['b_tp'].rolling(20).apply(lambda x: pd.Series(x).mad())
df['b_cci'] = (df['b_tp'] - df['b_sma']) / (0.015 * df['b_mad'])

df['pmf'] = 0
df['nmf'] = 0
for i in range(len(df['close']) - 1):
    # 당일의 중심가격이 전일의 중심가격보다 크면 긍정적 현금흐름
    if df['tp'].values[i] < df['tp'].values[i + 1]:
        df['pmf'].values[i + 1] = df['tp'].values[i + 1] * df['volume'].values[i + 1]
        df['nmf'].values[i + 1] = 0
    # 당일의 중심가격이 전일의 중심가격보다 작거나 같으면 부정적 현금흐름
    else:
        df['nmf'].values[i + 1] = df['tp'].values[i + 1] * df['volume'].values[i + 1]
        df['pmf'].values[i + 1] = 0

df['mfr'] = df['pmf'].rolling(window=10).sum() / df['nmf'].rolling(window=10).sum()
df['mfi10'] = 100 - 100 / (1 + df['mfr'])

df['b_pmf'] = 0
df['b_nmf'] = 0
for i in range(len(df['b_close']) - 1):
    # 당일의 중심가격이 전일의 중심가격보다 크면 긍정적 현금흐름
    if df['b_tp'].values[i] < df['b_tp'].values[i + 1]:
        df['b_pmf'].values[i + 1] = df['b_tp'].values[i + 1] * df['b_volume'].values[i + 1]
        df['b_nmf'].values[i + 1] = 0
    # 당일의 중심가격이 전일의 중심가격보다 작거나 같으면 부정적 현금흐름
    else:
        df['b_nmf'].values[i + 1] = df['b_tp'].values[i + 1] * df['b_volume'].values[i + 1]
        df['b_pmf'].values[i + 1] = 0

df['b_mfr'] = df['b_pmf'].rolling(window=10).sum() / df['b_nmf'].rolling(window=10).sum()
df['b_mfi10'] = 100 - 100 / (1 + df['b_mfr'])

df['n_high'] = df['high'].rolling(window=14, min_periods=1).max()
df['n_low'] = df['low'].rolling(window=14, min_periods=1).min()
df['fast_k'] = ((df['close'] - df['n_low']) / (df['n_high'] - df['n_low'])) * 100
df['fast_d'] = df['fast_k'].rolling(window=5).mean()
df['slow_k'] = df['fast_k'].ewm(span=5).mean()
df['slow_d'] = df['slow_k'].ewm(span=5).mean()

df['b_n_high'] = df['b_high'].rolling(window=14, min_periods=1).max()
df['b_n_low'] = df['b_low'].rolling(window=14, min_periods=1).min()
df['b_fast_k'] = ((df['b_close'] - df['b_n_low']) / (df['b_n_high'] - df['b_n_low'])) * 100
df['b_fast_d'] = df['b_fast_k'].rolling(window=5).mean()
df['b_slow_k'] = df['b_fast_k'].ewm(span=5).mean()
df['b_slow_d'] = df['b_slow_k'].ewm(span=5).mean()

window = 14
adjust = False
delta = df['close'].diff(1).dropna()
loss = delta.copy()
gains = delta.copy()

gains[gains < 0] = 0
loss[loss > 0] = 0

gain_ewm = gains.ewm(com=window - 1, adjust=adjust).mean()
loss_ewm = abs(loss.ewm(com=window - 1, adjust=adjust).mean())

RS = gain_ewm / loss_ewm
RSI = 100 - 100 / (1 + RS)

df['rsi'] = RSI

window = 14
adjust = False
b_delta = df['b_close'].diff(1).dropna()
b_loss = b_delta.copy()
b_gains = b_delta.copy()

b_gains[b_gains < 0] = 0
b_loss[b_loss > 0] = 0

b_gain_ewm = b_gains.ewm(com=window - 1, adjust=adjust).mean()
b_loss_ewm = abs(b_loss.ewm(com=window - 1, adjust=adjust).mean())

b_RS = b_gain_ewm / b_loss_ewm
b_RSI = 100 - 100 / (1 + b_RS)

df['b_rsi'] = b_RSI

df['ma5'] = df['close'].rolling(window=5).mean().shift(1)
df['ma10'] = df['close'].rolling(window=10).mean().shift(1)
df['ma15'] = df['close'].rolling(window=15).mean().shift(1)
df['ma20'] = df['close'].rolling(window=20).mean().shift(1)

df['b_ma5'] = df['b_close'].rolling(window=5).mean().shift(1)
df['b_ma10'] = df['b_close'].rolling(window=10).mean().shift(1)
df['b_ma15'] = df['b_close'].rolling(window=15).mean().shift(1)
df['b_ma20'] = df['b_close'].rolling(window=20).mean().shift(1)

df['hour'] = df.index.map(lambda x: datetime.strptime(str(x)[:16], "%Y-%m-%d %H:%M").hour)
df['minute'] = df.index.map(lambda x: datetime.strptime(str(x)[:16], "%Y-%m-%d %H:%M").minute)

data = df.dropna()
X, Y, buy, sell = make_dataset(data, 180, 60)

shape1 = X.shape[1]
shape2 = X.shape[2]
X = np.reshape(X, (X.shape[0], shape1 * shape2))
scaler = load(open('./crypto_scaler2.pkl', 'rb'))
scale_x = scaler.transform(X)
scale_x = np.reshape(scale_x, (scale_x.shape[0], shape1, shape2))

result = model.predict(scale_x)
max = np.argmax(result, axis=1)
init_money = 1000000
money = 1000000
amount = 0
win = 0
lose = 0

for i in range(len(result) - 1):
    if max[i] == 2 and money * 0.2 >= 5000:
        amount += 1
        b = buy[i]
        s = sell[i]

        money -= init_money * 0.2
        balance = init_money * 0.2 / b * 0.9985

        if Y[i] == 1:
            win += 1
            money += ((b * 1.007) * balance * 0.9985)
            print("win", result[i][0], result[i][1], result[i][2])
        else:
            lose += 1
            money += (s * balance * 0.9985)
            print("lose", result[i][0], result[i][1], result[i][2])

        init_money = money
print(amount)
print(win)
print(lose)
print(init_money)
