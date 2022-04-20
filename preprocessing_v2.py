import datetime
import warnings
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
np.random.seed(42)

def make_dataset(data, window_size=20, rangeY = 2):
    feature_list = []
    label_list = []
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
            for j in range(1, window_size+1):
                date = datetime.strptime(str(data.index[i-1][:16]), "%Y-%m-%d %H:%M") + timedelta(minutes=j)
                t = data.loc[str(date)]
            for j in range(1, rangeY):
                date = datetime.strptime(str(data.index[i+window_size][:16]), "%Y-%m-%d %H:%M") + timedelta(minutes=j)
                t = data.loc[str(date)]
            if True in np.array(x['close'][-1] * 1.007 < data[i+window_size:i+(window_size + rangeY)]['close']):
                y = 2
            elif True in np.array(x['close'][-1] * 0.995 >= data[i+window_size:i+(window_size + rangeY)]['close'][-1]):
                y = 1
            else:
                y = 0
            # y = True in np.array(x['close'][-1] * 1.005 < data[i+window_size:i+(window_size + rangeY)]['close'])
            x[upbit_feature] = (x[upbit_feature] / x[upbit_feature].shift(1) - 1) * 100
            x[binance_feature] = (x[binance_feature] / x[binance_feature].shift(1) - 1) * 100

            label_list.append(y)
            feature_list.append(x[1:].values)
        except Exception as e:
            pass
    return np.array(feature_list), np.array(label_list)


if __name__ == '__main__':
    result = []

    ticker = "ETC"
    dfs = []
    DEBUG = False

    try:
        dataset = pd.read_csv(ticker + "_data_1m.csv", index_col=0)

        train = dataset
        if DEBUG:
            train = train[-30000:]
        else:
            train = train[-150000:]

        train['hour'] = train.index.map(lambda x: datetime.strptime(str(x[:16]), "%Y-%m-%d %H:%M").hour)
        train['minute'] = train.index.map(lambda x: datetime.strptime(str(x[:16]), "%Y-%m-%d %H:%M").minute)

        train['tp'] = (train['high'] + train['low'] + train['close']) / 3

        train['b_tp'] = (train['b_high'] + train['b_low'] + train['b_close']) / 3

        train['sma'] = train['tp'].rolling(20).mean()
        train['mad'] = train['tp'].rolling(20).apply(lambda x: pd.Series(x).mad())
        train['cci'] = (train['tp'] - train['sma']) / (0.015 * train['mad'])

        train['b_sma'] = train['b_tp'].rolling(20).mean()
        train['b_mad'] = train['b_tp'].rolling(20).apply(lambda x: pd.Series(x).mad())
        train['b_cci'] = (train['b_tp'] - train['b_sma']) / (0.015 * train['b_mad'])

        train['pmf'] = 0
        train['nmf'] = 0
        for i in range(len(train['close']) - 1):
            # 당일의 중심가격이 전일의 중심가격보다 크면 긍정적 현금흐름
            if train['tp'].values[i] < train['tp'].values[i + 1]:
                train['pmf'].values[i + 1] = train['tp'].values[i + 1] * train['volume'].values[i + 1]
                train['nmf'].values[i + 1] = 0
            # 당일의 중심가격이 전일의 중심가격보다 작거나 같으면 부정적 현금흐름
            else:
                train['nmf'].values[i + 1] = train['tp'].values[i + 1] * train['volume'].values[i + 1]
                train['pmf'].values[i + 1] = 0

        train['mfr'] = train['pmf'].rolling(window=10).sum() / train['nmf'].rolling(window=10).sum()
        train['mfi10'] = 100 - 100 / (1 + train['mfr'])

        train['b_pmf'] = 0
        train['b_nmf'] = 0
        for i in range(len(train['b_close']) - 1):
            # 당일의 중심가격이 전일의 중심가격보다 크면 긍정적 현금흐름
            if train['b_tp'].values[i] < train['b_tp'].values[i + 1]:
                train['b_pmf'].values[i + 1] = train['b_tp'].values[i + 1] * train['b_volume'].values[i + 1]
                train['b_nmf'].values[i + 1] = 0
            # 당일의 중심가격이 전일의 중심가격보다 작거나 같으면 부정적 현금흐름
            else:
                train['b_nmf'].values[i + 1] = train['b_tp'].values[i + 1] * train['b_volume'].values[i + 1]
                train['b_pmf'].values[i + 1] = 0

        train['b_mfr'] = train['b_pmf'].rolling(window=10).sum() / train['b_nmf'].rolling(window=10).sum()
        train['b_mfi10'] = 100 - 100 / (1 + train['b_mfr'])

        train['n_high'] = train['high'].rolling(window=14, min_periods=1).max()
        train['n_low'] = train['low'].rolling(window=14, min_periods=1).min()
        train['fast_k'] = ((train['close'] - train['n_low']) / (train['n_high'] - train['n_low'])) * 100
        train['fast_d'] = train['fast_k'].rolling(window=5).mean()
        train['slow_k'] = train['fast_k'].ewm(span=5).mean()
        train['slow_d'] = train['slow_k'].ewm(span=5).mean()

        train['b_n_high'] = train['b_high'].rolling(window=14, min_periods=1).max()
        train['b_n_low'] = train['b_low'].rolling(window=14, min_periods=1).min()
        train['b_fast_k'] = ((train['b_close'] - train['b_n_low']) / (train['b_n_high'] - train['b_n_low'])) * 100
        train['b_fast_d'] = train['b_fast_k'].rolling(window=5).mean()
        train['b_slow_k'] = train['b_fast_k'].ewm(span=5).mean()
        train['b_slow_d'] = train['b_slow_k'].ewm(span=5).mean()

        window = 14
        adjust = False
        delta = train['close'].diff(1).dropna()
        loss = delta.copy()
        gains = delta.copy()

        gains[gains < 0] = 0
        loss[loss > 0] = 0

        gain_ewm = gains.ewm(com=window - 1, adjust=adjust).mean()
        loss_ewm = abs(loss.ewm(com=window - 1, adjust=adjust).mean())

        RS = gain_ewm / loss_ewm
        RSI = 100 - 100 / (1 + RS)

        train['rsi'] = RSI

        window = 14
        adjust = False
        b_delta = train['b_close'].diff(1).dropna()
        b_loss = b_delta.copy()
        b_gains = b_delta.copy()

        b_gains[b_gains < 0] = 0
        b_loss[b_loss > 0] = 0

        b_gain_ewm = b_gains.ewm(com=window - 1, adjust=adjust).mean()
        b_loss_ewm = abs(b_loss.ewm(com=window - 1, adjust=adjust).mean())

        b_RS = b_gain_ewm / b_loss_ewm
        b_RSI = 100 - 100 / (1 + b_RS)

        train['b_rsi'] = b_RSI

        train['ma5'] = train['close'].rolling(window=5).mean().shift(1)
        train['ma10'] = train['close'].rolling(window=10).mean().shift(1)
        train['ma15'] = train['close'].rolling(window=15).mean().shift(1)
        train['ma20'] = train['close'].rolling(window=20).mean().shift(1)

        train['b_ma5'] = train['b_close'].rolling(window=5).mean().shift(1)
        train['b_ma10'] = train['b_close'].rolling(window=10).mean().shift(1)
        train['b_ma15'] = train['b_close'].rolling(window=15).mean().shift(1)
        train['b_ma20'] = train['b_close'].rolling(window=20).mean().shift(1)

        train = train.dropna()
        dfs.append(train)
    except Exception as e:
        print(e)
        pass
    df = pd.concat(dfs)
    X, Y = make_dataset(df, 180, 60)

    # neg, pos = np.bincount(Y)
    # total = neg + pos
    # print(neg)
    print(np.bincount(Y))
    print(X.shape[0], X.shape[1])
    np.savez('etc_data_3d2.npz', x_train=X, y_train=Y)
