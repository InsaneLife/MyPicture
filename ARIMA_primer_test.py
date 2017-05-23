# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

rcParams['figure.figsize'] = 15, 6


# 移动平均图
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()


# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()


def test_stationarity(timeseries):
    # 决定起伏统计
    rolmean = pd.rolling_mean(timeseries, window=12)  # 对size个数据进行移动平均
    rolstd = pd.rolling_std(timeseries, window=12)  # 偏离原始值多少
    # 画出起伏统计
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    # 进行df测试
    print 'Result of Dickry-Fuller test'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value(%s)' % key] = value
    print dfoutput


# data=pd.read_csv('/Users/wangtuntun/Desktop/AirPassengers.csv')
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
# paese_dates指定日期在哪列  ;index_dates将年月日的哪个作为索引 ;date_parser将字符串转为日期
data = pd.read_csv('D:\\Competition\\AirPassengers.csv', parse_dates=['Month'], index_col='Month',
                   date_parser=dateparse)

ts = data['#Passengers']
# plt.plot(ts)
# plt.show()
# test_stationarity(ts)
# plt.show()

# 估计estimating
ts_log = np.log(ts)
# plt.plot(ts_log)
# plt.show()
moving_avg = pd.rolling_mean(ts_log, 12)
# plt.plot(moving_avg)
# plt.plot(moving_avg,color='red')
# plt.show()
ts_log_moving_avg_diff = ts_log - moving_avg
# print ts_log_moving_avg_diff.head(12)
ts_log_moving_avg_diff.dropna(inplace=True)
# test_stationarity(ts_log_moving_avg_diff)
# plt.show()
# 差分differencing

ts_log_diff = ts_log.diff(1)
ts_log_diff.dropna(inplace=True)
# test_stationarity(ts_log_diff)
# plt.show()

# 分解decomposing
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend  # 趋势
seasonal = decomposition.seasonal  # 季节性
residual = decomposition.resid  # 剩余的
'''
plt.subplot(411)
plt.plot(ts_log,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonarity')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
'''

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
# test_stationarity(ts_log_decompose)
# plt.show()
##预测##


# 确定参数
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
# q的获取:ACF图中曲线第一次穿过上置信区间.这里q取2
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')  # lowwer置信区间
plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')  # upper置信区间
plt.title('Autocorrelation Function')
# p的获取:PACF图中曲线第一次穿过上置信区间.这里p取2
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

# AR model
model = ARIMA(ts_log, order=(2, 1, 0))
result_AR = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(result_AR.fittedvalues, color='red')
plt.title('AR model RSS:%.4f' % sum(result_AR.fittedvalues - ts_log_diff) ** 2)
plt.show()

# MA model
model = ARIMA(ts_log, order=(0, 1, 2))
result_MA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(result_MA.fittedvalues, color='red')
plt.title('MA model RSS:%.4f' % sum(result_MA.fittedvalues - ts_log_diff) ** 2)
plt.show()

# ARIMA 将两个结合起来  效果更好
model = ARIMA(ts_log, order=(2, 1, 2))
result_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(result_ARIMA.fittedvalues, color='red')
plt.title('ARIMA RSS:%.4f' % sum(result_ARIMA.fittedvalues - ts_log_diff) ** 2)
plt.show()


predictions_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)
# print predictions_ARIMA_diff.head()#发现数据是没有第一行的,因为有1的延迟

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
# print predictions_ARIMA_diff_cumsum.head()

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
# print predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('predictions_ARIMA RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts) ** 2) / len(ts)))
plt.show()
