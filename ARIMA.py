import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

os.chdir("D:\\HYUSTL\\5. 수업\\2기\\물류네트워크설계\\2. Research\\4. 코드\\")
df = pd.read_csv('0. 데이터\\input_teu_elec_210917.csv')
df = df[['date','total_electricity']]
df = df.set_index('date')
df.plot()
plt.show()
values = df.values
values = values.astype('float32')
result = adfuller(values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

autocorrelation_plot(df)
plt.show()

""" 15, 10"""
model = ARIMA(df, order = (10, 1, 0))
model_fit = model.fit()

print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

""" 10: 1766063.371, 0.34, 15: 2140946.370, 0.03"""
size = int(len(values) * 0.66)
train, test = values[0:size], values[size:len(values)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(10,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

r2_score(test, predictions)