import requests
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing

today = 25  # datetime.today().day
warnings.filterwarnings("ignore")

# constants for accessing and parsing accuweather.com
url = "https://www.accuweather.com/ru/ua/kyiv/324505/october-weather/324505?year=2023"
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36"
header = {"User-Agent": user_agent}


# noinspection PyTypeChecker
def parsing(URL, headers):
    """

    Parameters
    ----------
    URL: URL of the website
    headers: headers for request

    Returns
    -------
    high_temperatures, low_temperatures: maximum and minimum temperatures for the day
    """
    response = requests.get(URL, headers=headers)
    soup = bs(response.content, "html.parser")
    high_temperatures = soup.find_all("div", class_="high")[:today - 1]
    low_temperatures = soup.find_all("div", class_="low")[:today - 1]

    for i in range(today - 1):
        high_temperatures[i] = int(high_temperatures[i].get_text()[7:-7])
        low_temperatures[i] = int(low_temperatures[i].get_text()[7:-7])
    return high_temperatures, low_temperatures


# original data transformation
high_temps, low_temps = parsing(url, header)
dates = range(1, today + 8)
X = np.arange(1, today).reshape(-1, 1)
y = ((np.array(high_temps) + np.array(low_temps)) / 2).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# exponential smoothing model
exponential_smoothing = ExponentialSmoothing(y[:-4], trend='mul', seasonal='add', seasonal_periods=5)
exponential_smoothing_fit = exponential_smoothing.fit()
exp_predicted = exponential_smoothing_fit.predict(start=0, end=today + 6)
exp_mse = mean_squared_error(y[-4:], exp_predicted[today - 4:today])
print("Exponential smoothing:")
for p, e in zip(exp_predicted, y_test):
    e = float(e[0])
    print(f'predicted: {p:.2f}, expected: {e:.2f}, difference: {(e - p) / p * 100:.2f}%')
print('\n')

# ARIMA model
arima = sm.tsa.ARIMA(y[:-4], order=(2, 1, 2))
arima_fit = arima.fit()
arima_predicted = arima_fit.predict(start=0, end=today + 6)
arima_predicted[0] = np.mean(y)
arima_mse = mean_squared_error(y[-4:], arima_predicted[today - 4:today])
print("Autoregressive integrated moving average:")
for p, e in zip(arima_predicted, y_test):
    e = float(e[0])
    print(f'predicted: {p:.2f}, expected: {e:.2f}, difference: {(e - p) / (p) * 100:.2f}%')
print('\n')

# SVM model
svm = SVR(C=5)
svm.fit(X_train, y_train)
svm_predicted = svm.predict(X_test)
svm_forecast = svm.predict(np.arange(1, today + 8).reshape(-1, 1))
svm_mse = mean_squared_error(y_test, svm_predicted)
print("Support vector machine:")
for p, e in zip(svm_predicted, y_test):
    e = float(e[0])
    print(f'predicted: {p:.2f}, expected: {e:.2f}, difference: {(e - p) / p * 100:.2f}%')
print('\n')

print(f'Mean squared error for exponential smoothing: {exp_mse:.3f}')
print(f'Mean squared error for autoregressive integrated moving average: {arima_mse:.3f}')
print(f'Mean squared error for support vector regression: {svm_mse:.3f}')

average_prediction = ((exp_predicted + arima_predicted + svm_forecast) / 3)  # average prediction of three models
print("Final weather forecast for the week:\n", average_prediction[-7:])

# 2D visualisation
plt.scatter(X, y, label='Actual Data')
plt.plot(dates, exp_predicted, label='Exponential smoothing')
plt.plot(dates, arima_predicted, label='Autoregressive integrated moving average')
plt.plot(dates, svm_forecast, color='red', label='Support Vector Regression')
plt.plot(dates, average_prediction, label='Prediction based on the previous models')
plt.xlabel('Days of October')
plt.ylabel('Temperature in Kyiv')
plt.ylim(0, 18)
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot each 2D plot as a separate layer in 3D
ax.bar(dates, exp_predicted, zdir='y', zs=0, label='Exponential smoothing')
ax.bar(dates, arima_predicted, zdir='y', zs=1, label='Autoregressive integrated moving average')
ax.bar(dates, svm_forecast, zdir='y', zs=2, label='Support Vector Regression')
ax.bar(dates, average_prediction, zdir='y', zs=3, label='Prediction based on the previous models')

ax.set_xlabel('Days of October')
ax.set_zlabel('Temperature in Kyiv')
ax.set_ylabel('Model')
ax.set_ylim(0, 4)
ax.set_yticklabels(['1','', '2','', '3','', '4'])
ax.set_zlim(0, 18)
# ax.view_init(100, -90)
plt.legend()
plt.show()
