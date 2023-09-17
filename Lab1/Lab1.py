'''
Виконав: Васильєв Єгор
Lab_work_1, III рівень складності
'''

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs

# constants for accessing and parsing finance.yahoo.com
url = "https://finance.yahoo.com/quote/BTC-USD/history?period1=1631577600&period2=1694649600&interval=1wk&filter=history&frequency=1wk&includeAdjustedClose=true"
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36"
headers = {"User-Agent": user_agent}


def stat_characteristics(lst):
    '''
    :param lst: sample for which characteristics are calculated
    :return: mean, median, variance, and root-mean-square deviation of the sample
    '''
    mean = np.mean(lst)
    median = np.median(lst)
    variance = np.var(lst)
    RMSD = np.sqrt(variance)
    return mean, median, variance, RMSD


def print_characteristics(mean, median, variance, RMSD):
    '''
    :param mean: mean
    :param median: median
    :param variance: variance
    :param RMSD: root-mean-square deviation
    :return: void
    '''
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Variance: {variance:.2f}")
    print(f"Root mean square deviation: {RMSD:.2f}\n")


def percentage_difference(old_value, new_value):
    '''
    :param old_value: first number
    :param new_value: second number
    :return: percentage difference between two numbers
    '''
    return ((new_value - old_value) / old_value) * 100

# parsing the website
response = requests.get(url, headers=headers)
soup = bs(response.content, "html.parser")
temperatures_html = soup.find_all("table", class_="W(100%) M(0)")[0]

# retrieving data from the html table
btc_rate = []

for row in temperatures_html.find_all("tr"):
    row_data = []

    for cell in row.find_all("td"):
        row_data.append(cell.text)

    btc_rate.append(row_data)

columns = temperatures_html.find_all("th")
column_list = [item.get_text(strip=True) for item in columns]
rates = pd.DataFrame(btc_rate, columns=column_list)


# data cleansing and transformation
rates = rates.dropna()
rates['Open'] = rates['Open'].str.replace(',', '').astype(float)
rates['High'] = rates['High'].str.replace(',', '').astype(float)
rates['Low'] = rates['Low'].str.replace(',', '').astype(float)
rates['Close*'] = rates['Close*'].str.replace(',', '').astype(float)
rates['Adj Close**'] = rates['Adj Close**'].str.replace(',', '').astype(float)
rates['Volume'] = rates['Volume'].str.replace(',', '').astype(float)
rates['Date'] = pd.to_datetime(rates['Date'], format='%b %d, %Y').dt.date
rates.to_csv("output.csv")

# data visualisation
rates.plot(figsize=(8, 6), x='Date', y='Close*')
plt.grid(True)
plt.ylabel('BTC price in USD')
plt.title('Bitcoin price change')
plt.show()

# calculating statistical characteristics for the data
print("Statistical characteristics for the dataset")
mean_data, median_data, variance_data, RMSD_data = stat_characteristics(rates['Close*'])
print_characteristics(mean_data, median_data, variance_data, RMSD_data)

# synthesis of a model similar to real data
a = 12
b = -850
c = 30000
x_values_synthetic = [x for x in range(len(rates))]
y_values_synthetic = [a * x ** 2 + b * x + c for x in x_values_synthetic]

# visualisation of quadratic model along with real data
plt.figure(figsize=(12, 6))
plt.scatter(rates['Date'], rates['Close*'], label='Historical Prices', alpha=0.5)
plt.plot(rates['Date'], y_values_synthetic, color='red', label='Synthetic Quadratic Model')
plt.xlabel('Date')
plt.ylabel('BTC price in USD')
plt.title('Historical Bitcoin Prices with Synthetic Quadratic Model')
plt.legend()
plt.grid(True)
plt.show()

# calculating statistical characteristics for the synthetic model
print("Statistical characteristics for the synthetic model")
mean_synthetic, median_synthetic, variance_synthetic, RMSD_synthetic = stat_characteristics(y_values_synthetic)
print_characteristics(mean_synthetic, median_synthetic, variance_synthetic, RMSD_synthetic)

# calculating difference between statistical characteristics of real data and artificial model
print("Difference between real data and synthetic model:")
print(f"Mean: {percentage_difference(mean_data, mean_synthetic):.2f}%")
print(f"Median: {percentage_difference(median_data, median_synthetic):.2f}%")
print(f"Variance: {percentage_difference(variance_data, variance_synthetic):.2f}%")
print(f"Root mean square deviation: {percentage_difference(RMSD_data, RMSD_synthetic):.2f}%")
