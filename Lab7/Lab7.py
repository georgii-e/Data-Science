import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

sales_df = pd.read_excel(r'C:\Users\egorv\PycharmProjects\pythonProject\Data science\Lab7\Example\Pr_1.xls',
                         usecols=[x for x in range(6)])
store_locations_df = pd.read_excel(r'C:\Users\egorv\PycharmProjects\pythonProject\Data science\Lab7\Example\Pr_1.xls',
                                   usecols=[9, 10], nrows=199)
sales_df.rename(
    columns={'Місяц': 'Місяць', 'КільКість реалізацій': 'Кількість реалізацій', 'Ціна реализації': 'Ціна реалізації'},
    inplace=True)
store_locations_df.rename(columns={'Код магазину.1': 'Код магазину'}, inplace=True)

print(sales_df['Місяць'].unique())
print(sales_df[sales_df['Місяць'] == 'Стоп'])
sales_df['Місяць'] = sales_df['Місяць'].replace('Стоп', 'Листопад')

print(store_locations_df['Код магазину'].isin(sales_df['Код магазину'].unique()).all())

sales_df['Виторг'] = sales_df['Кількість реалізацій'] * sales_df['Ціна реалізації']
sales_df['Прибуток'] = (sales_df['Ціна реалізації'] - sales_df['Собівартість одиниці']) * sales_df[
    'Кількість реалізацій']
monthly_sales = sales_df.groupby('Місяць')['Виторг'].sum()
monthly_profit = sales_df.groupby('Місяць')['Прибуток'].sum()
months_order = ['Січень', 'Лютий', 'Березень', 'Квітень', 'Травень', 'Червень', 'Липень', 'Серпень', 'Вересень',
                'Жовтень', 'Листопад', 'Грудень']
monthly_sales = monthly_sales.reindex(months_order)
monthly_profit = monthly_profit.reindex(months_order)
print(monthly_sales, monthly_profit)
plt.figure(figsize=(10, 6))  # plotting sales and profit by months
plt.plot(monthly_sales.index, monthly_sales, label='Продажі', marker='o')
plt.plot(monthly_profit.index, monthly_profit, label='Прибуток', marker='o')
plt.xlabel('Місяць')
plt.ylabel('Сума')
plt.title('Продажі та прибуток по місяцям')
plt.legend()
plt.show()

df_merged = sales_df.merge(store_locations_df, on='Код магазину', how='left')
regions = df_merged['Регіон'].unique()
df_merged['Місяць цифра'] = df_merged['Дата'].dt.month
months = np.sort(df_merged['Місяць цифра'].unique())
forecast_df = pd.DataFrame(index=regions,
                           columns=[f'{month} 2022' for month in months_order[:6]])  # DataFrame for storing forecasts
df_pivot = df_merged.pivot_table(values='Виторг', index='Регіон', columns='Місяць',
                                 aggfunc="sum")  # Pivoting the DataFrame for ease of use
df_pivot = df_pivot[months_order]
# Defining forecasts for the next 6 months for each region
for region in regions:
    region_sales = df_pivot.loc[region, months_order]
    coefficients = np.polyfit(months, region_sales, 1)
    forecast_sales = np.polyval(coefficients, months[-1] + np.arange(1, 7))
    forecast_df.loc[region] = forecast_sales
# forecast_df.plot(kind='bar', figsize=(12, 8))
# plt.xlabel('Регіон')
# plt.ylabel('Прогнозовані Продажі')
# plt.title('Прогноз зміни продажів на 6 місяців вперед за регіонами')
# plt.show()
sales_and_forecast_df = pd.concat([df_pivot, forecast_df], axis=1)
plt.figure(figsize=(12, 8))
for region in regions:
    p = sales_and_forecast_df.loc[region]
    plt.plot(sales_and_forecast_df.loc[region], label=region, marker='o')
plt.xlabel('Місяць')
plt.ylabel('Прогнозовані Продажі')
plt.title('Прогноз зміни продажів на 6 місяців вперед за регіонами')
plt.legend()
plt.show()

sales_and_forecast_df.to_excel('Output.xlsx', index=False)
