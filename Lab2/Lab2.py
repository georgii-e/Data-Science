"""
Виконав: Васильєв Єгор
Lab_work_2, I група вимог
"""

import sys
import random
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.append('C:/Users/egorv/PycharmProjects/pythonProject/Data science/Lab1')
from Lab1 import y_values_synthetic, rates, print_characteristics, stat_characteristics


def check_normality(data):
    """

    Parameters
    ----------
    data: the data to be tested for normality.

    Returns
    -------
    void
    """
    # Perform the Shapiro-Wilk test
    statistic, p_value = stats.shapiro(data)
    alpha = 0.05
    print(p_value)
    if p_value > alpha:
        print("The data appears to be normally distributed (fail to reject the null hypothesis)\n")
    else:
        print("The data does not appear to be normally distributed (reject the null hypothesis)\n")


def add_normal_noise(data, std_dev, mean=0):
    """

    Parameters
    ----------
    data: our data - synthetic model
    mean: mean of the normal distribution
    std_dev: standard deviation of the normal distributio

    Returns
    -------
    noisy_data: data containing the noice
    """
    errors = np.random.normal(mean, std_dev, len(data))
    print("Statistical characteristics of the normal measurement error:")
    print_characteristics(lst=errors)
    plt.hist(errors)
    plt.title('Noise distributed according to a normal law')
    plt.show()
    noisy_data = data + errors
    print("Statistical characteristics of the model with noise:")
    print_characteristics(lst=noisy_data)
    return noisy_data


def add_abnormal_observation(data, noisy_data, std_dev, mean=0, abnormal_measurements_amount=10,
                             abnormal_coefficient=3):
    """

    Parameters
    ----------
    data: our data - synthetic model
    noisy_data: model + noice
    std_dev: standard deviation of the normal distributio
    mean: mean
    abnormal_measurements_amount: the number of anomalies that will be created
    abnormal_coefficient: advantage ratio of anomalous measurements

    Returns
    -------
    abnormal_model_with_noise: model + noice + anomalous measurements
    random_samples: indexes of the samples where anomalous measurements were added
    """
    abnormal_model_with_noise = noisy_data
    abnormal = np.random.normal(mean, (abnormal_coefficient * std_dev), abnormal_measurements_amount)
    random_samples = random.sample(range(len(data)), 10)
    for i in range(abnormal_measurements_amount):
        k = random_samples[i]
        abnormal_model_with_noise[k] += abnormal[i]
    return abnormal_model_with_noise, random_samples


def show_plot(data1, data2, title=None):
    """

    Parameters
    ----------
    data1: first dataset for visualisation
    data2: second dataset for visualisation
    title: title of the plot

    Returns
    -------
    void
    """
    plt.plot(data1)
    plt.plot(data2)
    plt.grid(True)
    plt.ylabel('BTC price in USD')
    plt.title(title)
    plt.show()


def replace_outliers(window_size, z_threshold, model):
    """

    Parameters
    ----------
    window_size: size of the sliding window
    z_threshold: threshold for identifying outliers
    model: our data - synthetic model with noice and anomalies

    Returns
    -------
    model without noice
    """
    moving_average = np.zeros_like(model)
    moving_std_dev = np.zeros_like(model)
    for i in range(len(model)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        window_data = model[start_idx:end_idx]
        moving_average[i] = np.mean(window_data)
        moving_std_dev[i] = np.std(window_data)
    z_scores = (model - moving_average) / (moving_std_dev + np.finfo(float).eps)
    outliers = np.abs(z_scores) > z_threshold  # Identify outliers based on the Z-Score threshold
    print("Indices where anomalies have been added: ", sorted(abnormal_indices))
    print("Indices where anomalies have been replaced: ", np.where(outliers)[0])
    clear_model = model.copy()
    clear_model[outliers] = np.mean(model)
    return clear_model


def build_regression(model):
    """

    Parameters
    ----------
    model: our data - model + noice + anomalous measurements

    Returns
    -------
    coefficients a, b and c from quadratic polynomial
    """
    indices = np.arange(len(model)) + 1
    X = np.column_stack(
        (np.ones_like(indices), indices, indices ** 2))  # Construct the design matrix (matrix of features)
    coefficients = np.linalg.lstsq(X, model, rcond=None)[0]  # Solve for the coefficients using LSM formula
    coef1, coef2, coef3 = coefficients
    equation = f"Regression Equation: y = {coef1:.2f} + {coef2:.2f}x + {coef3:.2f}x^2"
    print(equation)
    return coef1, coef2, coef3


# removing the trend from the original data
clear_data = rates["Close*"].values - y_values_synthetic
mean_clear, median_clear, variance_clear, std_dev_clear = stat_characteristics(clear_data)
print("\nStatistical characteristics of the original data without trend:")
print_characteristics(mean_clear, median_clear, variance_clear, std_dev_clear)

# normality test for data without trend
check_normality(clear_data)
plt.hist(clear_data)
plt.title('The law of distribution of the random component of real data')
plt.show()

# adding noice to the model
model_with_noise = add_normal_noise(y_values_synthetic, std_dev_clear)
show_plot(model_with_noise, y_values_synthetic, 'Trend model with normal distributed error')

# adding anomalous measurements to the model
complete_model, abnormal_indices = add_abnormal_observation(y_values_synthetic, model_with_noise, std_dev_clear)
show_plot(complete_model, y_values_synthetic, 'Trend model with normal distributed error and abnormal observations')
print("Statistical characteristics of the data with normal distributed error and abnormal observations:")
print_characteristics(lst=complete_model)

# removing outliers from the model using sliding window
cleared_model = replace_outliers(6, 1.6, complete_model)
show_plot(cleared_model, y_values_synthetic, 'Model cleared from outliers using sliding window and Z-Scores')
print("\nStatistical characteristics of the data cleared from outliers:")
print_characteristics(lst=cleared_model)

# building polynomial regression
a, b, c = build_regression(complete_model)
lsm_regression = a + b * (np.arange(len(complete_model))) + c * (np.arange(len(complete_model))) ** 2
show_plot(complete_model, lsm_regression, "Quadratic Model Fitting using LSM")
print("\nStatistical characteristics of the quadratic model fitted with LSM:")
print_characteristics(lst=lsm_regression)

# extrapolation of the model
extrapolated_x = np.arange(1.5 * len(complete_model)) + 1
extrapolated_y = a + b * extrapolated_x + c * extrapolated_x ** 2
show_plot(complete_model, extrapolated_y, "Extrapolated regression curve 50% ahead of the existing data")
