# AI Model Used: ChatGPT

# Prompts Used:
# 1. Explanation of how to load and preprocess data using pandas.
# 2. Guidance on implementing linear regression manually without using scikit-learn.
# 3. Steps to normalize data for linear regression.
# 4. How to handle command line arguments in Python for data input.
# 5. Instructions for plotting data and saving plots using matplotlib.
# 6. Help with interpreting linear regression results and predicting future values.
# 7. Advice on debugging Python code and handling specific errors like non-zero exit statuses.
# 8. Assistance with formatting output for Gradescope submissions.

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(filename):
    return pd.read_csv(filename)

def normalize_data(X):
    m = np.min(X)
    M = np.max(X)
    return (X - m) / (M - m), m, M

def perform_linear_regression(X_b, y):
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best

def predict(X_b, theta_best):
    return X_b.dot(theta_best)

def plot_results(X, y, y_predict):
    plt.figure(figsize=(10, 5))
    plt.plot(X, y, "b.")
    plt.plot(X, y_predict, "r-", linewidth=2, label="Predictions")
    plt.xlabel("Year")
    plt.ylabel("Days of Ice Cover")
    plt.title("Linear Regression - Predictions vs Actual")
    plt.legend()
    plt.savefig("data_plot.jpg")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python hw5.py <filename> <learning_rate> <iterations>")
        sys.exit(1)

    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])  # Not used here
    iterations = int(sys.argv[3])  # Not used here

    data = load_data(filename)
    X = data['year'].values.reshape(-1, 1)
    y = data['days'].values

    # Q3: Normalize data
    X_normalized, m, M = normalize_data(X)
    print("Q3: Normalized Data:", X_normalized)

    # Adding intercept term
    X_b = np.c_[np.ones((X.shape[0], 1)), X_normalized]

    # Q4: Perform linear regression
    theta_best = perform_linear_regression(X_b, y)
    print("Q4: Theta:", theta_best)

    # Predictions
    y_predict = predict(X_b, theta_best)

    # Q6: Predict future value for 2023 (assuming last year in data is 2022)
    future_x = (2023 - m) / (M - m)
    future_x_b = np.array([1, future_x])
    future_prediction = future_x_b.dot(theta_best)
    print("Q6: Prediction for 2023:", future_prediction)

    # Plot results
    plot_results(X, y, y_predict)

    # More outputs for other questions would be added here
