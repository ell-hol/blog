---
id: btc-prediction
title: "Predicting Bitcoin Prices Using Python"
date: "2024-09-03T00:00:00+00:00"
author: "Yacine Zahidi"
layout: post
guid: "btc-prediction-1"
custom_permalink:
  - /article/btc-prediction-1
categories:
  - Articles
image: https://github.com/user-attachments/assets/66a486b5-57f8-45f7-a808-3ee8216cd756
---

Bitcoin, the first and most well-known cryptocurrency, has captivated the world with its unpredictable price movements and massive growth over the years. In this article, we'll use Python to predict Bitcoin prices by applying a simple mathematical model based on historical data. Whether you're a data science enthusiast, a financial analyst, or just curious about cryptocurrency, this tutorial will guide you through the process of predicting Bitcoin prices using a few lines of Python code.

## Introduction to the Model

Bitcoin's price is influenced by many factors, including market demand, regulatory news, macroeconomic conditions, and technological advancements. While accurately predicting prices is challenging due to their volatility, we can create a basic model that uses historical data to provide a rough forecast of future prices.

For this purpose, we use a simple power-law model with the following formula:

\[
\text{Predicted Price} = C \times (\text{Days Since Genesis Block})^k
\]

- **C**: A scaling constant, set to \(1.60 \times 10^{-18}\).
- **k**: A growth rate parameter, set to 6.04.
- **Days Since Genesis Block**: The number of days since Bitcoin's creation on January 3, 2009.

This model is rudimentary but illustrates how price predictions can be made based on historical data trends.

## Python Code for Predicting Bitcoin Prices

Let's dive into the Python script that implements this model and predicts Bitcoin prices for a given period.

### Step 1: Importing Required Libraries

We start by importing the necessary libraries: `pandas` for data manipulation, `numpy` for numerical calculations, `matplotlib` for plotting, and `argparse` for handling command-line arguments.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
```

### Step 2: Defining Constants and Loading Data

We define the constants for the model (`C` and `k`) and the genesis block date (January 3, 2009). We also create a function to load historical Bitcoin data from a CSV file.

```python
# Model parameters and genesis block date
C = 1.60e-18
k = 6.04
genesis_block_date = datetime(2009, 1, 3)

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data
```

### Step 3: Predicting Bitcoin Prices

The `predict_bitcoin_prices` function calculates the predicted prices for a range of dates, given the start and end years and the month for prediction.

```python
def predict_bitcoin_prices(start_year, end_year, month):
    start_date = f"{start_year}-{month:02d}-01"
    end_date = datetime(end_year, month, 1) + pd.offsets.MonthEnd(1)
    date_range = pd.date_range(start=start_date, end=end_date)
    days_since_genesis = (date_range - genesis_block_date).days
    predicted_prices = C * days_since_genesis**k
    predictions_df = pd.DataFrame({'Date': date_range, 'Predicted Price (USD)': predicted_prices})
    return predictions_df
```

### Step 4: Plotting Predictions

We use Matplotlib to plot both historical and predicted Bitcoin prices. The plot uses a logarithmic scale for the y-axis to better visualize the price changes.

```python
def plot_predictions(btc_data, predictions):
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    if btc_data is not None:
        plt.plot(btc_data['Date'], btc_data['Close'], label='Historical Prices')
    plt.plot(predictions['Date'], predictions['Predicted Price (USD)'], label='Predicted Prices')
    plt.xlabel('Year')
    plt.ylabel('Price (USD)')
    plt.title('Bitcoin Price Predictions')
    plt.legend()
    plt.show()
```

### Step 5: Running the Script

The script can be run with various command-line options to specify the input CSV file, the year and month for prediction, and whether to plot the results.

```python
def main():
    parser = argparse.ArgumentParser(description="Generate Bitcoin price predictions.")
    parser.add_argument("--csv", "-c", type=str, help="Path to the CSV file containing historical Bitcoin prices.")
    parser.add_argument("--year", "-y", type=int, help="Start year for which to predict prices.")
    parser.add_argument("--month", "-m", type=int, help="Month for which to predict prices.")
    parser.add_argument("--plot", action='store_true', help="Plot predictions along with historical data.")
    args = parser.parse_args()

    btc_data = None
    if args.csv:
        btc_data = load_data(args.csv)

    if args.plot:
        predictions = predict_bitcoin_prices(2014, args.year + 25, args.month)
        plot_predictions(btc_data, predictions)
    else:
        predictions = predict_bitcoin_prices(args.year, args.year, args.month)
        print(predictions)

if __name__ == "__main__":
    main()
```

### Example Usage

To use this script, save it as `bitcoin_predictor.py` and run it from the command line. For example, to predict Bitcoin prices from 2024 for 25 years and plot them against historical data:

```sh
python bitcoin_predictor.py --csv historical_prices.csv --year 2024 --month 1 --plot
```

This command loads historical prices from a CSV file (`historical_prices.csv`), starts predicting from January 2024, and plots the results.

## Conclusion

Predicting Bitcoin prices can be an exciting and educational exercise. This script provides a basic model that demonstrates how Python can be used to predict financial data. Although this model is simplified and may not provide highly accurate forecasts, it can serve as a foundation for building more sophisticated models by incorporating additional factors like market sentiment, macroeconomic indicators, or advanced machine learning techniques.

By experimenting with different models and parameters, you can further enhance your understanding of cryptocurrency price dynamics and data science techniques.

Feel free to explore, modify, and extend this code to suit your needs! Happy coding and predicting!
