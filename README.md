# Stock Price Prediction and Strategy Analysis

This project involves predicting stock prices using an LSTM (Long Short-Term Memory) neural network model and backtesting a trading strategy based on historical data. The code includes data preprocessing, model training, prediction, and performance evaluation.

## Features

- **Data Loading**: Reads stock market data from a CSV file.
- **Data Preprocessing**: Renames columns, scales features, and prepares data for training and testing.
- **Feature Selection**: Chooses relevant features for model training.
- **Model Training**: Implements an LSTM model for time series forecasting.
- **Prediction**: Predicts stock prices and evaluates the model.
- **Strategy Backtesting**: Implements a trading strategy and evaluates its performance using historical data.
- **Visualization**: Plots the results of predictions and strategy performance.

## Files

- `main.py`: Main script containing the code for data preprocessing, model training, prediction, and backtesting.
- `notebook.py`: Jupyter notebook for executed codes
## Dependencies

The following Python libraries are required:

- `pandas`
- `numpy`
- `sklearn`
- `keras`
- `matplotlib`
- `seaborn`
- `yfinance`
- `pyfolio`
- `pandas_datareader`

Install the necessary libraries using:

```bash
pip install pandas numpy scikit-learn keras matplotlib seaborn yfinance pyfolio pandas_datareader
```

## Usage

1. **Load the Dataset**: Update the file path in the `pd.read_csv()` function call to point to your dataset.
2. **Run the Script**: Execute `main.py` to process the data, train the model, make predictions, and perform strategy backtesting.

## Notes

- Ensure the dataset has the necessary columns for the code to function correctly.
- Adjust hyperparameters and model architecture as needed for your specific use case.
