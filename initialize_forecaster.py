# initialize_forecaster.py
from rate_forecaster import RateForecaster
from data_preprocessing import load_historical_rates

if __name__ == "__main__":
    historical_rates = load_historical_rates()
    forecaster = RateForecaster()
    forecaster.train(historical_rates)
    print("Rate forecaster trained and saved successfully.")