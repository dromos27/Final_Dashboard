# PA4_rate_forecaster.py
"""
Electricity Rate Forecasting Module
Uses Holt-Winters time series for rate predictions up to 2027
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class RateForecaster:
    """Forecasts electricity rates using time series analysis"""
    
    def __init__(self, assets_dir='assets'):
        self.assets_dir = assets_dir
        self.model = None
        self.historical_rates = None
        self.last_historical_date = None
        
        try:
            # Try to load pre-trained model
            self.model = joblib.load('rate_forecaster.joblib')
            print("✓ Loaded pre-trained rate forecaster")
            
            # Load historical rates for reference
            self.historical_rates = self._load_historical_rates()
            if not self.historical_rates.empty:
                self.last_historical_date = self.historical_rates['date'].max()
            else:
                self.last_historical_date = pd.Timestamp('2025-12-01')
            
        except FileNotFoundError:
            print("No pre-trained model found, initializing...")
            self._initialize()
        except Exception as e:
            print(f"Error loading model: {e}")
            self._initialize()
    
    def _initialize(self):
        """Initialize or train the model"""
        try:
            self.historical_rates = self._load_historical_rates()
            
            if not self.historical_rates.empty:
                # Train model with historical data
                self.model = self._train_model(self.historical_rates)
                
                # Save the trained model
                joblib.dump(self.model, 'rate_forecaster.joblib')
                print("✓ Rate forecaster trained and saved")
                
                self.last_historical_date = self.historical_rates['date'].max()
            else:
                print("Warning: No historical rate data available")
                self.model = None
                self.last_historical_date = pd.Timestamp('2025-12-01')
                
        except Exception as e:
            print(f"Error initializing rate forecaster: {e}")
            self.model = None
            self.last_historical_date = pd.Timestamp('2025-12-01')
    
    def _load_historical_rates(self):
        """Load 2021-2025 historical electricity rates"""
        try:
            csv_path = os.path.join(self.assets_dir, 'energy_rate.csv')
            if not os.path.exists(csv_path):
                print(f"Rate file not found: {csv_path}")
                return self._create_synthetic_rates()
            
            rates_df = pd.read_csv(csv_path)
            print(f"Loaded rate data. Shape: {rates_df.shape}")
            print(f"Columns: {rates_df.columns.tolist()}")
            
            # Check required columns
            required_cols = ['year', 'month', 'rate_per_kWh']
            for col in required_cols:
                if col not in rates_df.columns:
                    print(f"Warning: Column '{col}' missing in rate data")
                    return self._create_synthetic_rates()
            
            # Create datetime column
            rates_df['date'] = pd.to_datetime(rates_df[['year', 'month']].assign(day=1))
            
            # Sort by date
            rates_df = rates_df[['date', 'rate_per_kWh']].sort_values('date')
            
            # Summary
            print(f"Date range: {rates_df['date'].min()} to {rates_df['date'].max()}")
            print(f"Rate range: ₱{rates_df['rate_per_kWh'].min():.2f} to ₱{rates_df['rate_per_kWh'].max():.2f}")
            print(f"Average rate: ₱{rates_df['rate_per_kWh'].mean():.2f}")
            
            return rates_df
            
        except Exception as e:
            print(f"Error loading historical rates: {e}")
            return self._create_synthetic_rates()
    
    def _create_synthetic_rates(self):
        """Create synthetic rates for development/testing"""
        print("Creating synthetic rate data for development...")
        
        dates = pd.date_range(start='2021-01-01', end='2025-12-01', freq='MS')
        
        # Create realistic rate progression
        base_rate = 11.0
        rates = []
        
        for i, date in enumerate(dates):
            # Add small monthly increase with some randomness
            monthly_increase = 0.08 + np.random.normal(0, 0.02)
            seasonal_factor = np.sin(2 * np.pi * date.month / 12) * 0.3
            
            rate = base_rate + (i * 0.08) + seasonal_factor + np.random.normal(0, 0.1)
            rates.append(max(10.0, min(13.5, rate)))  # Keep within reasonable bounds
        
        synthetic_rates = pd.DataFrame({
            'date': dates,
            'rate_per_kWh': rates
        })
        
        print(f"Created synthetic rates from {dates[0]} to {dates[-1]}")
        return synthetic_rates
    
    def _train_model(self, historical_data):
        """Train Holt-Winters model on historical data"""
        try:
            # Prepare time series
            rates_series = historical_data.set_index('date').asfreq('MS')['rate_per_kWh']
            
            print(f"Training model on {len(rates_series)} months of data...")
            
            # Train Holt-Winters model
            model = ExponentialSmoothing(
                rates_series,
                trend='add',
                seasonal='add',
                seasonal_periods=12,
                initialization_method='estimated'
            ).fit()
            
            print(f"Model trained successfully")
            print(f"AIC: {model.aic:.2f}, BIC: {model.bic:.2f}")
            
            # Test forecast
            test_forecast = model.forecast(6)
            print(f"Next 6 months forecast:")
            for date, rate in zip(test_forecast.index, test_forecast.values):
                print(f"  {date.strftime('%b %Y')}: ₱{rate:.2f}/kWh")
            
            return model
            
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def predict_rate(self, date):
        """
        Predict rate for a given date:
        - Returns exact historical rate if available
        - Forecasts future rate if beyond historical data
        - Uses fallback if no model available
        """
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)
        
        # Check if we have historical data
        if self.historical_rates is not None and not self.historical_rates.empty:
            monthly_date = pd.to_datetime(f"{date.year}-{date.month}-01")
            
            # Check for exact historical match
            historical_match = self.historical_rates[
                self.historical_rates['date'] == monthly_date
            ]
            
            if not historical_match.empty:
                return float(historical_match['rate_per_kWh'].iloc[0])
        
        # Use model for forecasting if available
        if self.model is not None and self.last_historical_date:
            # Calculate months to forecast
            n_months = ((date.year - self.last_historical_date.year) * 12 +
                        (date.month - self.last_historical_date.month))
            
            if n_months > 0:
                try:
                    forecast = self.model.forecast(n_months)
                    return float(forecast.iloc[-1])
                except:
                    # If forecast fails, use extrapolation
                    return self._extrapolate_rate(date)
        
        # Fallback: Use last known rate or default
        if self.historical_rates is not None and not self.historical_rates.empty:
            return float(self.historical_rates['rate_per_kWh'].iloc[-1])
        
        # Ultimate fallback
        return 11.0

    def _extrapolate_rate(self, date):
        """Extrapolate rate using trend from historical data"""
        if self.historical_rates is None or len(self.historical_rates) < 2:
            return 11.0

        # Get recent trend
        recent = self.historical_rates.tail(12)
        if len(recent) < 2:
            return float(self.historical_rates['rate_per_kWh'].iloc[-1])

        # Calculate monthly trend
        start_rate = recent['rate_per_kWh'].iloc[0]
        end_rate = recent['rate_per_kWh'].iloc[-1]
        months = len(recent) - 1

        if months > 0:
            monthly_trend = (end_rate - start_rate) / months
        else:
            monthly_trend = 0.08  # Default monthly increase

        # Calculate months from last historical date
        n_months = ((date.year - self.last_historical_date.year) * 12 +
                    (date.month - self.last_historical_date.month))

        # Extrapolate
        last_rate = float(self.historical_rates['rate_per_kWh'].iloc[-1])
        extrapolated = last_rate + (monthly_trend * n_months)

        # Add seasonal adjustment
        seasonal = np.sin(2 * np.pi * date.month / 12) * 0.3
        extrapolated += seasonal

        # REMOVED: Randomness for realism
        # extrapolated *= (1 + np.random.uniform(-0.02, 0.02))

        # Keep within reasonable bounds
        return max(10.0, min(15.0, extrapolated))
    
    def forecast_range(self, start_date, end_date):
        """Forecast rates for a date range"""
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        forecasts = []
        
        for date in dates:
            rate = self.predict_rate(date)
            forecasts.append({
                'date': date,
                'rate_per_kWh': rate
            })
        
        return pd.DataFrame(forecasts)


# Simple test
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING RATE FORECASTER")
    print("=" * 60)
    
    forecaster = RateForecaster()
    
    # Test predictions
    test_dates = [
        pd.Timestamp('2024-06-01'),
        pd.Timestamp('2025-01-01'),
        pd.Timestamp('2026-06-01'),
        pd.Timestamp('2027-12-01')
    ]
    
    print("\nTest Predictions:")
    for date in test_dates:
        rate = forecaster.predict_rate(date)
        print(f"  {date.strftime('%b %Y')}: ₱{rate:.2f}/kWh")