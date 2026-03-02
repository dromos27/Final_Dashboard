# PA2_model_training.py (UPDATED)
"""
Model training module
Trains and exports all machine learning models
"""

import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# Import data preprocessing
from PA1_data_preprocessing import DataPreprocessor

class ModelTrainer:
    """Trains and exports all predictive models"""

    def __init__(self, assets_dir='assets'):
        self.assets_dir = assets_dir
        self.preprocessor = DataPreprocessor(assets_dir)
        self.kwh_model = None
        self.rate_model = None
        self.encoder = None
        self.room_encoder = None  # NEW: For encoding room numbers

    def train_kwh_model(self, save_path='kwh_predictor.joblib'):
        """Train the kWh prediction model (XGBoost)"""
        print("=" * 50)
        print("Training kWh Prediction Model")
        print("=" * 50)

        # Load and preprocess data
        data = self.preprocessor.load_energy_data()

        if data.empty:
            print("Error: No data available for training")
            return None

        # Prepare features - REMOVE 'room' column for now
        feature_columns = [
            'year', 'month', 'day', 'day_of_week', 'is_weekend',
            'month_sin', 'month_cos', 'floor'  # Use floor instead of room
        ]

        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in data.columns:
                print(f"Warning: Column '{col}' not found in data")

        # Create room_type encoding
        if 'room_type' in data.columns:
            self.encoder = self.preprocessor.create_encoder(data)
            encoded_features = self.encoder.transform(data[['room_type']])
            encoded_df = pd.DataFrame(
                encoded_features,
                columns=self.encoder.get_feature_names_out(['room_type'])
            )
            data = pd.concat([data, encoded_df], axis=1)
            feature_columns.extend(encoded_df.columns.tolist())

        # Remove 'room_type' if it's still there
        if 'room_type' in feature_columns:
            feature_columns.remove('room_type')

        # Encode room numbers if we want to include them
        # Option 1: Use floor instead (simpler)
        # Option 2: Encode room numbers
        if 'room' in data.columns and False:  # Set to True if you want to encode rooms
            print("Encoding room numbers...")
            self.room_encoder = LabelEncoder()
            data['room_encoded'] = self.room_encoder.fit_transform(data['room'])
            feature_columns.append('room_encoded')
            # Save room encoder
            joblib.dump(self.room_encoder, 'room_encoder.joblib')

        # Prepare X and y
        X = data[feature_columns]
        y = data['energy_kwh']

        # Convert all columns to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)

        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Feature columns: {X.columns.tolist()}")
        print(f"Feature dtypes: {X.dtypes.tolist()}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train XGBoost model
        self.kwh_model = XGBRegressor(
            n_estimators=200,  # Reduced for faster training
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror',
            enable_categorical=False  # Explicitly disable categorical
        )

        print(f"Training on {len(X_train)} samples...")
        self.kwh_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.kwh_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\nModel Performance:")
        print(f"MAE: {mae:.2f} kWh")
        print(f"RMSE: {rmse:.2f} kWh")
        print(f"R² Score: {self.kwh_model.score(X_test, y_test):.3f}")

        # Feature importance
        if hasattr(self.kwh_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.kwh_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nTop 5 Important Features:")
            for _, row in feature_importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")

        # Save model
        joblib.dump(self.kwh_model, save_path)
        print(f"\n✓ kWh model saved to: {save_path}")

        return self.kwh_model

    # In train_simple_kwh_model method, add MORE date features:
    def train_simple_kwh_model(self, save_path='kwh_predictor.joblib'):
        """Train model with enhanced date features"""
        print("=" * 50)
        print("Training kWh Prediction Model (WITH ENHANCED DATE FEATURES)")
        print("=" * 50)

        # Load and preprocess data
        data = self.preprocessor.load_energy_data()

        if data.empty:
            print("Error: No data available for training")
            return None

        # Enhanced date-based features
        feature_columns = [
            'year', 'month', 'day', 'day_of_week', 'is_weekend',
            'month_sin', 'month_cos',
            # Add more seasonal features
        ]

        # Add cyclic day encoding
        data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
        data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
        feature_columns.extend(['day_sin', 'day_cos'])

        # Add quarter of year
        data['quarter'] = ((data['month'] - 1) // 3) + 1
        feature_columns.append('quarter')

        # Add is_holiday flag (you can enhance this later)
        data['is_holiday'] = 0  # Placeholder - add real holiday data if available
        feature_columns.append('is_holiday')

        # Add room-specific features
        if 'room' in data.columns:
            from sklearn.preprocessing import LabelEncoder
            room_encoder = LabelEncoder()
            data['room_encoded'] = room_encoder.fit_transform(data['room'])
            feature_columns.append('room_encoded')

            # Save room encoder for prediction
            joblib.dump(room_encoder, 'room_encoder.joblib')
            print(f"Encoded {len(room_encoder.classes_)} unique rooms")

        # Add floor level
        if 'floor' in data.columns:
            feature_columns.append('floor')

        # Add room type as numeric
        if 'room_type' in data.columns:
            data['room_type_numeric'] = data['room_type'].map({'lecture': 0, 'lab': 1}).fillna(0)
            feature_columns.append('room_type_numeric')

        # Prepare X and y
        X = data[feature_columns]
        y = data['energy_kwh']

        # Convert all columns to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)

        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model with more complexity
        self.kwh_model = XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror',
            min_child_weight=3
        )

        print(f"Training on {len(X_train)} samples...")
        self.kwh_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.kwh_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\nModel Performance:")
        print(f"MAE: {mae:.2f} kWh")
        print(f"RMSE: {rmse:.2f} kWh")
        print(f"R² Score: {self.kwh_model.score(X_test, y_test):.3f}")

        # Feature importance
        if hasattr(self.kwh_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.kwh_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nTop 10 Important Features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                print(f"  {i:2d}. {row['feature']:20s}: {row['importance']:.3f}")

        # Save model
        joblib.dump(self.kwh_model, save_path)
        print(f"\n✓ kWh model saved to: {save_path}")

        return self.kwh_model

    def train_rate_model(self, save_path='rate_forecaster.joblib'):
        """Train the electricity rate forecasting model (Holt-Winters)"""
        print("\n" + "=" * 50)
        print("Training Rate Forecasting Model")
        print("=" * 50)

        # Load historical rates
        rates_df = self.preprocessor.load_historical_rates()

        if rates_df.empty:
            print("No historical rate data found, creating synthetic rates...")
            dates = pd.date_range(start='2021-01-01', end='2025-12-01', freq='MS')
            rates_df = pd.DataFrame({
                'date': dates,
                'rate_per_kWh': [11.0 + (i * 0.08) for i in range(len(dates))]
            })

        # Prepare time series data
        rates_df = rates_df.set_index('date').asfreq('MS')
        rates_series = rates_df['rate_per_kWh']

        # Train Holt-Winters model
        print(f"Training on {len(rates_series)} months of data...")
        self.rate_model = ExponentialSmoothing(
            rates_series,
            trend='add',
            seasonal='add',
            seasonal_periods=12,
            initialization_method='estimated'
        ).fit()

        # Evaluate
        forecast_steps = 12
        forecast = self.rate_model.forecast(forecast_steps)

        print(f"\nModel Summary:")
        print(f"AIC: {self.rate_model.aic:.2f}")
        print(f"BIC: {self.rate_model.bic:.2f}")

        print(f"\nNext 6 months forecast:")
        for i, (date, rate) in enumerate(zip(forecast.index[:6], forecast.values[:6])):
            print(f"  {date.strftime('%b %Y')}: ₱{rate:.2f}/kWh")

        # Save model
        joblib.dump(self.rate_model, save_path)
        print(f"\n✓ Rate model saved to: {save_path}")

        return self.rate_model

    def train_all_models(self):
        """Train and export all models"""
        print("=" * 60)
        print("TRAINING ALL PREDICTIVE MODELS")
        print("=" * 60)

        # Train kWh model (using simple version)
        print("\n1. Training kWh prediction model...")
        kwh_model = self.train_simple_kwh_model()

        # Train rate model
        print("\n2. Training rate forecasting model...")
        rate_model = self.train_rate_model()

        # Create model registry
        model_registry = {
            'kwh_model': kwh_model,
            'rate_model': rate_model,
            'trained_date': pd.Timestamp.now(),
            'version': '1.0',
            'description': 'Energy prediction models for rooms 1xx and 7xx',
            'features_used': ['year', 'month', 'day', 'day_of_week', 'is_weekend',
                             'month_sin', 'month_cos', 'floor', 'room_type_numeric']
        }

        # Save registry
        joblib.dump(model_registry, 'model_registry.joblib')
        print(f"\n✓ Model registry saved to: model_registry.joblib")

        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETE")
        print("=" * 60)
        print("\nGenerated files:")
        print("  1. kwh_predictor.joblib - XGBoost model for kWh prediction")
        print("  2. rate_forecaster.joblib - Holt-Winters model for rate forecasting")
        print("  3. room_type_encoder.joblib - OneHotEncoder for room types")
        print("  4. model_registry.joblib - Combined model registry")

        return model_registry

    def export_model_bundle(self, bundle_path='model_bundle.joblib'):
        """Export all models and metadata in a single bundle"""
        print("\n" + "=" * 50)
        print("Creating Model Bundle")
        print("=" * 50)

        # Ensure all models are trained
        if self.kwh_model is None:
            self.train_simple_kwh_model()
        if self.rate_model is None:
            self.train_rate_model()
        if self.encoder is None:
            data = self.preprocessor.load_energy_data()
            self.encoder = self.preprocessor.create_encoder(data)

        # Load only recommendations
        try:
            recommendations = pd.read_csv(os.path.join(self.assets_dir, 'recommendations.csv'))
        except:
            print("Warning: Could not load recommendations CSV")
            recommendations = pd.DataFrame({
                'threshold_type': ['energy_kWh', 'energy_cost_Php'],
                'threshold_value': [30, 500],
                'recommendation': ['Consider energy-saving mode', 'Review usage patterns']
            })

        # Create comprehensive bundle
        model_bundle = {
            'models': {
                'kwh_predictor': self.kwh_model,
                'rate_forecaster': self.rate_model,
                'room_type_encoder': self.encoder
            },
            'metadata': {
                'created': pd.Timestamp.now(),
                'version': '1.0',
                'supported_rooms': self.preprocessor.get_prediction_rooms(),
                'forecast_horizon': 24,  # Up to 2027
                'description': 'Complete predictive analytics bundle'
            },
            'config': {
                'default_rate': 11.0,
                'assets_dir': self.assets_dir
            },
            'recommendations': recommendations.to_dict('records')  # Add recommendations to bundle
        }

        # Save bundle
        joblib.dump(model_bundle, bundle_path)
        print(f"✓ Model bundle saved to: {bundle_path}")
        print(f"  Contains: {len(model_bundle['models'])} models + metadata + recommendations")

        return model_bundle


# Main execution
if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer()

    # Train all models
    trainer.train_all_models()

    print("\n" + "=" * 60)
    print("READY FOR DEPLOYMENT")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Copy the generated .joblib files to your dashboard directory")
    print("  2. Update PA3_energy_predictor.py to use these models")
    print("  3. Test predictions using PA3_energy_predictor.py")