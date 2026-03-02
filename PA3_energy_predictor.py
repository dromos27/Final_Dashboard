# PA3_energy_predictor.py
"""
Energy Predictor Module with proper weekend handling
ZERO predictions for Sundays, minimal for Saturdays
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import rate forecaster
try:
    from PA4_rate_forecaster import RateForecaster
except ImportError as e:
    print(f"Warning: Could not import RateForecaster: {e}")

    class RateForecaster:
        def __init__(self):
            self.default_rate = 11.0

        def predict_rate(self, date):
            return self.default_rate

class EnergyPredictor:
    """Main prediction engine for energy consumption and costs"""

    def __init__(self, assets_dir='assets'):
        self.assets_dir = assets_dir

        # Try to import data preprocessor
        try:
            from PA1_data_preprocessing import DataPreprocessor
            self.DataPreprocessor = DataPreprocessor
        except ImportError:
            self.DataPreprocessor = None

        try:
            # Load trained models
            model_paths = [
                'kwh_predictor.joblib',
                os.path.join(current_dir, 'kwh_predictor.joblib'),
                os.path.join('models', 'kwh_predictor.joblib')
            ]

            for model_path in model_paths:
                if os.path.exists(model_path):
                    self.kwh_model = joblib.load(model_path)
                    print(f"✓ Loaded kWh model from: {model_path}")
                    break
            else:
                print("✗ kWh model not found")
                self.kwh_model = None

            # Load encoder
            encoder_paths = [
                'room_type_encoder.joblib',
                os.path.join(current_dir, 'room_type_encoder.joblib'),
                os.path.join('models', 'room_type_encoder.joblib')
            ]

            for encoder_path in encoder_paths:
                if os.path.exists(encoder_path):
                    self.encoder = joblib.load(encoder_path)
                    print(f"✓ Loaded encoder from: {encoder_path}")
                    break
            else:
                print("✗ Encoder not found")
                self.encoder = None

            # Initialize rate forecaster
            self.rate_forecaster = RateForecaster()

            # Load supporting data
            self._load_supporting_data()

            # Get available rooms
            self.available_rooms = self._get_available_rooms()

            # Hourly patterns for breakdown
            self.hourly_patterns = self._load_hourly_patterns()

            print(f"✓ EnergyPredictor initialized")
            print(f"✓ Available rooms: {self.available_rooms}")

        except Exception as e:
            print(f"✗ Error initializing EnergyPredictor: {e}")
            import traceback
            traceback.print_exc()
            self._setup_fallback()

    def _load_supporting_data(self):
        """Load CSV files with error handling"""
        try:
            # Only load recommendations file
            recs_path = os.path.join(self.assets_dir, 'recommendations.csv')
            if os.path.exists(recs_path):
                self.recommendations = pd.read_csv(recs_path)
                print(f"✓ Loaded recommendations: {len(self.recommendations)} rules")
            else:
                print(f"✗ Recommendations file not found: {recs_path}")
                self.recommendations = pd.DataFrame({
                    'threshold_type': ['energy_kWh', 'energy_cost_Php', 'energy_kWh'],
                    'threshold_value': [30, 500, 50],
                    'recommendation': [
                        'Consider energy-saving mode during off-peak hours',
                        'Review equipment usage patterns for optimization',
                        'High energy consumption detected. Consider scheduling equipment usage.'
                    ]
                })

        except Exception as e:
            print(f"✗ Error loading supporting data: {e}")
            self._setup_fallback()

    def _get_available_rooms(self):
        """Get list of available rooms (101-111, 701-705)"""
        # Define the rooms we want to show
        available_rooms = []

        # Level 1 rooms: 101-111
        for i in range(101, 112):
            available_rooms.append(str(i))

        # Level 7 rooms: 701-705
        for i in range(701, 706):
            available_rooms.append(str(i))

        # Check if we have data preprocessor to get actual rooms
        if self.DataPreprocessor is not None:
            try:
                preprocessor = self.DataPreprocessor(self.assets_dir)
                data = preprocessor.load_energy_data()
                if not data.empty:
                    # Get actual rooms from data
                    actual_rooms = preprocessor.get_prediction_rooms(data)
                    # Filter to only include our defined rooms
                    filtered_rooms = [r for r in actual_rooms if r in available_rooms]
                    if filtered_rooms:
                        available_rooms = filtered_rooms
            except:
                pass  # Use default rooms if preprocessor fails

        return sorted(available_rooms)

    def _load_hourly_patterns(self):
        """Load typical hourly consumption patterns"""
        # Based on room type patterns
        patterns = {
            'lab': [0.01, 0.005, 0.003, 0.002, 0.002, 0.01,   # 12AM-5AM
                    0.02, 0.05, 0.08, 0.10, 0.10, 0.10,        # 6AM-11AM
                    0.08, 0.07, 0.06, 0.05, 0.04, 0.03,        # 12PM-5PM
                    0.03, 0.02, 0.02, 0.015, 0.01, 0.01],      # 6PM-11PM
            'office': [0.002, 0.001, 0.001, 0.001, 0.001, 0.01, # 12AM-5AM
                       0.03, 0.08, 0.12, 0.15, 0.15, 0.12,      # 6AM-11AM
                       0.08, 0.04, 0.03, 0.02, 0.01, 0.005,     # 12PM-5PM
                       0.003, 0.002, 0.002, 0.001, 0.001, 0.001] # 6PM-11PM
        }
        return patterns

    def _setup_fallback(self):
        """Setup fallback values if initialization fails"""
        print("Setting up fallback mode...")
        self.kwh_model = None
        self.encoder = None
        self.rate_forecaster = RateForecaster()

        # Default recommendations
        self.recommendations = pd.DataFrame({
            'threshold_type': ['energy_kWh', 'energy_cost_Php'],
            'threshold_value': [30, 500],
            'recommendation': [
                'Consider energy-saving mode during off-peak hours',
                'Review equipment usage patterns for optimization'
            ]
        })

        self.available_rooms = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111',
                                '701', '702', '703', '704', '705']
        self.hourly_patterns = self._load_hourly_patterns()
        print("✓ Fallback mode activated")

    def _get_room_type(self, room_number):
        """Determine room type based on room number"""
        room_str = str(room_number)
        if room_str.startswith('7'):
            return 'lab'
        elif room_str.startswith('1'):
            return 'office'
        else:
            return 'office'  # Default

    def _get_zero_prediction(self, input_date, room_number, room_type=None):
        """Return zero prediction for Sundays and very low for Saturdays"""
        if room_type is None:
            room_type = self._get_room_type(room_number)

        day_of_week = input_date.weekday()  # Monday=0, Sunday=6
        day_name = input_date.strftime('%A')

        # Still get rate prediction for completeness
        rate_pred = self.rate_forecaster.predict_rate(input_date)

        if day_of_week == 6:  # Sunday
            return {
                'total_daily_kWh': 0.0,
                'avg_hourly_kWh': 0.0,
                'predicted_rate': round(rate_pred, 2),
                'total_cost_Php': 0.0,
                'avg_hourly_cost_Php': 0.0,
                'hourly_breakdown': [0.0] * 24,
                'recommendations': ["No energy consumption expected on Sundays (building closed)"],
                'room_type': 'Laboratory' if room_type == 'lab' else 'Office',
                'date': input_date.strftime('%Y-%m-%d'),
                'day_name': day_name,
                'is_weekend': True,
                'room_multiplier': 1.0  # Keep for compatibility but not used
            }
        else:  # Saturday
            # Very minimal usage for Saturday
            base_kwh = 1.0  # Minimal base
            total_kwh = base_kwh * (1.5 if room_type == 'lab' else 1.0)
            total_cost = round(total_kwh * rate_pred, 2)

            # Even distribution for Saturday
            hourly_kwh = [round(total_kwh * 0.04, 3)] * 24

            return {
                'total_daily_kWh': round(total_kwh, 2),
                'avg_hourly_kWh': round(total_kwh / 24, 3),
                'predicted_rate': round(rate_pred, 2),
                'total_cost_Php': total_cost,
                'avg_hourly_cost_Php': round(total_cost / 24, 2),
                'hourly_breakdown': hourly_kwh,
                'recommendations': ["Minimal energy consumption expected on Saturdays (limited operations)"],
                'room_type': 'Laboratory' if room_type == 'lab' else 'Office',
                'date': input_date.strftime('%Y-%m-%d'),
                'day_name': day_name,
                'is_weekend': True,
                'room_multiplier': 1.5 if room_type == 'lab' else 1.0  # Keep for compatibility
            }

    def get_recommendations(self, kwh, cost):
        """Generate recommendations based on predicted values"""
        recommendations = []

        try:
            # Energy-based recommendations from CSV
            kwh_recs = self.recommendations[
                (self.recommendations['threshold_type'] == 'energy_kWh') &
                (self.recommendations['threshold_value'] <= kwh)
            ]

            # Cost-based recommendations from CSV
            cost_recs = self.recommendations[
                (self.recommendations['threshold_type'] == 'energy_cost_Php') &
                (self.recommendations['threshold_value'] <= cost)
            ]

            # Combine and deduplicate
            all_recs = pd.concat([kwh_recs, cost_recs]).drop_duplicates()

            # Convert to list
            recommendations = all_recs['recommendation'].tolist()

            # Add dynamic recommendations based on values
            if kwh > 50:
                recommendations.append("High energy consumption detected. Consider scheduling equipment usage during off-peak hours.")
            if cost > 800:
                recommendations.append("Cost exceeding typical range. Review peak hour usage and consider energy-efficient alternatives.")
            if kwh < 1 and kwh > 0:
                recommendations.append("Low energy usage - typical for weekends or holidays.")

            # Remove duplicates
            recommendations = list(dict.fromkeys(recommendations))

            # Ensure we have at least one recommendation
            if not recommendations:
                recommendations = ["Monitor energy consumption patterns for optimization opportunities."]

        except Exception as e:
            print(f"Error generating recommendations: {e}")
            recommendations = ["Monitor energy consumption patterns for optimization opportunities."]

        return recommendations

    def predict(self, year, month, day, room_number):
        """
        Main prediction function with ENHANCED and CONSISTENT predictions
        """
        try:
            # Convert inputs
            room_number = str(room_number)
            input_date = pd.to_datetime(f"{year}-{month}-{day}")
            day_of_week = input_date.weekday()  # Monday=0, Sunday=6
            day_name = input_date.strftime('%A')
            is_weekend = 1 if day_of_week in [5, 6] else 0

            # ================================================
            # Handle weekends
            # ================================================
            if day_of_week >= 5:  # Saturday (5) or Sunday (6)
                room_type = self._get_room_type(room_number)
                return self._get_zero_prediction(input_date, room_number, room_type)

            # 1. Get room type
            room_type = self._get_room_type(room_number)

            # 2. Predict electricity rate
            rate_pred = self.rate_forecaster.predict_rate(input_date)

            # 3. Prepare ENHANCED features for kWh prediction
            input_features = pd.DataFrame({
                'year': [year],
                'month': [month],
                'day': [day],
                'day_of_week': [day_of_week],
                'is_weekend': [0],
                'month_sin': [np.sin(2 * np.pi * month / 12)],
                'month_cos': [np.cos(2 * np.pi * month / 12)],
                'day_sin': [np.sin(2 * np.pi * day / 31)],  # NEW: Daily cycle
                'day_cos': [np.cos(2 * np.pi * day / 31)],  # NEW: Daily cycle
                'quarter': [((month - 1) // 3) + 1],  # NEW: Quarter of year
                'is_holiday': [0],  # NEW: Holiday flag
                'floor': [int(room_number[0]) if room_number[0].isdigit() else 0],
                'room_type_numeric': [1 if room_type == 'lab' else 0]
            })

            # 4. Add room encoding if available
            try:
                if os.path.exists('room_encoder.joblib'):
                    room_encoder = joblib.load('room_encoder.joblib')
                    if room_number in room_encoder.classes_:
                        room_encoded = room_encoder.transform([room_number])[0]
                    else:
                        room_encoded = len(room_encoder.classes_)
                    input_features['room_encoded'] = [room_encoded]
            except:
                # If no encoder, use last 2 digits of room number
                try:
                    room_num = int(room_number)
                    input_features['room_last_digits'] = [room_num % 100]
                except:
                    pass

            # 5. Predict kWh
            kwh_pred = 0
            if self.kwh_model is not None:
                try:
                    # Ensure all columns are numeric
                    input_features = input_features.apply(pd.to_numeric, errors='coerce').fillna(0)

                    # Make prediction - NO RANDOM VARIATION
                    kwh_pred = float(self.kwh_model.predict(input_features)[0])
                    kwh_pred = max(0, kwh_pred)  # Ensure non-negative

                    # REMOVED: Random variation between days
                    # Instead, use DETERMINISTIC variation based on day of month
                    if day <= 7:  # First week - slightly lower
                        kwh_pred *= 0.98
                    elif day <= 14:  # Second week - normal
                        kwh_pred *= 1.0
                    elif day <= 21:  # Third week - slightly higher
                        kwh_pred *= 1.02
                    else:  # Last week - lower
                        kwh_pred *= 0.97

                except Exception as e:
                    print(f"Model prediction error: {e}, using fallback")
                    kwh_pred = self._get_fallback_kwh(room_type, day_of_week, month, day)
            else:
                # Fallback prediction WITHOUT random variation
                kwh_pred = self._get_fallback_kwh(room_type, day_of_week, month, day)

            # 6. Calculate costs
            total_cost = round(kwh_pred * rate_pred, 2)
            avg_hourly_kwh = round(kwh_pred / 24, 3)
            avg_hourly_cost = round(total_cost / 24, 2)

            # 7. Get hourly breakdown
            hourly_pattern = self.hourly_patterns.get(room_type, self.hourly_patterns['lab'])
            hourly_kwh = [round(kwh_pred * pattern, 3) for pattern in hourly_pattern]

            # 8. Get recommendations
            recommendations = self.get_recommendations(kwh_pred, total_cost)

            # 9. Prepare result
            result = {
                'total_daily_kWh': round(kwh_pred, 2),
                'avg_hourly_kWh': avg_hourly_kwh,
                'predicted_rate': round(rate_pred, 2),
                'total_cost_Php': total_cost,
                'avg_hourly_cost_Php': avg_hourly_cost,
                'hourly_breakdown': hourly_kwh,
                'recommendations': recommendations,
                'room_type': 'Laboratory' if room_type == 'lab' else 'Office',
                'date': input_date.strftime('%Y-%m-%d'),
                'day_name': day_name,
                'is_weekend': False,
                'day_of_month': day,
                'prediction_id': f"{room_number}_{year}{month:02d}{day:02d}"  # Add unique ID
            }

            return result

        except Exception as e:
            print(f"Prediction error for {room_number} on {year}-{month}-{day}: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_prediction(year, month, day, room_number)

    def _get_fallback_kwh(self, room_type, day_of_week, month, day):
        """Get fallback kWh with DETERMINISTIC daily variation"""
        # Base values by room type
        if room_type == 'lab':
            base_kwh = 25.0
        else:
            base_kwh = 15.0

        # Seasonal adjustment
        if month in [5, 6, 7, 8]:  # Summer months
            base_kwh *= 1.2
        elif month in [11, 12, 1, 2]:  # Winter months
            base_kwh *= 0.8

        # Day of month variation - DETERMINISTIC
        if day <= 7:  # First week - lower usage
            base_kwh *= 0.95
        elif day <= 14:  # Second week - normal
            base_kwh *= 1.0
        elif day <= 21:  # Third week - higher
            base_kwh *= 1.05
        else:  # Last week - lower
            base_kwh *= 0.9

        # REMOVED: Random variation
        # base_kwh *= random.uniform(0.95, 1.05)

        return max(0.5, base_kwh)  # Minimum 0.5 kWh

    def _get_fallback_prediction(self, year, month, day, room_number):
        """Provide fallback prediction when model fails"""
        input_date = pd.to_datetime(f"{year}-{month}-{day}")
        day_of_week = input_date.weekday()
        day_name = input_date.strftime('%A')
        room_type = self._get_room_type(room_number)

        # Check if weekend
        if day_of_week >= 5:
            return self._get_zero_prediction(input_date, room_number, room_type)

        # Simple fallback logic for weekdays
        if room_type == 'lab':
            total_kwh = 25.0
        else:
            total_kwh = 15.0

        # Adjust for month
        if month in [5, 6, 7, 8]:  # Summer
            total_kwh *= 1.2
        elif month in [11, 12, 1, 2]:  # Winter
            total_kwh *= 0.8

        rate = 11.0  # Default rate
        total_cost = round(total_kwh * rate, 2)

        # Get hourly pattern
        hourly_pattern = self.hourly_patterns.get(room_type, self.hourly_patterns['lab'])
        hourly_kwh = [round(total_kwh * pattern, 3) for pattern in hourly_pattern]

        return {
            'total_daily_kWh': round(total_kwh, 2),
            'avg_hourly_kWh': round(total_kwh / 24, 3),
            'predicted_rate': rate,
            'total_cost_Php': total_cost,
            'avg_hourly_cost_Php': round(total_cost / 24, 2),
            'hourly_breakdown': hourly_kwh,
            'recommendations': ["Using fallback prediction mode"],
            'room_type': 'Laboratory' if room_type == 'lab' else 'Office',
            'date': f"{year}-{month}-{day}",
            'day_name': day_name,
            'is_weekend': False
        }

    def get_available_rooms(self):
        """Get list of available rooms for dropdown WITHOUT DUPLICATES"""
        # Define the rooms we want to show
        available_rooms = []

        # Level 1 rooms: 101-111
        for i in range(101, 112):
            available_rooms.append(str(i))

        # Level 7 rooms: 701-705
        for i in range(701, 706):
            available_rooms.append(str(i))

        # Check if we have data preprocessor to get actual rooms
        if self.DataPreprocessor is not None:
            try:
                preprocessor = self.DataPreprocessor(self.assets_dir)
                data = preprocessor.load_energy_data()
                if not data.empty:
                    # Get actual rooms from data
                    actual_rooms = preprocessor.get_prediction_rooms(data)
                    # Filter to only include our defined rooms
                    filtered_rooms = [r for r in actual_rooms if r in available_rooms]
                    if filtered_rooms:
                        available_rooms = filtered_rooms
            except:
                pass  # Use default rooms if preprocessor fails

        # REMOVE DUPLICATES and sort
        unique_rooms = []
        seen = set()
        for room in sorted(available_rooms, key=lambda x: int(x)):
            if room not in seen:
                seen.add(room)
                unique_rooms.append(room)

        print(f"Available rooms (unique): {unique_rooms}")
        return unique_rooms

    def predict_range(self, start_date, end_date, room_number):
        """Predict for a date range"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        predictions = []

        for date in dates:
            pred = self.predict(date.year, date.month, date.day, room_number)
            predictions.append(pred)

        return pd.DataFrame(predictions)


# For backward compatibility
def get_predictor():
    """Helper function for dashboard compatibility"""
    return EnergyPredictor()


if __name__ == "__main__":
    # Test the predictor with different days
    print("=" * 60)
    print("TESTING ENERGY PREDICTOR - WEEKEND HANDLING")
    print("=" * 60)

    predictor = EnergyPredictor()

    # Test different days
    test_cases = [
        ("Weekday - Monday", 2024, 12, 2, '101'),   # Dec 2, 2024 = Monday
        ("Weekday - Friday", 2024, 12, 6, '101'),   # Dec 6, 2024 = Friday
        ("Saturday", 2024, 12, 7, '101'),           # Dec 7, 2024 = Saturday
        ("Sunday", 2024, 12, 1, '101'),             # Dec 1, 2024 = Sunday
        ("Weekday Lab", 2024, 12, 3, '705'),        # Dec 3, 2024 = Tuesday, Room 705 (Lab)
        ("Saturday Lab", 2024, 12, 7, '705'),       # Dec 7, 2024 = Saturday, Room 705
        ("Sunday Lab", 2024, 12, 1, '705'),         # Dec 1, 2024 = Sunday, Room 705
    ]

    for test_name, year, month, day, room in test_cases:
        print(f"\n{'='*40}")
        print(f"TEST: {test_name}")
        print(f"Room: {room}, Date: {year}-{month}-{day}")

        result = predictor.predict(year, month, day, room)

        print(f"Day: {result['day_name']}")
        print(f"Total kWh: {result['total_daily_kWh']} kWh")
        print(f"Total Cost: ₱{result['total_cost_Php']}")
        print(f"Recommendations: {result['recommendations'][0]}")

        # Check if weekend prediction is working
        if result['day_name'] in ['Saturday', 'Sunday']:
            if result['total_daily_kWh'] > 2:  # Should be very low
                print(f"❌ ERROR: {result['day_name']} has high prediction: {result['total_daily_kWh']} kWh")
            else:
                print(f"✅ CORRECT: {result['day_name']} has low prediction: {result['total_daily_kWh']} kWh")

    print(f"\nAvailable rooms: {predictor.get_available_rooms()}")

