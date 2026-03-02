"""
Complete Data Preprocessing Module for Energy Prediction System
Handles loading, cleaning, and preparing data for ALL rooms
Filters for specific rooms (Level 1: 101-111, Level 7: 701-705) in prediction
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import OneHotEncoder
import joblib

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles all data preprocessing tasks for energy prediction"""

    def __init__(self, assets_dir='assets'):
        self.assets_dir = assets_dir
        self.encoder = None

    def load_energy_data(self, csv_path=None):
        """
        Load ALL per-second energy data from sample_data.csv
        Aggregates to daily totals and adds features
        """
        if csv_path is None:
            csv_path = os.path.join(self.assets_dir, 'sample_data.csv')

        print(f"Loading energy data from: {csv_path}")

        try:
            # Load the CSV file
            df = pd.read_csv(csv_path)

            # Debug information
            print(f"CSV loaded successfully. Shape: {df.shape}")
            print(f"Original columns: {df.columns.tolist()}")

            # Standardize column names (FIXED: Handle Room_Type separately)
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if 'date' in col_lower:
                    column_mapping[col] = 'date'
                elif 'time' in col_lower or 'hour' in col_lower:
                    column_mapping[col] = 'time'
                elif 'energy' in col_lower or 'kwh' in col_lower:
                    column_mapping[col] = 'energy_kwh'
                elif col == 'Room':  # Specifically handle Room column
                    column_mapping[col] = 'room'
                elif col == 'Room_Type':  # Specifically handle Room_Type column
                    column_mapping[col] = 'room_type'
                elif 'voltage' in col_lower or 'volt' in col_lower:
                    column_mapping[col] = 'voltage'
                elif 'current' in col_lower:
                    column_mapping[col] = 'current'

            # Apply column renaming
            df = df.rename(columns=column_mapping)
            print(f"Renamed columns: {column_mapping}")
            print(f"New columns: {df.columns.tolist()}")

            # Check for required columns
            required_cols = ['date', 'energy_kwh', 'room']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"Error: Missing required columns: {missing_cols}")
                print(f"Available columns: {df.columns.tolist()}")
                return pd.DataFrame()

            # Ensure room is string and clean
            df['room'] = df['room'].astype(str).str.strip()

            # Show all unique rooms found
            all_rooms = sorted(df['room'].unique().tolist())
            print(f"All rooms in dataset: {all_rooms}")
            print(f"Total rooms: {len(all_rooms)}")

            # If room_type column exists, show unique types
            if 'room_type' in df.columns:
                df['room_type'] = df['room_type'].astype(str).str.strip()
                room_types = df['room_type'].unique().tolist()
                print(f"Room types: {room_types}")

            # Parse timestamps
            if 'time' in df.columns:
                # Handle different time formats
                try:
                    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
                except:
                    df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
            else:
                df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')

            # Handle invalid timestamps
            invalid_timestamps = df['timestamp'].isnull().sum()
            if invalid_timestamps > 0:
                print(f"Warning: {invalid_timestamps} invalid timestamps found, dropping them")
                df = df.dropna(subset=['timestamp'])

            # Convert energy to numeric
            df['energy_kwh'] = pd.to_numeric(df['energy_kwh'], errors='coerce')

            # Drop rows with invalid energy
            invalid_energy = df['energy_kwh'].isnull().sum()
            if invalid_energy > 0:
                print(f"Dropping {invalid_energy} rows with invalid energy values")
                df = df.dropna(subset=['energy_kwh'])

            # Calculate date-only column for aggregation
            df['date_only'] = df['timestamp'].dt.date

            # Aggregate per-second data to daily totals
            print("Aggregating per-second data to daily totals...")

            # Start with energy sum
            daily_agg = df.groupby(['room', 'date_only']).agg({
                'energy_kwh': 'sum'
            }).reset_index()

            # Add other metrics if available
            numeric_cols = ['voltage', 'current']
            for col in numeric_cols:
                if col in df.columns:
                    col_avg = df.groupby(['room', 'date_only'])[col].mean()
                    daily_agg = daily_agg.merge(col_avg, on=['room', 'date_only'], how='left')
                    print(f"  Added {col} average")

            # Add room_type if it exists
            if 'room_type' in df.columns:
                # Get the most common room_type for each room-date combination
                room_type_map = df.groupby(['room', 'date_only'])['room_type'].agg(
                    lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
                ).reset_index()
                daily_agg = daily_agg.merge(room_type_map, on=['room', 'date_only'], how='left')
                print("  Added room_type")

            print(f"Aggregated to {len(daily_agg)} daily records")

            # Add date features
            daily_agg['date'] = pd.to_datetime(daily_agg['date_only'])
            daily_agg['year'] = daily_agg['date'].dt.year
            daily_agg['month'] = daily_agg['date'].dt.month
            daily_agg['day'] = daily_agg['date'].dt.day
            daily_agg['day_of_week'] = daily_agg['date'].dt.dayofweek
            daily_agg['is_weekend'] = daily_agg['day_of_week'].isin([5, 6]).astype(int)

            # Add seasonal features (cyclic encoding)
            daily_agg['month_sin'] = np.sin(2 * np.pi * daily_agg['month'] / 12)
            daily_agg['month_cos'] = np.cos(2 * np.pi * daily_agg['month'] / 12)

            # Add floor level based on room number
            daily_agg['floor'] = daily_agg['room'].apply(self._get_floor_level)

            # If room_type doesn't exist, create it based on floor
            if 'room_type' not in daily_agg.columns:
                daily_agg['room_type'] = daily_agg['floor'].apply(
                    lambda x: 'lab' if x == 7 else 'office'
                )

            # Summary statistics
            print(f"\n=== DATA SUMMARY ===")
            print(f"Date range: {daily_agg['date'].min()} to {daily_agg['date'].max()}")
            print(f"Total records: {len(daily_agg)}")
            print(f"Total energy: {daily_agg['energy_kwh'].sum():.2f} kWh")
            print(f"Average daily energy per room: {daily_agg.groupby('room')['energy_kwh'].mean().mean():.2f} kWh")

            # Show rooms by floor
            print(f"\n=== ROOMS BY FLOOR ===")
            for floor in sorted(daily_agg['floor'].unique()):
                floor_rooms = daily_agg[daily_agg['floor'] == floor]['room'].unique()
                room_count = len(floor_rooms)
                sample_rooms = sorted(floor_rooms)[:5]  # Show first 5
                sample_str = ', '.join(sample_rooms)
                if room_count > 5:
                    sample_str += f", ... (+{room_count-5} more)"
                print(f"Floor {floor}: {room_count} rooms - {sample_str}")


                # After loading and aggregating data:
                print("Analyzing day-of-week patterns...")

                # Analyze day distribution
                day_distribution = daily_agg['day_of_week'].value_counts().sort_index()
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

                print("\n=== DAY OF WEEK DISTRIBUTION ===")
                for day_num in range(7):
                    count = day_distribution.get(day_num, 0)
                    print(f"{day_names[day_num]}: {count} records ({count / len(daily_agg) * 100:.1f}%)")

                # Check for missing weekend data
                weekend_days = [5, 6]  # Saturday, Sunday
                weekend_records = daily_agg[daily_agg['day_of_week'].isin(weekend_days)]

                if len(weekend_records) == 0:
                    print("\n⚠️  WARNING: No weekend data found in dataset!")
                    print("   Sunday predictions will be forced to zero.")
                    print("   Saturday predictions will be reduced to 10% of weekdays.")
                elif len(weekend_records) < len(daily_agg) * 0.1:  # Less than 10% weekend data
                    print(f"\n⚠️  WARNING: Limited weekend data ({len(weekend_records)} records)")
                    print("   Consider adding synthetic weekend data or adjusting predictions.")

                return daily_agg

        except Exception as e:
            print(f"✗ Error loading energy data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _get_floor_level(self, room_str):
        """Extract floor level from room number (first digit)"""
        try:
            room_str = str(room_str).strip()
            if room_str and room_str[0].isdigit():
                return int(room_str[0])
            return 0
        except:
            return 0

    def filter_rooms_for_prediction(self, data=None, allowed_floors=[1, 7], allowed_ranges=None):
        """
        Filter data for specific floors and room ranges

        Parameters:
        -----------
        data: DataFrame or None
            If None, loads data automatically
        allowed_floors: list
            Floor levels to include (default: [1, 7])
        allowed_ranges: dict or None
            Specific room ranges per floor, e.g.,
            {'1': [101, 111], '7': [701, 705]}
        """
        if data is None:
            data = self.load_energy_data()

        if data.empty:
            print("Warning: No data to filter")
            return data

        # First filter by floor level
        filtered = data[data['floor'].isin(allowed_floors)].copy()

        # Then apply specific room ranges if provided
        if allowed_ranges:
            room_mask = []
            for _, row in filtered.iterrows():
                try:
                    room_num = int(row['room'])
                    floor = row['floor']

                    if str(floor) in allowed_ranges:
                        min_room, max_room = allowed_ranges[str(floor)]
                        room_mask.append(min_room <= room_num <= max_room)
                    else:
                        room_mask.append(False)
                except:
                    room_mask.append(False)

            filtered = filtered[room_mask]

        print(f"\n=== FILTERED DATA ===")
        print(f"Filtered to {len(filtered)} records")
        print(f"Allowed floors: {allowed_floors}")

        if allowed_ranges:
            print(f"Room ranges: {allowed_ranges}")

        # Show rooms after filtering
        for floor in sorted(filtered['floor'].unique()):
            floor_rooms = filtered[filtered['floor'] == floor]['room'].unique()
            print(f"Floor {floor} rooms: {sorted(floor_rooms)}")

        return filtered

    def get_prediction_rooms(self, data=None):
        """
        Get ONLY the rooms for prediction tab:
        - Level 1: Rooms 101-111
        - Level 7: Rooms 701-705

        Returns list of room numbers as strings
        """
        if data is None:
            data = self.load_energy_data()

        if data.empty:
            print("Warning: No data available")
            return []

        # Define allowed ranges for prediction tab
        allowed_ranges = {
            '1': [101, 111],  # Level 1: Rooms 101-111
            '7': [701, 705]   # Level 7: Rooms 701-705
        }

        prediction_rooms = []

        for _, row in data.iterrows():
            try:
                room_num = int(row['room'])
                floor = row['floor']

                if str(floor) in allowed_ranges:
                    min_room, max_room = allowed_ranges[str(floor)]
                    if min_room <= room_num <= max_room:
                        if room_num not in prediction_rooms:
                            prediction_rooms.append(str(room_num))
            except ValueError:
                continue

        # Sort numerically
        prediction_rooms.sort(key=lambda x: int(x))

        print(f"\n=== PREDICTION TAB ROOMS ===")
        print(f"Total: {len(prediction_rooms)} rooms")

        # Group by floor for display
        floor_groups = {}
        for room in prediction_rooms:
            floor = room[0]  # First digit
            if floor not in floor_groups:
                floor_groups[floor] = []
            floor_groups[floor].append(room)

        for floor in sorted(floor_groups.keys()):
            rooms = sorted(floor_groups[floor], key=lambda x: int(x))
            print(f"Floor {floor}: {rooms}")

        return prediction_rooms

    def create_encoder(self, data, column='room_type'):
        """
        Create and save OneHotEncoder for categorical features

        Parameters:
        -----------
        data: DataFrame containing the column to encode
        column: Column name to encode (default: 'room_type')

        Returns:
        --------
        encoder: Trained OneHotEncoder
        """
        print(f"\n=== CREATING ENCODER ===")

        if column not in data.columns:
            print(f"Warning: Column '{column}' not found in data")
            return None

        # Get unique values
        unique_values = data[column].unique()
        print(f"Unique values in '{column}': {unique_values}")

        # Create encoder
        self.encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            drop=None  # Keep all categories
        )

        # Fit encoder
        self.encoder.fit(data[[column]])

        # Get feature names
        feature_names = self.encoder.get_feature_names_out([column])
        print(f"Encoded features created: {feature_names}")

        # Save encoder
        joblib.dump(self.encoder, 'room_type_encoder.joblib')
        print(f"✓ Encoder saved to 'room_type_encoder.joblib'")

        return self.encoder

    def encode_features(self, data, column='room_type'):
        """Apply encoding to new data"""
        if self.encoder is None:
            print("Encoder not created yet. Call create_encoder() first.")
            return data

        if column not in data.columns:
            print(f"Column '{column}' not found in data")
            return data

        # Transform the column
        encoded = self.encoder.transform(data[[column]])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out([column])
        )

        # Combine with original data
        result = pd.concat([data.drop(column, axis=1), encoded_df], axis=1)

        return result

    def load_historical_rates(self):
        """
        Load 2021-2025 historical electricity rates from CSV

        Returns:
        --------
        DataFrame with columns: ['date', 'rate_per_kWh']
        """
        print(f"\n=== LOADING HISTORICAL RATES ===")

        try:
            csv_path = os.path.join(self.assets_dir, 'energy_rate.csv')
            rates_df = pd.read_csv(csv_path)

            print(f"Loaded rate data. Shape: {rates_df.shape}")
            print(f"Columns: {rates_df.columns.tolist()}")

            # Check required columns
            required_cols = ['year', 'month', 'rate_per_kWh']
            missing_cols = [col for col in required_cols if col not in rates_df.columns]

            if missing_cols:
                print(f"Warning: Missing columns in rate data: {missing_cols}")
                return pd.DataFrame(columns=['date', 'rate_per_kWh'])

            # Create datetime column
            rates_df['date'] = pd.to_datetime(rates_df[['year', 'month']].assign(day=1))

            # Sort by date
            rates_df = rates_df[['date', 'rate_per_kWh']].sort_values('date')

            # Summary
            print(f"Date range: {rates_df['date'].min()} to {rates_df['date'].max()}")
            print(f"Rate range: ₱{rates_df['rate_per_kWh'].min():.2f} to ₱{rates_df['rate_per_kWh'].max():.2f}")
            print(f"Average rate: ₱{rates_df['rate_per_kWh'].mean():.2f}")

            return rates_df

        except FileNotFoundError:
            print(f"✗ Rate file not found: {csv_path}")
            return pd.DataFrame(columns=['date', 'rate_per_kWh'])
        except Exception as e:
            print(f"✗ Error loading historical rates: {e}")
            return pd.DataFrame(columns=['date', 'rate_per_kWh'])

    def prepare_training_data(self, data=None):
        """
        Prepare final dataset for model training

        Returns:
        --------
        DataFrame ready for model training
        """
        if data is None:
            data = self.load_energy_data()

        if data.empty:
            print("Warning: No data available for training")
            return pd.DataFrame()

        print(f"\n=== PREPARING TRAINING DATA ===")

        # Select features for training
        feature_cols = [
            'room', 'year', 'month', 'day', 'day_of_week', 'is_weekend',
            'month_sin', 'month_cos', 'room_type', 'floor'
        ]

        # Ensure all columns exist
        available_cols = [col for col in feature_cols if col in data.columns]
        missing_cols = set(feature_cols) - set(available_cols)

        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")

        # Prepare features and target
        X = data[available_cols]
        y = data['energy_kwh']

        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {available_cols}")

        return X, y

    def get_data_summary(self, data=None):
        """Generate summary statistics of the data"""
        if data is None:
            data = self.load_energy_data()

        if data.empty:
            return "No data available"

        summary = []
        summary.append("=== DATA SUMMARY ===")
        summary.append(f"Total records: {len(data):,}")
        summary.append(f"Date range: {data['date'].min()} to {data['date'].max()}")
        summary.append(f"Number of rooms: {data['room'].nunique()}")
        summary.append(f"Number of floors: {data['floor'].nunique()}")
        summary.append(f"Total energy: {data['energy_kwh'].sum():,.2f} kWh")
        summary.append(f"Average daily energy: {data['energy_kwh'].mean():.2f} kWh")
        summary.append(f"Min daily energy: {data['energy_kwh'].min():.2f} kWh")
        summary.append(f"Max daily energy: {data['energy_kwh'].max():.2f} kWh")

        # By floor
        summary.append("\n=== BY FLOOR ===")
        for floor in sorted(data['floor'].unique()):
            floor_data = data[data['floor'] == floor]
            summary.append(f"Floor {floor}: {len(floor_data)} records, "
                          f"{floor_data['room'].nunique()} rooms, "
                          f"Avg: {floor_data['energy_kwh'].mean():.2f} kWh")

        return "\n".join(summary)


# Legacy functions for backward compatibility
def load_and_preprocess_data():
    """Legacy function for compatibility with existing code"""
    processor = DataPreprocessor()
    return processor.load_energy_data()

def load_historical_rates():
    """Legacy function for compatibility with existing code"""
    processor = DataPreprocessor()
    return processor.load_historical_rates()


if __name__ == "__main__":
    # Test the data preprocessing
    print("=" * 60)
    print("TESTING DATA PREPROCESSING MODULE")
    print("=" * 60)

    # Initialize processor
    processor = DataPreprocessor()

    # Load all data
    data = processor.load_energy_data()

    if not data.empty:
        # Show data summary
        print("\n" + processor.get_data_summary(data))

        # Get rooms for prediction tab
        prediction_rooms = processor.get_prediction_rooms(data)

        # Filter data for prediction rooms
        filtered_data = processor.filter_rooms_for_prediction(
            data,
            allowed_floors=[1, 7],
            allowed_ranges={'1': [101, 111], '7': [701, 705]}
        )

        # Create encoder
        if 'room_type' in data.columns:
            encoder = processor.create_encoder(data)

        # Load historical rates
        rates = processor.load_historical_rates()

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)

    else:
        print("✗ Failed to load data. Check your CSV file.")