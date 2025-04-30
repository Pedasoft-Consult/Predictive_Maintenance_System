import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import logging
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf


class PerformanceMonitor:
    """
    Monitor model performance and detect concept drift.
    """

    def __init__(self, log_dir='./logs', metrics_file='performance_metrics.csv',
                 drift_threshold=0.2, evaluation_window=100):
        """
        Initialize the performance monitor.

        Args:
            log_dir (str): Directory to store logs
            metrics_file (str): File to store performance metrics
            drift_threshold (float): Threshold for concept drift detection
            evaluation_window (int): Number of samples before performance evaluation
        """
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, metrics_file)
        self.drift_threshold = drift_threshold
        self.evaluation_window = evaluation_window

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Initialize data storage
        self.predictions = []
        self.true_values = []
        self.timestamps = []

        # Initialize metrics history
        self.metrics_history = self._load_metrics_history()

        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        """
        Set up logging for the performance monitor.
        """
        # Configure logging
        log_path = os.path.join(self.log_dir, 'performance_monitor.log')
        self.logger = logging.getLogger('performance_monitor')
        self.logger.setLevel(logging.INFO)

        # Add file handler if not already present
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Also log to console
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _load_metrics_history(self):
        """
        Load metrics history from file.

        Returns:
            pd.DataFrame: Metrics history
        """
        if os.path.exists(self.metrics_file):
            return pd.read_csv(self.metrics_file)
        else:
            return pd.DataFrame(columns=['timestamp', 'rmse', 'mae', 'r2', 'drift_detected'])

    def add_observation(self, prediction, true_value, timestamp=None):
        """
        Add a new observation to the monitor.

        Args:
            prediction (float): Predicted value
            true_value (float): True value
            timestamp (str): Timestamp (optional)
        """
        # Skip if true value is not available
        if true_value is None or np.isnan(true_value):
            return

        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Add to storage
        self.predictions.append(float(prediction))
        self.true_values.append(float(true_value))
        self.timestamps.append(timestamp)

        # Check if we need to evaluate performance
        if len(self.predictions) >= self.evaluation_window:
            self.evaluate_performance()

    def evaluate_performance(self):
        """
        Evaluate model performance and check for concept drift.

        Returns:
            dict: Performance metrics
        """
        # Check if we have enough data
        if len(self.predictions) < self.evaluation_window:
            self.logger.warning("Not enough data for performance evaluation.")
            return None

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(self.true_values, self.predictions))
        mae = mean_absolute_error(self.true_values, self.predictions)
        r2 = r2_score(self.true_values, self.predictions)

        # Check for concept drift
        drift_detected = False
        if len(self.metrics_history) > 0:
            # Get last recorded RMSE
            last_rmse = self.metrics_history['rmse'].iloc[-1]

            # Calculate relative change
            if last_rmse > 0:
                relative_change = abs(rmse - last_rmse) / last_rmse
                drift_detected = relative_change > self.drift_threshold

                if drift_detected:
                    self.logger.warning(
                        f"Concept drift detected! RMSE change: {relative_change:.4f}, "
                        f"Threshold: {self.drift_threshold:.4f}"
                    )

        # Current timestamp
        current_time = datetime.now().isoformat()

        # Add metrics to history
        new_metrics = pd.DataFrame({
            'timestamp': [current_time],
            'rmse': [rmse],
            'mae': [mae],
            'r2': [r2],
            'drift_detected': [drift_detected]
        })

        self.metrics_history = pd.concat([self.metrics_history, new_metrics], ignore_index=True)

        # Save metrics history
        self.metrics_history.to_csv(self.metrics_file, index=False)

        # Log metrics
        self.logger.info(
            f"Performance metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, "
            f"Drift detected: {drift_detected}"
        )

        # Create metrics dictionary for return
        metrics = {
            'timestamp': current_time,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'drift_detected': drift_detected,
            'n_samples': len(self.predictions)
        }

        # Reset data storage
        self.predictions = []
        self.true_values = []
        self.timestamps = []

        return metrics

    def visualize_metrics_history(self):
        """
        Visualize the metrics history.

        Returns:
            bool: True if visualization was successful, False otherwise
        """
        if len(self.metrics_history) < 2:
            self.logger.warning("Not enough metrics data for visualization.")
            return False

        try:
            # Create directory for visualizations
            os.makedirs('visualizations', exist_ok=True)

            # Plot metrics over time
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Convert timestamp to datetime
            self.metrics_history['datetime'] = pd.to_datetime(self.metrics_history['timestamp'])

            # Plot RMSE and MAE
            ax1.plot(self.metrics_history['datetime'], self.metrics_history['rmse'], 'b-', label='RMSE')
            ax1.plot(self.metrics_history['datetime'], self.metrics_history['mae'], 'g-', label='MAE')

            # Highlight drift points
            drift_points = self.metrics_history[self.metrics_history['drift_detected']]
            if len(drift_points) > 0:
                ax1.scatter(drift_points['datetime'], drift_points['rmse'],
                            color='red', s=100, zorder=5, label='Drift Detected')

            ax1.set_title('Error Metrics Over Time')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Error')
            ax1.legend()
            ax1.grid(True)

            # Plot R² score
            ax2.plot(self.metrics_history['datetime'], self.metrics_history['r2'], 'r-', label='R²')
            ax2.set_title('R² Score Over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('R²')
            ax2.set_ylim(0, 1)
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig('visualizations/performance_metrics_history.png')
            plt.close()

            self.logger.info("Metrics history visualization saved to visualizations/performance_metrics_history.png")
            return True

        except Exception as e:
            self.logger.error(f"Error visualizing metrics history: {e}")
            return False


class ModelRetrainer:
    """
    Retrainer for predictive maintenance models.
    """

    def __init__(self, models_dir='./models', data_dir='./data',
                 retraining_interval=7, min_samples=1000):
        """
        Initialize the model retrainer.

        Args:
            models_dir (str): Directory containing the models
            data_dir (str): Directory containing the data
            retraining_interval (int): Interval between retraining in days
            min_samples (int): Minimum number of samples needed for retraining
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.retraining_interval = retraining_interval
        self.min_samples = min_samples

        # Create directories if they don't exist
        os.makedirs(os.path.join(models_dir, 'archive'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'collected'), exist_ok=True)

        # Initialize data storage
        self.collected_data = []

        # Set up logging
        self._setup_logging()

        # Load retraining history
        self.retraining_history = self._load_retraining_history()

        # Check last retraining date
        self.last_retraining_date = self._get_last_retraining_date()

    def _setup_logging(self):
        """
        Set up logging for the model retrainer.
        """
        # Configure logging
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, 'model_retrainer.log')
        self.logger = logging.getLogger('model_retrainer')
        self.logger.setLevel(logging.INFO)

        # Add file handler if not already present
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Also log to console
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _load_retraining_history(self):
        """
        Load retraining history from file.

        Returns:
            dict: Retraining history
        """
        history_file = os.path.join(self.models_dir, 'retraining_history.json')

        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except:
                self.logger.error(f"Error loading retraining history. Creating new history.")
                return {
                    'retraining_dates': [],
                    'models': {},
                    'metrics': {}
                }
        else:
            return {
                'retraining_dates': [],
                'models': {},
                'metrics': {}
            }

    def _save_retraining_history(self):
        """
        Save retraining history to file.
        """
        history_file = os.path.join(self.models_dir, 'retraining_history.json')

        try:
            with open(history_file, 'w') as f:
                json.dump(self.retraining_history, f, indent=4)
            self.logger.info(f"Retraining history saved to {history_file}")
        except Exception as e:
            self.logger.error(f"Error saving retraining history: {e}")

    def _get_last_retraining_date(self):
        """
        Get the date of the last retraining.

        Returns:
            datetime: Last retraining date or None if never retrained
        """
        if 'retraining_dates' in self.retraining_history and self.retraining_history['retraining_dates']:
            last_date_str = self.retraining_history['retraining_dates'][-1]
            return datetime.fromisoformat(last_date_str)
        else:
            return None

    def collect_data_point(self, data_point):
        """
        Collect a data point for future retraining.

        Args:
            data_point (dict): Data point with features and target
        """
        # Skip if no true value or sequence
        if data_point.get('true_rul') is None or data_point.get('sequence') is None:
            return

        # Create a new entry with the necessary data
        entry = {
            'unit_nr': data_point['unit_nr'],
            'cycle': data_point['cycle'],
            'sequence': data_point['sequence'],
            'true_rul': data_point['true_rul'],
            'timestamp': data_point.get('timestamp', datetime.now().isoformat())
        }

        # Add to collected data
        self.collected_data.append(entry)

        # Save periodically
        if len(self.collected_data) % 100 == 0:
            self._save_collected_data()

    def _save_collected_data(self):
        """
        Save collected data to file.
        """
        if not self.collected_data:
            return

        # Create filename with current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'collected_data_{timestamp}.npz'
        file_path = os.path.join(self.data_dir, 'collected', filename)

        try:
            # Convert data to numpy arrays
            unit_nrs = np.array([d['unit_nr'] for d in self.collected_data])
            cycles = np.array([d['cycle'] for d in self.collected_data])
            sequences = np.array([d['sequence'] for d in self.collected_data])
            rul_values = np.array([d['true_rul'] for d in self.collected_data])
            timestamps = np.array([d['timestamp'] for d in self.collected_data])

            # Save as compressed numpy file
            np.savez_compressed(
                file_path,
                unit_nr=unit_nrs,
                cycle=cycles,
                sequence=sequences,
                rul=rul_values,
                timestamp=timestamps
            )

            # Log saved data
            self.logger.info(f"Saved {len(self.collected_data)} data points to {file_path}")

            # Clear collected data
            self.collected_data = []

        except Exception as e:
            self.logger.error(f"Error saving collected data: {e}")

    def check_retraining_needed(self, force=False, drift_detected=False):
        """
        Check if retraining is needed based on time interval or concept drift.

        Args:
            force (bool): Force retraining regardless of conditions
            drift_detected (bool): Whether concept drift has been detected

        Returns:
            bool: True if retraining is needed, False otherwise
        """
        # If forcing retraining
        if force:
            self.logger.info("Retraining forced.")
            return True

        # If drift detected
        if drift_detected:
            self.logger.info("Retraining needed due to concept drift.")
            return True

        # Check interval since last retraining
        if self.last_retraining_date:
            days_since_last = (datetime.now() - self.last_retraining_date).days
            if days_since_last >= self.retraining_interval:
                self.logger.info(
                    f"Retraining needed due to time interval. Days since last retraining: {days_since_last}"
                )
                return True
        else:
            # If never retrained, check if we have enough data
            total_samples = self._count_collected_samples()
            if total_samples >= self.min_samples:
                self.logger.info(
                    f"Initial training needed. Collected samples: {total_samples}"
                )
                return True

        return False

    def _count_collected_samples(self):
        """
        Count the total number of collected samples.

        Returns:
            int: Total number of samples
        """
        # Count samples in memory
        in_memory_count = len(self.collected_data)

        # Count samples in saved files
        saved_count = 0
        collected_dir = os.path.join(self.data_dir, 'collected')
        for filename in os.listdir(collected_dir):
            if filename.endswith('.npz'):
                try:
                    file_path = os.path.join(collected_dir, filename)
                    with np.load(file_path) as data:
                        saved_count += len(data['rul'])
                except:
                    continue

        return in_memory_count + saved_count

    def prepare_training_data(self):
        """
        Prepare data for model retraining.

        Returns:
            tuple: (X_train_tabular, y_train_tabular, X_train_sequence, y_train_sequence)
        """
        # Save any remaining data in memory
        self._save_collected_data()

        # Initialize arrays for combined data
        all_unit_nrs = []
        all_cycles = []
        all_sequences = []
        all_ruls = []

        # Load and combine all collected data
        collected_dir = os.path.join(self.data_dir, 'collected')
        for filename in os.listdir(collected_dir):
            if filename.endswith('.npz'):
                try:
                    file_path = os.path.join(collected_dir, filename)
                    with np.load(file_path) as data:
                        all_unit_nrs.append(data['unit_nr'])
                        all_cycles.append(data['cycle'])
                        all_sequences.append(data['sequence'])
                        all_ruls.append(data['rul'])
                except Exception as e:
                    self.logger.error(f"Error loading data file {filename}: {e}")
                    continue

        # Combine data
        if all_unit_nrs:
            all_unit_nrs = np.concatenate(all_unit_nrs)
            all_cycles = np.concatenate(all_cycles)
            all_sequences = np.concatenate(all_sequences)
            all_ruls = np.concatenate(all_ruls)
        else:
            self.logger.error("No collected data found for retraining.")
            return None, None, None, None

        # Log data summary
        self.logger.info(f"Prepared {len(all_ruls)} samples for retraining")

        # Create sequence data (for LSTM)
        X_train_sequence = np.array(all_sequences)
        y_train_sequence = np.array(all_ruls)

        # Create tabular data (for XGBoost)
        # Use the last timestep of each sequence
        X_train_tabular = X_train_sequence[:, -1, :]
        y_train_tabular = y_train_sequence

        return X_train_tabular, y_train_tabular, X_train_sequence, y_train_sequence

    def retrain_models(self, force=False, drift_detected=False):
        """
        Retrain machine learning models if needed.

        Args:
            force (bool): Force retraining regardless of conditions
            drift_detected (bool): Whether concept drift has been detected

        Returns:
            bool: True if retraining was performed, False otherwise
        """
        # Check if retraining is needed
        if not self.check_retraining_needed(force, drift_detected):
            self.logger.info("Retraining not needed at this time.")
            return False

        # Prepare data for retraining
        X_train_tabular, y_train_tabular, X_train_sequence, y_train_sequence = self.prepare_training_data()

        if X_train_tabular is None or len(X_train_tabular) < self.min_samples:
            self.logger.warning(f"Not enough data for retraining. Minimum required: {self.min_samples}")
            return False

        # Log retraining start
        self.logger.info(f"Starting model retraining with {len(X_train_tabular)} samples")

        try:
            # Archive current models
            self._archive_current_models()

            # Retrain models
            trad_model = self._retrain_traditional_model(X_train_tabular, y_train_tabular)
            dl_model = self._retrain_deep_learning_model(X_train_sequence, y_train_sequence)

            # Update retraining history
            current_time = datetime.now().isoformat()
            self.retraining_history['retraining_dates'].append(current_time)
            self.last_retraining_date = datetime.now()

            # Save retraining history
            self._save_retraining_history()

            # Log success
            self.logger.info("Model retraining completed successfully.")

            return True

        except Exception as e:
            self.logger.error(f"Error during model retraining: {e}")
            return False

    def _archive_current_models(self):
        """
        Archive current models before retraining.
        """
        # Create timestamp for archive
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Archive XGBoost model
        xgb_path = os.path.join(self.models_dir, 'xgboost_model.pkl')
        if os.path.exists(xgb_path):
            archive_path = os.path.join(self.models_dir, 'archive', f'xgboost_model_{timestamp}.pkl')
            try:
                os.rename(xgb_path, archive_path)
                self.logger.info(f"Archived XGBoost model to {archive_path}")
            except Exception as e:
                self.logger.error(f"Error archiving XGBoost model: {e}")

        # Archive LSTM model
        lstm_path = os.path.join(self.models_dir, 'lstm_model.h5')
        if os.path.exists(lstm_path):
            archive_path = os.path.join(self.models_dir, 'archive', f'lstm_model_{timestamp}.h5')
            try:
                os.rename(lstm_path, archive_path)
                self.logger.info(f"Archived LSTM model to {archive_path}")
            except Exception as e:
                self.logger.error(f"Error archiving LSTM model: {e}")

    def _retrain_traditional_model(self, X_train, y_train):
        """
        Retrain the traditional ML model (XGBoost).

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            model: Trained model
        """
        try:
            from xgboost import XGBRegressor

            # Initialize model
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )

            # Train model
            model.fit(X_train, y_train)

            # Save model
            model_path = os.path.join(self.models_dir, 'xgboost_model.pkl')
            joblib.dump(model, model_path)
            self.logger.info(f"XGBoost model retrained and saved to {model_path}")

            return model

        except Exception as e:
            self.logger.error(f"Error retraining XGBoost model: {e}")
            raise

    def _retrain_deep_learning_model(self, X_train, y_train):
        """
        Retrain the deep learning model (LSTM).

        Args:
            X_train: Training sequences
            y_train: Training targets

        Returns:
            model: Trained model
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

            # Get input shape
            input_shape = (X_train.shape[1], X_train.shape[2])

            # Build model
            model = Sequential([
                # First LSTM layer
                LSTM(128, input_shape=input_shape, return_sequences=True),
                BatchNormalization(),
                Dropout(0.2),

                # Second LSTM layer
                LSTM(64, return_sequences=False),
                BatchNormalization(),
                Dropout(0.2),

                # Dense layers
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),

                # Output layer
                Dense(1)  # Linear activation for regression
            ])

            # Compile model
            model.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae']
            )

            # Set up callbacks
            model_path = os.path.join(self.models_dir, 'lstm_model.h5')
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            ]

            # Train model with validation split
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )

            self.logger.info(f"LSTM model retrained and saved to {model_path}")

            return model

        except Exception as e:
            self.logger.error(f"Error retraining LSTM model: {e}")
            raise


def main():
    """
    Main function to demonstrate the monitoring and retraining functionality.
    """
    # Initialize components
    performance_monitor = PerformanceMonitor(
        log_dir='./logs',
        metrics_file='performance_metrics.csv',
        drift_threshold=0.2,
        evaluation_window=100
    )

    model_retrainer = ModelRetrainer(
        models_dir='./models',
        data_dir='./data',
        retraining_interval=7,
        min_samples=1000
    )

    # Simulate adding observations
    print("Simulating performance monitoring and retraining...")

    # Generate some synthetic data
    n_samples = 200
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n_samples)]

    # Simulate initial good performance
    for i in range(100):
        # True RUL values between 50 and 150
        true_rul = np.random.uniform(50, 150)
        # Predictions with small errors
        pred_rul = true_rul + np.random.normal(0, 5)

        # Add to monitor
        performance_monitor.add_observation(pred_rul, true_rul, timestamps[i].isoformat())

        # Collect data point for retraining
        data_point = {
            'unit_nr': np.random.randint(1, 10),
            'cycle': np.random.randint(1, 100),
            'sequence': np.random.normal(0, 1, (10, 21)),  # 10 timesteps, 21 features
            'true_rul': true_rul,
            'timestamp': timestamps[i].isoformat()
        }
        model_retrainer.collect_data_point(data_point)

    # Evaluate performance
    metrics = performance_monitor.evaluate_performance()
    print(f"Initial metrics: {metrics}")

    # Simulate concept drift
    for i in range(100, 200):
        # True RUL values between 50 and 150
        true_rul = np.random.uniform(50, 150)
        # Add systematic bias to simulate drift
        bias = (i - 100) * 0.2  # Increasing bias
        pred_rul = true_rul + bias + np.random.normal(0, 8)  # Increased noise too

        # Add to monitor
        performance_monitor.add_observation(pred_rul, true_rul, timestamps[i].isoformat())

        # Collect data point for retraining
        data_point = {
            'unit_nr': np.random.randint(1, 10),
            'cycle': np.random.randint(1, 100),
            'sequence': np.random.normal(0, 1, (10, 21)),
            'true_rul': true_rul,
            'timestamp': timestamps[i].isoformat()
        }
        model_retrainer.collect_data_point(data_point)

    # Evaluate performance again
    metrics = performance_monitor.evaluate_performance()
    print(f"Metrics after drift: {metrics}")

    # Visualize metrics history
    performance_monitor.visualize_metrics_history()

    # Check if retraining is needed
    drift_detected = metrics.get('drift_detected', False)
    retraining_needed = model_retrainer.check_retraining_needed(
        drift_detected=drift_detected
    )

    print(f"Retraining needed: {retraining_needed}")

    # Note about actual retraining
    if retraining_needed:
        print("In a real system, retraining would now be triggered.")
        print("We don't actually retrain here to avoid unnecessary computation.")

    print("Monitoring and retraining simulation completed.")


if __name__ == "__main__":
    main()