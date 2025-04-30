import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
import onnxruntime as ort
from datetime import datetime
import threading
import queue
import logging
import json
from collections import deque


class DataSimulator:
    """
    Simulates real-time sensor data streaming.
    """

    def __init__(self, data_path, sequence_length=10, delay=0.5):
        """
        Initialize the data simulator.

        Args:
            data_path (str): Path to test data
            sequence_length (int): Length of input sequences
            delay (float): Delay between data points (in seconds)
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.delay = delay
        self.running = False
        self.data_queue = queue.Queue(maxsize=100)
        self.current_unit = None

    def load_data(self):
        """
        Load test data for simulation.

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Load preprocessed data
            X_test_tabular = np.load(os.path.join(self.data_path, "X_test_tabular.npy"), allow_pickle=True)
            X_test_sequence = np.load(os.path.join(self.data_path, "X_test_sequence.npy"), allow_pickle=True)
            y_test = np.load(os.path.join(self.data_path, "y_test_tabular.npy"), allow_pickle=True)

            # If available, load original test data to get unit numbers and cycles
            test_data_file = os.path.join(os.path.dirname(self.data_path), "raw", "test_FD001.txt")
            if os.path.exists(test_data_file):
                # Column names for the dataset
                columns = ['unit_nr', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                          [f'sensor_{i}' for i in range(1, 22)]

                test_data = pd.read_csv(test_data_file, sep=' ', header=None, names=columns)
                test_data = test_data.dropna(axis=1, how='all')

                # Add RUL column (use y_test if shapes match)
                if len(y_test) == len(test_data['unit_nr'].unique()):
                    # Map RUL to units
                    rul_dict = {}
                    current_unit = None
                    rul_idx = 0

                    for i, unit in enumerate(test_data['unit_nr']):
                        if unit != current_unit:
                            current_unit = unit
                            rul_dict[unit] = y_test[rul_idx]
                            rul_idx += 1

                    test_data['RUL'] = test_data['unit_nr'].map(rul_dict)

                return test_data
            else:
                # Create a simple DataFrame from X_test_tabular or X_test_sequence
                if len(X_test_tabular) > 0:
                    data = pd.DataFrame(X_test_tabular)
                    data['RUL'] = y_test
                    # Add unit_nr and cycle columns (assumed)
                    data['unit_nr'] = np.arange(len(data)) // 20 + 1  # Each unit has ~20 cycles
                    data['cycle'] = np.arange(len(data)) % 20 + 1
                    return data
                else:
                    # Create from sequence data (use only the last point of each sequence)
                    data = pd.DataFrame(X_test_sequence[:, -1, :])
                    data['RUL'] = y_test
                    data['unit_nr'] = np.arange(len(data)) // 20 + 1  # Each unit has ~20 cycles
                    data['cycle'] = np.arange(len(data)) % 20 + 1
                    return data

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            # Create synthetic data if loading fails
            return self._create_synthetic_data()

    def _create_synthetic_data(self, n_units=5, cycles_per_unit=100, n_sensors=21):
        """
        Create synthetic data for simulation.

        Args:
            n_units (int): Number of units
            cycles_per_unit (int): Number of cycles per unit
            n_sensors (int): Number of sensors

        Returns:
            pd.DataFrame: Synthetic data
        """
        # Create empty dataframe
        columns = ['unit_nr', 'cycle'] + [f'sensor_{i}' for i in range(1, n_sensors + 1)]
        data = pd.DataFrame(columns=columns)

        # Generate data for each unit
        for unit in range(1, n_units + 1):
            unit_data = []

            for cycle in range(1, cycles_per_unit + 1):
                # Base sensor values (healthy state)
                sensor_values = np.random.normal(0.5, 0.1, n_sensors)

                # Add degradation pattern
                degradation_factor = cycle / cycles_per_unit
                # Sensors 2, 7, 12, 17 show clear degradation patterns
                for s in [1, 6, 11, 16]:  # 0-indexed
                    sensor_values[s] += degradation_factor * np.random.uniform(0.2, 0.5)

                # Add noise
                sensor_values += np.random.normal(0, 0.05, n_sensors)

                # Create row
                row = [unit, cycle] + sensor_values.tolist()
                unit_data.append(row)

            # Add to dataframe
            unit_df = pd.DataFrame(unit_data, columns=columns)
            data = pd.concat([data, unit_df], ignore_index=True)

        # Calculate RUL
        max_cycles = data.groupby('unit_nr')['cycle'].max().reset_index()
        max_cycles.columns = ['unit_nr', 'max_cycle']
        data = data.merge(max_cycles, on='unit_nr', how='left')
        data['RUL'] = data['max_cycle'] - data['cycle']
        data = data.drop('max_cycle', axis=1)

        return data

    def start_simulation(self):
        """
        Start the data simulation in a separate thread.
        """
        if self.running:
            logging.warning("Simulation is already running.")
            return

        self.running = True
        self.simulation_thread = threading.Thread(target=self._simulate_data_stream)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        logging.info("Data simulation started.")

    def stop_simulation(self):
        """
        Stop the data simulation.
        """
        self.running = False
        if hasattr(self, 'simulation_thread') and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
        logging.info("Data simulation stopped.")

    def _simulate_data_stream(self):
        """
        Simulate streaming data from the test dataset.
        """
        # Load data
        data = self.load_data()

        if data is None or len(data) == 0:
            logging.error("No data available for simulation.")
            self.running = False
            return

        # Get unique units
        units = data['unit_nr'].unique()

        # Sequence buffer for each unit
        unit_sequences = {unit: deque(maxlen=self.sequence_length) for unit in units}

        # Sensor columns
        sensor_cols = [col for col in data.columns if 'sensor' in col]

        # Simulate data streaming
        while self.running:
            # Pick a random unit if not already selected
            if self.current_unit is None or np.random.random() < 0.1:  # 10% chance to switch units
                self.current_unit = np.random.choice(units)

            # Get data for the current unit
            unit_data = data[data['unit_nr'] == self.current_unit].sort_values('cycle')

            # If more than 20 rows, select a random 20-row window
            if len(unit_data) > 20:
                start_idx = np.random.randint(0, len(unit_data) - 20)
                unit_data = unit_data.iloc[start_idx:start_idx + 20]

            # Stream each row
            for _, row in unit_data.iterrows():
                if not self.running:
                    break

                # Extract sensor values
                sensor_values = row[sensor_cols].values

                # Add to sequence buffer
                unit_sequences[self.current_unit].append(sensor_values)

                # Create data point with metadata
                data_point = {
                    'unit_nr': int(row['unit_nr']),
                    'cycle': int(row['cycle']),
                    'sensor_values': sensor_values,
                    'sequence': list(unit_sequences[self.current_unit])
                    if len(unit_sequences[self.current_unit]) == self.sequence_length
                    else None,
                    'timestamp': datetime.now().isoformat(),
                    'true_rul': float(row['RUL']) if 'RUL' in row else None
                }

                # Put in queue
                try:
                    self.data_queue.put(data_point, block=False)
                except queue.Full:
                    # If queue is full, remove oldest item
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put(data_point, block=False)
                    except:
                        pass

                # Delay to simulate real-time
                time.sleep(self.delay)

    def get_data_point(self, timeout=1.0):
        """
        Get the next data point from the queue.

        Args:
            timeout (float): Timeout in seconds

        Returns:
            dict: Data point
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class InferenceEngine:
    """
    Real-time inference engine for predictive maintenance.
    """

    def __init__(self, model_dir, model_type='tflite', threshold=30, window_size=10):
        """
        Initialize the inference engine.

        Args:
            model_dir (str): Directory containing the optimized models
            model_type (str): Type of model to use ('tflite' or 'onnx')
            threshold (float): Threshold for failure prediction
            window_size (int): Size of sliding window for smoothing predictions
        """
        self.model_dir = model_dir
        self.model_type = model_type
        self.threshold = threshold
        self.window_size = window_size
        self.running = False
        self.results_queue = queue.Queue(maxsize=1000)

        # Prediction history for each unit
        self.prediction_history = {}

        # Load model metadata
        self._load_model_metadata()

        # Initialize model
        self._initialize_model()

        # Set up logging
        self._setup_logging()

    def _load_model_metadata(self):
        """
        Load model metadata.
        """
        metadata_path = os.path.join(self.model_dir, 'model_metadata.npy')

        if os.path.exists(metadata_path):
            self.model_metadata = np.load(metadata_path, allow_pickle=True).item()
            logging.info(f"Loaded model metadata: {list(self.model_metadata.keys())}")
        else:
            # Default metadata
            self.model_metadata = {
                'lstm_tflite': {
                    'path': os.path.join(self.model_dir, 'lstm_model_optimized.tflite'),
                    'input_shape': (10, 21)  # Assuming 10 time steps, 21 sensors
                },
                'lstm_onnx': {
                    'path': os.path.join(self.model_dir, 'lstm_model_optimized.onnx'),
                    'input_shape': (10, 21)
                },
                'xgboost_onnx': {
                    'path': os.path.join(self.model_dir, 'xgboost_model_optimized.onnx'),
                    'input_shape': (21,)  # Assuming 21 sensors
                }
            }
            logging.warning("Model metadata not found. Using default values.")

    def _initialize_model(self):
        """
        Initialize the model based on the selected type.
        """
        try:
            if self.model_type == 'tflite':
                # Get model path
                if 'lstm_tflite' in self.model_metadata:
                    model_path = self.model_metadata['lstm_tflite']['path']
                else:
                    model_path = os.path.join(self.model_dir, 'lstm_model_optimized.tflite')

                # Load TFLite model
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()

                # Get input and output details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()

                logging.info(f"Loaded TFLite model: {model_path}")
                logging.info(f"Input details: {self.input_details}")
                logging.info(f"Output details: {self.output_details}")

            elif self.model_type == 'onnx':
                # Get model path
                if 'lstm_onnx' in self.model_metadata:
                    model_path = self.model_metadata['lstm_onnx']['path']
                else:
                    model_path = os.path.join(self.model_dir, 'lstm_model_optimized.onnx')

                # Load ONNX model
                self.session = ort.InferenceSession(model_path)

                # Get input and output names
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [output.name for output in self.session.get_outputs()]

                logging.info(f"Loaded ONNX model: {model_path}")
                logging.info(f"Input name: {self.input_name}")
                logging.info(f"Output names: {self.output_names}")

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise

    def _setup_logging(self):
        """
        Set up logging for the inference engine.
        """
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)

        # Configure logging
        log_path = os.path.join(log_dir, 'inference_engine.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def predict(self, data_point):
        """
        Make a prediction using the model.

        Args:
            data_point (dict): Data point containing sensor values

        Returns:
            dict: Prediction results
        """
        # Check if we have a valid sequence
        if data_point.get('sequence') is None:
            return None

        # Extract sequence and convert to numpy array
        sequence = np.array(data_point['sequence'])

        try:
            if self.model_type == 'tflite':
                # Prepare input data
                input_data = sequence.astype(np.float32)
                input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

                # Set input tensor
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

                # Run inference
                self.interpreter.invoke()

                # Get results
                rul_prediction = self.interpreter.get_tensor(self.output_details[0]['index'])

                # Process prediction
                if len(rul_prediction.shape) > 0 and rul_prediction.shape[0] > 0:
                    predicted_rul = float(rul_prediction[0][0])
                else:
                    predicted_rul = float(rul_prediction)

            elif self.model_type == 'onnx':
                # Prepare input data
                input_data = sequence.astype(np.float32)
                input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

                # Run inference
                outputs = self.session.run(None, {self.input_name: input_data})

                # Get results
                rul_prediction = outputs[0]

                # Process prediction
                if len(rul_prediction.shape) > 0 and rul_prediction.shape[0] > 0:
                    predicted_rul = float(rul_prediction[0][0] if len(rul_prediction.shape) > 1 else rul_prediction[0])
                else:
                    predicted_rul = float(rul_prediction)

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Store prediction in history
            unit_nr = data_point['unit_nr']
            if unit_nr not in self.prediction_history:
                self.prediction_history[unit_nr] = deque(maxlen=self.window_size)

            self.prediction_history[unit_nr].append(predicted_rul)

            # Smooth prediction using moving average
            smoothed_rul = sum(self.prediction_history[unit_nr]) / len(self.prediction_history[unit_nr])

            # Determine status based on threshold
            if smoothed_rul <= self.threshold:
                status = "ALERT: Maintenance Required"
            else:
                status = "Normal"

            # Create result dictionary
            result = {
                'unit_nr': unit_nr,
                'cycle': data_point['cycle'],
                'timestamp': data_point['timestamp'],
                'predicted_rul': predicted_rul,
                'smoothed_rul': smoothed_rul,
                'true_rul': data_point.get('true_rul'),
                'status': status,
                'threshold': self.threshold
            }

            # Log prediction
            logging.info(
                f"Unit {unit_nr} - Cycle {data_point['cycle']} - Predicted RUL: {predicted_rul:.2f} - Smoothed RUL: {smoothed_rul:.2f} - Status: {status}")

            return result

        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return None

    def start_inference(self, data_simulator):
        """
        Start the inference engine in a separate thread.

        Args:
            data_simulator (DataSimulator): Data simulator instance
        """
        if self.running:
            logging.warning("Inference engine is already running.")
            return

        self.running = True
        self.data_simulator = data_simulator
        self.inference_thread = threading.Thread(target=self._run_inference)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        logging.info("Inference engine started.")

    def stop_inference(self):
        """
        Stop the inference engine.
        """
        self.running = False
        if hasattr(self, 'inference_thread') and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        logging.info("Inference engine stopped.")

    def _run_inference(self):
        """
        Run inference on streaming data.
        """
        while self.running:
            # Get data point from simulator
            data_point = self.data_simulator.get_data_point(timeout=1.0)

            if data_point is not None:
                # Make prediction
                result = self.predict(data_point)

                if result is not None:
                    # Put result in queue
                    try:
                        self.results_queue.put(result, block=False)
                    except queue.Full:
                        # If queue is full, remove oldest item
                        try:
                            self.results_queue.get_nowait()
                            self.results_queue.put(result, block=False)
                        except:
                            pass

            # Short delay
            time.sleep(0.01)

    def get_latest_result(self, timeout=1.0):
        """
        Get the latest inference result from the queue.

        Args:
            timeout (float): Timeout in seconds

        Returns:
            dict: Inference result
        """
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class RealTimeMonitor:
    """
    Real-time monitoring dashboard for predictive maintenance.
    """

    def __init__(self, inference_engine, max_points=100, update_interval=1000):
        """
        Initialize the monitor.

        Args:
            inference_engine (InferenceEngine): Inference engine instance
            max_points (int): Maximum number of points to display
            update_interval (int): Update interval in milliseconds
        """
        self.inference_engine = inference_engine
        self.max_points = max_points
        self.update_interval = update_interval

        # Initialize data storage
        self.timestamps = []
        self.rul_predictions = []
        self.true_ruls = []
        self.unit_nrs = []
        self.statuses = []

        # Create figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Initialize plots
        self.line_predicted, = self.ax1.plot([], [], 'b-', label='Predicted RUL')
        self.line_true, = self.ax1.plot([], [], 'g--', label='True RUL')
        self.line_threshold, = self.ax1.plot([], [], 'r-', label='Threshold')

        # Set up axes
        self.ax1.set_title('Real-Time RUL Predictions')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('RUL')
        self.ax1.legend()
        self.ax1.grid(True)

        # Status display
        self.ax2.axis('off')
        self.status_text = self.ax2.text(0.5, 0.5, 'Waiting for data...',
                                         horizontalalignment='center',
                                         verticalalignment='center',
                                         fontsize=12)

        # Animation
        self.ani = None

    def init_plot(self):
        """
        Initialize the plot.

        Returns:
            tuple: Plot elements
        """
        self.line_predicted.set_data([], [])
        self.line_true.set_data([], [])
        self.line_threshold.set_data([], [])

        return self.line_predicted, self.line_true, self.line_threshold, self.status_text

    def update_plot(self, frame):
        """
        Update the plot with new data.

        Args:
            frame: Frame number (unused)

        Returns:
            tuple: Updated plot elements
        """
        # Get latest result
        result = self.inference_engine.get_latest_result(timeout=0.1)

        if result is not None:
            # Add new data
            timestamp = datetime.fromisoformat(result['timestamp'])
            self.timestamps.append(timestamp)
            self.rul_predictions.append(result['smoothed_rul'])
            self.true_ruls.append(result.get('true_rul', np.nan))
            self.unit_nrs.append(result['unit_nr'])
            self.statuses.append(result['status'])

            # Trim data if needed
            if len(self.timestamps) > self.max_points:
                self.timestamps = self.timestamps[-self.max_points:]
                self.rul_predictions = self.rul_predictions[-self.max_points:]
                self.true_ruls = self.true_ruls[-self.max_points:]
                self.unit_nrs = self.unit_nrs[-self.max_points:]
                self.statuses = self.statuses[-self.max_points:]

            # Update plots
            self.line_predicted.set_data(range(len(self.timestamps)), self.rul_predictions)

            # Update true RUL line if values are available
            true_ruls_not_nan = [r for r in self.true_ruls if not np.isnan(r)]
            if true_ruls_not_nan:
                true_indices = [i for i, r in enumerate(self.true_ruls) if not np.isnan(r)]
                true_values = [self.true_ruls[i] for i in true_indices]
                self.line_true.set_data(true_indices, true_values)

            # Update threshold line
            self.line_threshold.set_data(range(len(self.timestamps)),
                                         [result['threshold']] * len(self.timestamps))

            # Adjust axes
            self.ax1.relim()
            self.ax1.autoscale_view()

            # Update status text
            status_text = (
                f"Unit: {result['unit_nr']}\n"
                f"Cycle: {result['cycle']}\n"
                f"Predicted RUL: {result['smoothed_rul']:.2f}\n"
                f"Status: {result['status']}\n"
                f"Timestamp: {timestamp.strftime('%H:%M:%S')}"
            )

            # Color the status based on severity
            if "ALERT" in result['status']:
                status_color = 'red'
            else:
                status_color = 'green'

            self.status_text.set_text(status_text)
            self.status_text.set_color(status_color)

        return self.line_predicted, self.line_true, self.line_threshold, self.status_text

    def start(self):
        """
        Start the real-time monitor.
        """
        self.ani = FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot,
                                 interval=self.update_interval, blit=True)
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to run the real-time inference system.
    """
    # Set paths
    data_path = "./data/processed"
    model_dir = "./models/optimized"

    # Create data simulator
    simulator = DataSimulator(data_path=data_path, sequence_length=10, delay=0.5)

    # Create inference engine
    engine = InferenceEngine(model_dir=model_dir, model_type='tflite', threshold=30)

    # Start simulation
    simulator.start_simulation()

    # Start inference
    engine.start_inference(simulator)

    try:
        # Create and start monitor
        monitor = RealTimeMonitor(engine, max_points=100, update_interval=1000)
        monitor.start()

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        # Clean up
        simulator.stop_simulation()
        engine.stop_inference()
        print("System shut down.")


if __name__ == "__main__":
    main()