import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import onnx
import onnxruntime as ort
import tf2onnx


class ModelOptimizer:
    """
    Model optimizer for edge deployment.
    Converts models to TensorFlow Lite and ONNX formats for efficient edge inference.
    """

    def __init__(self, model_dir='./models', output_dir='./models/optimized'):
        """
        Initialize the optimizer.

        Args:
            model_dir (str): Directory containing the original models
            output_dir (str): Directory to save optimized models
        """
        self.model_dir = model_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def optimize_keras_model(self, model_path, quantize=True):
        """
        Optimize a Keras model for edge deployment.

        Args:
            model_path (str): Path to the original Keras model
            quantize (bool): Whether to apply quantization

        Returns:
            tuple: (tflite_path, onnx_path)
        """
        print(f"Optimizing Keras model: {model_path}")

        # Load the model
        model = load_model(model_path)

        # Get model name without extension
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        # Convert to TensorFlow Lite
        tflite_path = self._convert_to_tflite(model, model_name, quantize)

        # Convert to ONNX
        onnx_path = self._convert_keras_to_onnx(model, model_name)

        return tflite_path, onnx_path

    def optimize_sklearn_model(self, model_path):
        """
        Optimize a scikit-learn model for edge deployment.

        Args:
            model_path (str): Path to the original scikit-learn model

        Returns:
            str: Path to optimized ONNX model
        """
        print(f"Optimizing scikit-learn model: {model_path}")

        # Load the model
        model = joblib.load(model_path)

        # Get model name without extension
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        # Convert to ONNX
        onnx_path = self._convert_sklearn_to_onnx(model, model_name)

        return onnx_path

    def _convert_to_tflite(self, model, model_name, quantize=True):
        """
        Convert a Keras model to TensorFlow Lite.

        Args:
            model: TensorFlow model
            model_name (str): Name for the saved model
            quantize (bool): Whether to apply quantization

        Returns:
            str: Path to the saved TFLite model
        """
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Apply optimizations
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        # Convert the model
        tflite_model = converter.convert()

        # Save the model
        tflite_path = os.path.join(self.output_dir, f"{model_name}_optimized.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"TFLite model saved to: {tflite_path}")

        # Calculate model size
        model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
        print(f"TFLite model size: {model_size:.2f} MB")

        return tflite_path

    def _convert_keras_to_onnx(self, model, model_name):
        """
        Convert a Keras model to ONNX.

        Args:
            model: TensorFlow model
            model_name (str): Name for the saved model

        Returns:
            str: Path to the saved ONNX model
        """
        # Prepare ONNX file path
        onnx_path = os.path.join(self.output_dir, f"{model_name}_optimized.onnx")

        # Convert the model using tf2onnx
        input_signature = [tf.TensorSpec([None] + list(model.input.shape[1:]), tf.float32)]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

        # Save the model
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"ONNX model saved to: {onnx_path}")

        # Calculate model size
        model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        print(f"ONNX model size: {model_size:.2f} MB")

        return onnx_path

    def _convert_sklearn_to_onnx(self, model, model_name):
        """
        Convert a scikit-learn model to ONNX.

        Args:
            model: scikit-learn model
            model_name (str): Name for the saved model

        Returns:
            str: Path to the saved ONNX model
        """
        # We will use sklearn-onnx for conversion
        # Since we can't directly import it here, we'll implement a simplified approach

        # Get model type and determine appropriate conversion method
        model_type = type(model).__name__

        if 'XGBRegressor' in model_type:
            # For XGBoost, we need to use a specific approach
            try:
                import xgboost as xgb
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType

                # Define input type
                input_type = [('input', FloatTensorType([None, model.n_features_in_]))]

                # Convert to ONNX
                onnx_model = convert_sklearn(model, initial_types=input_type)

                # Save the model
                onnx_path = os.path.join(self.output_dir, f"{model_name}_optimized.onnx")
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())

                print(f"ONNX model saved to: {onnx_path}")

                # Calculate model size
                model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                print(f"ONNX model size: {model_size:.2f} MB")

                return onnx_path
            except ImportError:
                print("skl2onnx package is required for XGBoost conversion. Please install it.")
                return None
        else:
            print(f"Unsupported model type: {model_type}")
            return None

    def benchmark_inference(self, model_path, input_data, num_runs=100):
        """
        Benchmark inference speed for an optimized model.

        Args:
            model_path (str): Path to the optimized model
            input_data (np.ndarray): Input data for inference
            num_runs (int): Number of inference runs for benchmarking

        Returns:
            dict: Benchmark results
        """
        print(f"Benchmarking inference for: {model_path}")

        # Check model type
        if model_path.endswith('.tflite'):
            # TFLite benchmark
            return self._benchmark_tflite(model_path, input_data, num_runs)
        elif model_path.endswith('.onnx'):
            # ONNX benchmark
            return self._benchmark_onnx(model_path, input_data, num_runs)
        else:
            print(f"Unsupported model format: {model_path}")
            return None

    def _benchmark_tflite(self, model_path, input_data, num_runs=100):
        """
        Benchmark TFLite model inference.

        Args:
            model_path (str): Path to the TFLite model
            input_data (np.ndarray): Input data for inference
            num_runs (int): Number of inference runs

        Returns:
            dict: Benchmark results
        """
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare input data
        if len(input_data.shape) == 3:  # For LSTM
            input_data_shape = (1,) + input_data.shape[1:]
            input_data_reshaped = input_data[0:1]  # Get first sample
        else:  # For other models
            input_data_shape = (1,) + input_data.shape[1:]
            input_data_reshaped = input_data[0:1]  # Get first sample

        # Ensure input data type matches model expectations
        input_dtype = input_details[0]['dtype']
        if input_dtype == np.float32:
            input_data_reshaped = input_data_reshaped.astype(np.float32)

        # Run warm-up inference
        interpreter.set_tensor(input_details[0]['index'], input_data_reshaped)
        interpreter.invoke()

        # Run timed inference
        inference_times = []
        for _ in range(num_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data_reshaped)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            inference_times.append(time.time() - start_time)

        # Calculate statistics
        avg_time = np.mean(inference_times) * 1000  # ms
        std_time = np.std(inference_times) * 1000  # ms

        print(f"TFLite inference: {avg_time:.2f} ± {std_time:.2f} ms")

        return {
            'model_type': 'tflite',
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }

    def _benchmark_onnx(self, model_path, input_data, num_runs=100):
        """
        Benchmark ONNX model inference.

        Args:
            model_path (str): Path to the ONNX model
            input_data (np.ndarray): Input data for inference
            num_runs (int): Number of inference runs

        Returns:
            dict: Benchmark results
        """
        # Load ONNX model
        session = ort.InferenceSession(model_path)

        # Get input name
        input_name = session.get_inputs()[0].name

        # Prepare input data
        if len(input_data.shape) == 3:  # For LSTM
            input_data_reshaped = input_data[0:1]  # Get first sample
        else:  # For other models
            input_data_reshaped = input_data[0:1]  # Get first sample

        # Ensure input data type is float32
        input_data_reshaped = input_data_reshaped.astype(np.float32)

        # Run warm-up inference
        session.run(None, {input_name: input_data_reshaped})

        # Run timed inference
        inference_times = []
        for _ in range(num_runs):
            start_time = time.time()
            outputs = session.run(None, {input_name: input_data_reshaped})
            inference_times.append(time.time() - start_time)

        # Calculate statistics
        avg_time = np.mean(inference_times) * 1000  # ms
        std_time = np.std(inference_times) * 1000  # ms

        print(f"ONNX inference: {avg_time:.2f} ± {std_time:.2f} ms")

        return {
            'model_type': 'onnx',
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }

    def visualize_benchmark_results(self, results):
        """
        Visualize benchmark results.

        Args:
            results (list): List of benchmark result dictionaries
        """
        # Create directory for visualizations
        os.makedirs('visualizations', exist_ok=True)

        # Extract data
        model_names = [r.get('model_name', f"Model {i}") for i, r in enumerate(results)]
        model_types = [r['model_type'] for r in results]
        inference_times = [r['avg_inference_time_ms'] for r in results]
        model_sizes = [r['model_size_mb'] for r in results]

        # Create figures
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot inference times
        bar_colors = ['blue' if t == 'tflite' else 'green' for t in model_types]
        ax1.bar(model_names, inference_times, color=bar_colors)
        ax1.set_ylabel('Inference Time (ms)')
        ax1.set_title('Model Inference Time Comparison')
        ax1.set_xticklabels(model_names, rotation=45, ha='right')

        # Plot model sizes
        ax2.bar(model_names, model_sizes, color=bar_colors)
        ax2.set_ylabel('Model Size (MB)')
        ax2.set_title('Model Size Comparison')
        ax2.set_xticklabels(model_names, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('visualizations/model_benchmark_comparison.png')
        plt.close()

        # Create a summary table
        summary_data = {
            'Model': model_names,
            'Type': model_types,
            'Inference Time (ms)': [f"{t:.2f}" for t in inference_times],
            'Model Size (MB)': [f"{s:.2f}" for s in model_sizes]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('visualizations/model_benchmark_summary.csv', index=False)

        print(f"Benchmark visualization saved to visualizations/model_benchmark_comparison.png")
        print(f"Benchmark summary saved to visualizations/model_benchmark_summary.csv")


def main():
    """
    Main function to optimize and benchmark models.
    """
    # Paths to original models
    keras_model_path = os.path.join('./models', 'lstm_model.h5')
    sklearn_model_path = os.path.join('./models', 'xgboost_model.pkl')

    # Load test data for benchmarking
    data_path = "./data/processed"
    X_test_tabular = np.load(os.path.join(data_path, "X_test_tabular.npy"), allow_pickle=True)
    X_test_sequence = np.load(os.path.join(data_path, "X_test_sequence.npy"), allow_pickle=True)

    # Initialize optimizer
    optimizer = ModelOptimizer(model_dir='./models', output_dir='./models/optimized')

    # Optimize models
    tflite_path, onnx_path_keras = optimizer.optimize_keras_model(keras_model_path, quantize=True)
    onnx_path_sklearn = optimizer.optimize_sklearn_model(sklearn_model_path)

    # Benchmark models
    benchmark_results = []

    # Benchmark LSTM TFLite
    tflite_result = optimizer.benchmark_inference(tflite_path, X_test_sequence)
    tflite_result['model_name'] = 'LSTM TFLite'
    benchmark_results.append(tflite_result)

    # Benchmark LSTM ONNX
    onnx_keras_result = optimizer.benchmark_inference(onnx_path_keras, X_test_sequence)
    onnx_keras_result['model_name'] = 'LSTM ONNX'
    benchmark_results.append(onnx_keras_result)

    # Benchmark XGBoost ONNX if available
    if onnx_path_sklearn:
        onnx_sklearn_result = optimizer.benchmark_inference(onnx_path_sklearn, X_test_tabular)
        onnx_sklearn_result['model_name'] = 'XGBoost ONNX'
        benchmark_results.append(onnx_sklearn_result)

    # Visualize benchmark results
    optimizer.visualize_benchmark_results(benchmark_results)

    # Export model metadata for real-time inference
    model_metadata = {
        'lstm_tflite': {
            'path': tflite_path,
            'input_shape': X_test_sequence.shape[1:]
        },
        'lstm_onnx': {
            'path': onnx_path_keras,
            'input_shape': X_test_sequence.shape[1:]
        }
    }

    if onnx_path_sklearn:
        model_metadata['xgboost_onnx'] = {
            'path': onnx_path_sklearn,
            'input_shape': X_test_tabular.shape[1:]
        }

    # Save metadata
    np.save(os.path.join('./models/optimized', 'model_metadata.npy'), model_metadata)

    print("Model optimization and benchmarking completed!")


if __name__ == "__main__":
    main()