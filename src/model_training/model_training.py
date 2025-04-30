import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class TraditionalMLModel:
    """
    Traditional Machine Learning model for predictive maintenance.
    Uses XGBoost for regression to predict Remaining Useful Life (RUL).
    """

    def __init__(self, model_dir='./models', model_name='xgboost_model.pkl', random_state=42):
        """
        Initialize the model.

        Args:
            model_dir (str): Directory to save the trained model
            model_name (str): Name for the saved model file
            random_state (int): Random seed for reproducibility
        """
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, model_name)
        self.random_state = random_state

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Initialize XGBoost model
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.

        Args:
            X_train (np.ndarray or pd.DataFrame): Training features
            y_train (np.ndarray or pd.Series): Training targets
            X_val (np.ndarray or pd.DataFrame): Validation features
            y_val (np.ndarray or pd.Series): Validation targets

        Returns:
            dict: Training history and metrics
        """
        print("Training XGBoost model...")
        start_time = time.time()

        # Train with evaluation set if provided
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='rmse',
                early_stopping_rounds=10,
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Save model
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

        # Get feature importance
        feature_importance = self.model.feature_importances_

        # Return training information
        return {
            'training_time': training_time,
            'feature_importance': feature_importance
        }

    def evaluate(self, X_test, y_test, threshold=None):
        """
        Evaluate the model on test data.

        Args:
            X_test (np.ndarray or pd.DataFrame): Test features
            y_test (np.ndarray or pd.Series): Test targets
            threshold (float, optional): Threshold for binary classification
                                        (e.g., predict failure if RUL < threshold)

        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating XGBoost model...")

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate regression metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        # If threshold is provided, calculate classification metrics
        if threshold is not None:
            y_test_binary = (y_test < threshold).astype(int)
            y_pred_binary = (y_pred < threshold).astype(int)

            precision = precision_score(y_test_binary, y_pred_binary)
            recall = recall_score(y_test_binary, y_pred_binary)
            f1 = f1_score(y_test_binary, y_pred_binary)

            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        print("Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        return metrics

    def predict(self, X):
        """
        Make predictions with the model.

        Args:
            X (np.ndarray or pd.DataFrame): Features to predict on

        Returns:
            np.ndarray: Predicted RUL values
        """
        return self.model.predict(X)

    def load(self):
        """
        Load a saved model.

        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"Model file not found at {self.model_path}")
            return False

    def visualize_feature_importance(self, feature_names=None):
        """
        Visualize feature importance.

        Args:
            feature_names (list): List of feature names (optional)
        """
        importance = self.model.feature_importances_

        # If feature names not provided, use indices
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance))]

        # Create a DataFrame for easier visualization
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        # Visualize
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()

        # Save the visualization
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/xgboost_feature_importance.png')
        plt.close()


class DeepLearningModel:
    """
    Deep Learning model for predictive maintenance.
    Uses LSTM for time series regression to predict Remaining Useful Life (RUL).
    """

    def __init__(self, input_shape, model_dir='./models', model_name='lstm_model.h5', random_state=42):
        """
        Initialize the model.

        Args:
            input_shape (tuple): Shape of input data (sequence_length, n_features)
            model_dir (str): Directory to save the trained model
            model_name (str): Name for the saved model file
            random_state (int): Random seed for reproducibility
        """
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, model_name)
        self.random_state = random_state
        self.input_shape = input_shape

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Set random seed for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        # Initialize LSTM model
        self.model = self._build_model()

    def _build_model(self):
        """
        Build and compile the LSTM model.

        Returns:
            tf.keras.models.Sequential: Compiled LSTM model
        """
        model = Sequential([
            # First LSTM layer
            LSTM(128, input_shape=self.input_shape, return_sequences=True),
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

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=50, batch_size=32, verbose=1):
        """
        Train the model.

        Args:
            X_train (np.ndarray): Training features (shape: [n_samples, seq_len, n_features])
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level

        Returns:
            dict: Training history
        """
        print("Training LSTM model...")
        start_time = time.time()

        # Prepare callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        # Validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Save model summary to file
        os.makedirs('logs', exist_ok=True)
        with open('logs/lstm_model_summary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Visualize training
        self.visualize_training_history(history)

        return {
            'history': history.history,
            'training_time': training_time
        }

    def evaluate(self, X_test, y_test, threshold=None):
        """
        Evaluate the model on test data.

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            threshold (float, optional): Threshold for binary classification
                                        (e.g., predict failure if RUL < threshold)

        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating LSTM model...")

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Reshape predictions if needed
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()

        # Calculate regression metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        # If threshold is provided, calculate classification metrics
        if threshold is not None:
            y_test_binary = (y_test < threshold).astype(int)
            y_pred_binary = (y_pred < threshold).astype(int)

            precision = precision_score(y_test_binary, y_pred_binary)
            recall = recall_score(y_test_binary, y_pred_binary)
            f1 = f1_score(y_test_binary, y_pred_binary)

            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        print("Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        return metrics

    def predict(self, X):
        """
        Make predictions with the model.

        Args:
            X (np.ndarray): Features to predict on

        Returns:
            np.ndarray: Predicted RUL values
        """
        predictions = self.model.predict(X)

        # Reshape predictions if needed
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()

        return predictions

    def load(self):
        """
        Load a saved model.

        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"Model file not found at {self.model_path}")
            return False

    def visualize_training_history(self, history):
        """
        Visualize training history.

        Args:
            history: Training history object returned by model.fit()
        """
        # Create directory for visualizations
        os.makedirs('visualizations', exist_ok=True)

        # Plot training & validation loss
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout()
        plt.savefig('visualizations/lstm_training_history.png')
        plt.close()

    def visualize_predictions(self, X_test, y_test, n_samples=5):
        """
        Visualize model predictions vs actual values.

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            n_samples (int): Number of samples to visualize
        """
        # Get predictions
        y_pred = self.predict(X_test)

        # Create directory for visualizations
        os.makedirs('visualizations', exist_ok=True)

        # Plot predictions vs actual
        plt.figure(figsize=(15, 10))

        # Random indices for visualization
        if n_samples < len(y_test):
            indices = np.random.choice(len(y_test), n_samples, replace=False)
        else:
            indices = np.arange(len(y_test))

        for i, idx in enumerate(indices):
            plt.subplot(n_samples, 1, i + 1)
            plt.plot([y_test[idx]], 'go-', label='Actual')
            plt.plot([y_pred[idx]], 'ro-', label='Predicted')
            plt.title(f'Sample {idx}: Actual={y_test[idx]:.2f}, Predicted={y_pred[idx]:.2f}')
            plt.legend()

        plt.tight_layout()
        plt.savefig('visualizations/lstm_predictions.png')
        plt.close()


def main():
    """
    Main function to demonstrate training both models.
    """
    # Load prepared data
    data_path = "./data/processed"

    # Load traditional ML data
    X_train_tabular = np.load(os.path.join(data_path, "X_train_tabular.npy"), allow_pickle=True)
    y_train_tabular = np.load(os.path.join(data_path, "y_train_tabular.npy"), allow_pickle=True)
    X_val_tabular = np.load(os.path.join(data_path, "X_val_tabular.npy"), allow_pickle=True)
    y_val_tabular = np.load(os.path.join(data_path, "y_val_tabular.npy"), allow_pickle=True)
    X_test_tabular = np.load(os.path.join(data_path, "X_test_tabular.npy"), allow_pickle=True)
    y_test_tabular = np.load(os.path.join(data_path, "y_test_tabular.npy"), allow_pickle=True)

    # Load sequence data for deep learning
    X_train_sequence = np.load(os.path.join(data_path, "X_train_sequence.npy"), allow_pickle=True)
    y_train_sequence = np.load(os.path.join(data_path, "y_train_sequence.npy"), allow_pickle=True)
    X_val_sequence = np.load(os.path.join(data_path, "X_val_sequence.npy"), allow_pickle=True)
    y_val_sequence = np.load(os.path.join(data_path, "y_val_sequence.npy"), allow_pickle=True)
    X_test_sequence = np.load(os.path.join(data_path, "X_test_sequence.npy"), allow_pickle=True)
    y_test_sequence = np.load(os.path.join(data_path, "y_test_sequence.npy"), allow_pickle=True)

    # Train traditional ML model
    print("\n=== Training Traditional ML Model (XGBoost) ===")
    trad_model = TraditionalMLModel(model_dir='./models', model_name='xgboost_model.pkl')
    trad_model.train(X_train_tabular, y_train_tabular, X_val_tabular, y_val_tabular)

    # Evaluate traditional ML model
    print("\n=== Evaluating Traditional ML Model ===")
    trad_metrics = trad_model.evaluate(X_test_tabular, y_test_tabular, threshold=30)

    # Visualize feature importance
    if isinstance(X_train_tabular, np.ndarray):
        feature_names = [f'Feature_{i}' for i in range(X_train_tabular.shape[1])]
    else:  # pandas DataFrame
        feature_names = X_train_tabular.columns.tolist()

    trad_model.visualize_feature_importance(feature_names)

    # Train deep learning model
    print("\n=== Training Deep Learning Model (LSTM) ===")
    input_shape = (X_train_sequence.shape[1], X_train_sequence.shape[2])
    dl_model = DeepLearningModel(input_shape, model_dir='./models', model_name='lstm_model.h5')
    dl_model.train(X_train_sequence, y_train_sequence, X_val_sequence, y_val_sequence, epochs=50)

    # Evaluate deep learning model
    print("\n=== Evaluating Deep Learning Model ===")
    dl_metrics = dl_model.evaluate(X_test_sequence, y_test_sequence, threshold=30)

    # Visualize predictions
    dl_model.visualize_predictions(X_test_sequence, y_test_sequence, n_samples=5)

    # Compare models
    print("\n=== Model Comparison ===")
    metrics_comparison = {
        'XGBoost': trad_metrics,
        'LSTM': dl_metrics
    }

    for model_name, metrics in metrics_comparison.items():
        print(f"\n{model_name} Model:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()