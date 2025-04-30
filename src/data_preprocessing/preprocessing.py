import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class NASATurbofanPreprocessor:
    """
    Preprocessor for the NASA Turbofan Engine Degradation Simulation dataset.
    """

    def __init__(self, data_path, window_size=30, test_size=0.2, random_state=42):
        """
        Initialize the preprocessor.

        Args:
            data_path (str): Path to the dataset directory
            window_size (int): Size of the time window for feature engineering
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.window_size = window_size
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler()

    def load_data(self, filename="train_FD001.txt"):
        """
        Load the NASA Turbofan dataset.

        Args:
            filename (str): Name of the dataset file

        Returns:
            pd.DataFrame: Loaded dataset
        """
        # Column names for the dataset
        columns = ['unit_nr', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                  [f'sensor_{i}' for i in range(1, 22)]

        file_path = os.path.join(self.data_path, filename)
        data = pd.read_csv(file_path, sep=' ', header=None, names=columns)

        # Drop columns with NaN values (the dataset has extra spaces causing empty columns)
        data = data.dropna(axis=1, how='all')

        print(f"Loaded dataset with shape: {data.shape}")
        return data

    def calculate_rul(self, data):
        """
        Calculate Remaining Useful Life (RUL) for each cycle.

        Args:
            data (pd.DataFrame): Input dataset

        Returns:
            pd.DataFrame: Dataset with RUL column added
        """
        # Group by unit_nr and calculate max cycle for each unit
        max_cycles = data.groupby('unit_nr')['cycle'].max().reset_index()
        max_cycles.columns = ['unit_nr', 'max_cycle']

        # Merge with the original data
        data_with_max = data.merge(max_cycles, on='unit_nr', how='left')

        # Calculate RUL
        data_with_max['RUL'] = data_with_max['max_cycle'] - data_with_max['cycle']

        # Drop the max_cycle column
        data = data_with_max.drop('max_cycle', axis=1)

        return data

    def drop_unnecessary_columns(self, data):
        """
        Drop columns that are not useful for analysis.

        Args:
            data (pd.DataFrame): Input dataset

        Returns:
            pd.DataFrame: Dataset with unnecessary columns dropped
        """
        # Identify constant columns (zero variance)
        constant_cols = []
        for col in data.columns:
            if data[col].nunique() == 1:
                constant_cols.append(col)

        print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")

        # Drop constant columns
        data = data.drop(constant_cols, axis=1)

        return data

    def engineer_features(self, data):
        """
        Engineer new features based on rolling statistics.

        Args:
            data (pd.DataFrame): Input dataset

        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        # Create a copy to avoid modifying the original dataframe
        result = data.copy()

        # Group by unit_nr to ensure rolling calculations only use data from the same unit
        groups = result.groupby('unit_nr')

        # Initialize list for all engineered data
        engineered_dfs = []

        for unit_nr, group_data in groups:
            unit_data = group_data.copy().sort_values('cycle')

            # Get sensor columns
            sensor_cols = [col for col in unit_data.columns if 'sensor' in col]

            # Calculate rolling statistics
            for col in sensor_cols:
                # Rolling mean
                unit_data[f'{col}_rolling_mean'] = unit_data[col].rolling(
                    window=self.window_size, min_periods=1).mean()

                # Rolling standard deviation
                unit_data[f'{col}_rolling_std'] = unit_data[col].rolling(
                    window=self.window_size, min_periods=1).std().fillna(0)

                # Rolling trend (slope)
                unit_data[f'{col}_rolling_trend'] = unit_data[col].diff(
                    periods=5).rolling(window=self.window_size, min_periods=1).mean().fillna(0)

            engineered_dfs.append(unit_data)

        # Combine all engineered data
        result = pd.concat(engineered_dfs, ignore_index=True)

        print(f"Data shape after feature engineering: {result.shape}")
        return result

    def normalize_data(self, data, exclude_cols=None):
        """
        Normalize the features using Min-Max scaling.

        Args:
            data (pd.DataFrame): Input dataset
            exclude_cols (list): Columns to exclude from normalization

        Returns:
            pd.DataFrame: Normalized dataset
        """
        if exclude_cols is None:
            exclude_cols = ['unit_nr', 'cycle', 'RUL']

        # Identify columns to normalize
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        # Fit scaler on the feature columns
        self.scaler.fit(data[feature_cols])

        # Transform the feature columns
        normalized_features = self.scaler.transform(data[feature_cols])

        # Create a new dataframe with normalized features
        normalized_data = pd.DataFrame(normalized_features, columns=feature_cols)

        # Add back the excluded columns
        for col in exclude_cols:
            if col in data.columns:
                normalized_data[col] = data[col].values

        return normalized_data

    def create_sequences(self, data, sequence_length=10):
        """
        Create sequences for time series models.

        Args:
            data (pd.DataFrame): Input dataset
            sequence_length (int): Length of sequences

        Returns:
            tuple: (X_sequences, y_targets)
        """
        X = []
        y = []

        # Group by unit
        groups = data.groupby('unit_nr')

        # Get the feature columns (excluding unit_nr, cycle, and RUL)
        feature_cols = [col for col in data.columns
                        if col not in ['unit_nr', 'cycle', 'RUL']]

        for _, group in groups:
            # Sort by cycle
            group = group.sort_values('cycle')

            # Extract feature and target arrays
            feature_array = group[feature_cols].values
            target_array = group['RUL'].values

            # Create sequences
            for i in range(len(group) - sequence_length):
                X.append(feature_array[i:i + sequence_length])
                y.append(target_array[i + sequence_length])

        return np.array(X), np.array(y)

    def prepare_for_training(self, train_data, val_test_data=None, sequence_length=10):
        """
        Prepare data for model training and evaluation.

        Args:
            train_data (pd.DataFrame): Training dataset
            val_test_data (pd.DataFrame): Validation/test dataset (optional)
            sequence_length (int): Length of sequences for time series models

        Returns:
            dict: Dictionary containing prepared datasets
        """
        result = {}

        # Process training data
        print("Processing training data...")
        data_train = self.calculate_rul(train_data)
        data_train = self.drop_unnecessary_columns(data_train)
        data_train = self.engineer_features(data_train)
        data_train_normalized = self.normalize_data(data_train)

        # Split training data into train and validation sets
        if val_test_data is None:
            unit_nrs = data_train_normalized['unit_nr'].unique()
            train_units, val_units = train_test_split(
                unit_nrs, test_size=self.test_size, random_state=self.random_state)

            train_set = data_train_normalized[data_train_normalized['unit_nr'].isin(train_units)]
            val_set = data_train_normalized[data_train_normalized['unit_nr'].isin(val_units)]

            # Store tabular data (for traditional ML models)
            result['X_train_tabular'] = train_set.drop(['unit_nr', 'cycle', 'RUL'], axis=1)
            result['y_train_tabular'] = train_set['RUL']
            result['X_val_tabular'] = val_set.drop(['unit_nr', 'cycle', 'RUL'], axis=1)
            result['y_val_tabular'] = val_set['RUL']

            # Create sequences (for deep learning models)
            X_train_seq, y_train_seq = self.create_sequences(
                train_set, sequence_length=sequence_length)
            X_val_seq, y_val_seq = self.create_sequences(
                val_set, sequence_length=sequence_length)

            result['X_train_sequence'] = X_train_seq
            result['y_train_sequence'] = y_train_seq
            result['X_val_sequence'] = X_val_seq
            result['y_val_sequence'] = y_val_seq

        # Process separate test data if provided
        if val_test_data is not None:
            print("Processing test data...")
            data_test = self.calculate_rul(val_test_data)
            data_test = self.drop_unnecessary_columns(data_test)
            data_test = self.engineer_features(data_test)
            data_test_normalized = self.normalize_data(data_test)

            result['X_test_tabular'] = data_test_normalized.drop(['unit_nr', 'cycle', 'RUL'], axis=1)
            result['y_test_tabular'] = data_test_normalized['RUL']

            # Create sequences for test set
            X_test_seq, y_test_seq = self.create_sequences(
                data_test_normalized, sequence_length=sequence_length)

            result['X_test_sequence'] = X_test_seq
            result['y_test_sequence'] = y_test_seq

        return result

    def visualize_data(self, data):
        """
        Visualize the data.

        Args:
            data (pd.DataFrame): Dataset to visualize
        """
        # Create a directory for visualizations if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)

        # Select a random unit to visualize
        unit_nr = np.random.choice(data['unit_nr'].unique())
        unit_data = data[data['unit_nr'] == unit_nr].sort_values('cycle')

        # Get sensor columns
        sensor_cols = [col for col in unit_data.columns if 'sensor' in col and
                       not ('rolling' in col or 'trend' in col)]

        # Plot sensor readings over cycles
        plt.figure(figsize=(15, 10))
        for i, sensor in enumerate(sensor_cols[:6]):  # Plot first 6 sensors for clarity
            plt.subplot(3, 2, i + 1)
            plt.plot(unit_data['cycle'], unit_data[sensor])
            plt.title(f'{sensor} over time for unit {unit_nr}')
            plt.xlabel('Cycle')
            plt.ylabel('Value')

        plt.tight_layout()
        plt.savefig(f'visualizations/sensor_readings_unit_{unit_nr}.png')

        # Plot RUL degradation
        plt.figure(figsize=(10, 6))
        plt.plot(unit_data['cycle'], unit_data['RUL'])
        plt.title(f'RUL degradation for unit {unit_nr}')
        plt.xlabel('Cycle')
        plt.ylabel('RUL')
        plt.grid(True)
        plt.savefig(f'visualizations/rul_degradation_unit_{unit_nr}.png')

        # Plot correlation heatmap
        plt.figure(figsize=(16, 14))
        sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png')

        # Plot feature importance based on correlation with RUL
        if 'RUL' in data.columns:
            corr_with_rul = data.corr()['RUL'].sort_values()
            plt.figure(figsize=(10, 12))
            corr_with_rul.drop('RUL').plot(kind='barh')
            plt.title('Feature Correlation with RUL')
            plt.tight_layout()
            plt.savefig('visualizations/feature_correlation_with_rul.png')

        plt.close('all')


def main():
    """
    Main function to demonstrate usage of the preprocessor.
    """
    # Set paths and parameters
    data_path = "./data/raw"
    output_path = "./data/processed"
    os.makedirs(output_path, exist_ok=True)

    # Initialize preprocessor
    preprocessor = NASATurbofanPreprocessor(
        data_path=data_path,
        window_size=30,
        test_size=0.2,
        random_state=42
    )

    # Load data
    train_data = preprocessor.load_data(filename="train_FD001.txt")
    test_data = preprocessor.load_data(filename="test_FD001.txt")

    # Visualize raw data
    print("Visualizing raw data...")
    train_data_with_rul = preprocessor.calculate_rul(train_data)
    preprocessor.visualize_data(train_data_with_rul)

    # Prepare data for training
    print("Preparing data for training...")
    prepared_data = preprocessor.prepare_for_training(
        train_data=train_data,
        val_test_data=test_data,
        sequence_length=10
    )

    # Save prepared data
    print("Saving prepared data...")
    for name, data_obj in prepared_data.items():
        np.save(os.path.join(output_path, f"{name}.npy"), data_obj)

    print("Data preprocessing completed!")


if __name__ == "__main__":
    main()