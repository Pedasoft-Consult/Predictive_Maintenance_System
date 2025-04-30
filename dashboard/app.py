import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
import json
from datetime import datetime, timedelta
import threading
import queue
import logging
from collections import deque
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom modules
from src.real_time_inference.real_time_inference import DataSimulator, InferenceEngine
from src.monitoring_retraining.monitoring_retraining import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('dashboard')

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.simulator = None
    st.session_state.engine = None
    st.session_state.monitor = None
    st.session_state.results_queue = queue.Queue(maxsize=1000)
    st.session_state.history = {
        'timestamps': [],
        'unit_nrs': [],
        'cycles': [],
        'predicted_ruls': [],
        'true_ruls': [],
        'statuses': []
    }
    st.session_state.running = False
    st.session_state.thread = None
    st.session_state.update_interval = 0.5  # seconds


def init_system(data_path, model_dir, model_type, threshold, delay):
    """
    Initialize the predictive maintenance system.

    Args:
        data_path (str): Path to the data directory
        model_dir (str): Path to the model directory
        model_type (str): Type of model to use ('tflite' or 'onnx')
        threshold (float): Threshold for failure prediction
        delay (float): Delay between simulated data points
    """
    try:
        # Initialize simulator
        st.session_state.simulator = DataSimulator(
            data_path=data_path,
            sequence_length=10,
            delay=delay
        )

        # Initialize inference engine
        st.session_state.engine = InferenceEngine(
            model_dir=model_dir,
            model_type=model_type,
            threshold=threshold,
            window_size=10
        )

        # Initialize performance monitor
        st.session_state.monitor = PerformanceMonitor(
            log_dir='./logs',
            metrics_file='performance_metrics.csv',
            drift_threshold=0.2,
            evaluation_window=100
        )

        st.session_state.initialized = True
        logger.info("System initialized successfully.")

        return True

    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return False


def update_data():
    """
    Update data from the inference engine and update the history.
    """
    while st.session_state.running:
        try:
            # Get latest result from engine
            result = st.session_state.engine.get_latest_result(timeout=0.2)

            if result is not None:
                # Put in results queue for the dashboard
                try:
                    st.session_state.results_queue.put(result, block=False)
                except queue.Full:
                    # If queue is full, remove oldest item
                    try:
                        st.session_state.results_queue.get_nowait()
                        st.session_state.results_queue.put(result, block=False)
                    except:
                        pass

                # Add to performance monitor if true RUL is available
                if result.get('true_rul') is not None:
                    st.session_state.monitor.add_observation(
                        result['predicted_rul'],
                        result['true_rul'],
                        result['timestamp']
                    )

            # Short delay to prevent high CPU usage
            time.sleep(st.session_state.update_interval)

        except Exception as e:
            logger.error(f"Error in update thread: {e}")
            time.sleep(1)


def start_system():
    """
    Start the predictive maintenance system.
    """
    if not st.session_state.initialized:
        st.error("System not initialized. Please initialize first.")
        return False

    if st.session_state.running:
        st.warning("System is already running.")
        return True

    try:
        # Start data simulation
        st.session_state.simulator.start_simulation()

        # Start inference engine
        st.session_state.engine.start_inference(st.session_state.simulator)

        # Start update thread
        st.session_state.running = True
        st.session_state.thread = threading.Thread(target=update_data)
        st.session_state.thread.daemon = True
        st.session_state.thread.start()

        logger.info("System started successfully.")
        return True

    except Exception as e:
        logger.error(f"Error starting system: {e}")
        return False


def stop_system():
    """
    Stop the predictive maintenance system.
    """
    if not st.session_state.running:
        st.warning("System is not running.")
        return

    try:
        # Stop update thread
        st.session_state.running = False
        if st.session_state.thread is not None:
            st.session_state.thread.join(timeout=1.0)

        # Stop inference engine
        if st.session_state.engine is not None:
            st.session_state.engine.stop_inference()

        # Stop data simulator
        if st.session_state.simulator is not None:
            st.session_state.simulator.stop_simulation()

        logger.info("System stopped successfully.")

    except Exception as e:
        logger.error(f"Error stopping system: {e}")


def process_results():
    """
    Process results from the queue and update dashboard data.
    """
    # Maximum number of data points to keep
    max_points = 100

    # Get all available results from the queue
    while not st.session_state.results_queue.empty():
        try:
            result = st.session_state.results_queue.get_nowait()

            # Extract data
            timestamp = datetime.fromisoformat(result['timestamp'])
            unit_nr = result['unit_nr']
            cycle = result['cycle']
            predicted_rul = result['smoothed_rul']
            true_rul = result.get('true_rul')
            status = result['status']

            # Add to history
            st.session_state.history['timestamps'].append(timestamp)
            st.session_state.history['unit_nrs'].append(unit_nr)
            st.session_state.history['cycles'].append(cycle)
            st.session_state.history['predicted_ruls'].append(predicted_rul)
            st.session_state.history['true_ruls'].append(true_rul)
            st.session_state.history['statuses'].append(status)

            # Trim history if needed
            if len(st.session_state.history['timestamps']) > max_points:
                for key in st.session_state.history:
                    st.session_state.history[key] = st.session_state.history[key][-max_points:]

        except queue.Empty:
            break


def create_rul_chart():
    """
    Create a RUL chart with Plotly.

    Returns:
        go.Figure: Plotly figure
    """
    # Get data from history
    timestamps = st.session_state.history['timestamps']
    predicted_ruls = st.session_state.history['predicted_ruls']
    true_ruls = st.session_state.history['true_ruls']
    statuses = st.session_state.history['statuses']

    # Create dataframe
    df = pd.DataFrame({
        'timestamp': timestamps,
        'predicted_rul': predicted_ruls,
        'true_rul': true_ruls,
        'status': statuses
    })

    # Create figure
    fig = go.Figure()

    # Add predicted RUL line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['predicted_rul'],
        mode='lines+markers',
        name='Predicted RUL',
        line=dict(color='blue', width=2),
        marker=dict(size=5)
    ))

    # Add true RUL line if available
    if not all(x is None for x in true_ruls):
        # Filter out None values
        df_true = df[df['true_rul'].notna()]
        fig.add_trace(go.Scatter(
            x=df_true['timestamp'],
            y=df_true['true_rul'],
            mode='lines+markers',
            name='True RUL',
            line=dict(color='green', width=2, dash='dash'),
            marker=dict(size=5)
        ))

    # Add threshold line
    if st.session_state.engine:
        threshold = st.session_state.engine.threshold
        fig.add_trace(go.Scatter(
            x=[min(timestamps), max(timestamps)] if timestamps else [datetime.now(), datetime.now()],
            y=[threshold, threshold],
            mode='lines',
            name='Failure Threshold',
            line=dict(color='red', width=2, dash='dash')
        ))

    # Add colored regions
    if timestamps:
        # Highlight alert regions
        alert_periods = []
        current_alert = None

        for i, status in enumerate(statuses):
            if "ALERT" in status and current_alert is None:
                current_alert = i
            elif "ALERT" not in status and current_alert is not None:
                alert_periods.append((current_alert, i - 1))
                current_alert = None

        # If still in alert at the end
        if current_alert is not None:
            alert_periods.append((current_alert, len(statuses) - 1))

        # Add alert regions as shapes
        for start_idx, end_idx in alert_periods:
            fig.add_shape(
                type="rect",
                x0=timestamps[start_idx],
                x1=timestamps[end_idx],
                y0=0,
                y1=max(predicted_ruls) * 1.1,
                fillcolor="rgba(255, 0, 0, 0.1)",
                line=dict(width=0),
                layer="below"
            )

    # Update layout
    fig.update_layout(
        title='Remaining Useful Life (RUL) Prediction',
        xaxis_title='Time',
        yaxis_title='RUL',
        hovermode='closest',
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def create_status_indicator(latest_result):
    """
    Create a status indicator widget.

    Args:
        latest_result: Latest prediction result
    """
    if latest_result is None:
        status = "No data"
        color = "gray"
    elif "ALERT" in latest_result['status']:
        status = latest_result['status']
        color = "red"
    else:
        status = latest_result['status']
        color = "green"

    st.markdown(
        f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; '
        f'color: white; text-align: center; font-weight: bold;">{status}</div>',
        unsafe_allow_html=True
    )


def create_unit_info(latest_result):
    """
    Create a unit information panel.

    Args:
        latest_result: Latest prediction result
    """
    if latest_result is None:
        st.info("Waiting for data...")
        return

    # Create columns for displaying info
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Unit Number", latest_result['unit_nr'])
        st.metric("Current Cycle", latest_result['cycle'])

    with col2:
        st.metric(
            "Predicted RUL",
            f"{latest_result['smoothed_rul']:.2f}",
            delta=f"{latest_result['predicted_rul'] - latest_result['smoothed_rul']:.2f}"
        )

        if latest_result.get('true_rul') is not None:
            st.metric(
                "True RUL (for simulation)",
                f"{latest_result['true_rul']:.2f}",
                delta=f"{latest_result['smoothed_rul'] - latest_result['true_rul']:.2f}"
            )


def create_performance_metrics():
    """
    Create a performance metrics panel.
    """
    if st.session_state.monitor is None:
        return

    # Check if we have metrics history
    metrics_file = os.path.join('./logs', 'performance_metrics.csv')
    if not os.path.exists(metrics_file):
        st.info("No performance metrics available yet.")
        return

    try:
        # Load metrics history
        metrics_df = pd.read_csv(metrics_file)
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])

        if len(metrics_df) == 0:
            st.info("No performance metrics available yet.")
            return

        # Get latest metrics
        latest_metrics = metrics_df.iloc[-1]

        # Create metrics display
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("RMSE", f"{latest_metrics['rmse']:.4f}")

        with col2:
            st.metric("MAE", f"{latest_metrics['mae']:.4f}")

        with col3:
            st.metric("R²", f"{latest_metrics['r2']:.4f}")

        # Create performance trend chart
        if len(metrics_df) > 1:
            fig = go.Figure()

            # Add RMSE line
            fig.add_trace(go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df['rmse'],
                mode='lines+markers',
                name='RMSE',
                line=dict(color='blue', width=2)
            ))

            # Add MAE line
            fig.add_trace(go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df['mae'],
                mode='lines+markers',
                name='MAE',
                line=dict(color='green', width=2)
            ))

            # Update layout
            fig.update_layout(
                title='Error Metrics Over Time',
                xaxis_title='Time',
                yaxis_title='Error',
                legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
                height=300,
                margin=dict(l=20, r=20, t=60, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading performance metrics: {e}")


def show_sensor_data(latest_result):
    """
    Show sensor data in a collapsible section.

    Args:
        latest_result: Latest prediction result
    """
    if latest_result is None:
        return

    with st.expander("Sensor Data Visualization"):
        # Get the sequence data from the simulator
        simulator = st.session_state.simulator
        if simulator and simulator.current_unit is not None:
            # Get data for the current unit
            data_point = simulator.get_data_point(timeout=0.1)

            if data_point and data_point.get('sequence') is not None:
                sequence = np.array(data_point['sequence'])

                # Transpose to get sensor values over time
                sequence_T = sequence.T

                # Create dataframe
                df = pd.DataFrame(sequence_T, columns=[f'T-{10 - i}' for i in range(10)])
                df['Sensor'] = [f'Sensor {i + 1}' for i in range(len(sequence_T))]

                # Melt for Plotly
                df_melted = pd.melt(
                    df,
                    id_vars=['Sensor'],
                    value_vars=[f'T-{10 - i}' for i in range(10)],
                    var_name='Timestep',
                    value_name='Value'
                )

                # Create heatmap
                fig = px.imshow(
                    sequence_T,
                    labels=dict(x="Timestep", y="Sensor", color="Value"),
                    x=[f'T-{10 - i}' for i in range(10)],
                    y=[f'Sensor {i + 1}' for i in range(len(sequence_T))],
                    title="Sensor Values Heatmap",
                    color_continuous_scale="Viridis"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Create line chart for a few selected sensors
                selected_sensors = [0, 2, 4, 6, 8]  # Select a few sensors for clarity
                fig2 = go.Figure()

                for i in selected_sensors:
                    fig2.add_trace(go.Scatter(
                        x=[f'T-{10 - j}' for j in range(10)],
                        y=sequence_T[i],
                        mode='lines+markers',
                        name=f'Sensor {i + 1}'
                    ))

                fig2.update_layout(
                    title='Selected Sensor Values Over Time',
                    xaxis_title='Timestep',
                    yaxis_title='Sensor Value',
                    legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
                    height=300,
                    margin=dict(l=20, r=20, t=60, b=20)
                )

                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Waiting for sequence data...")
        else:
            st.info("No sensor data available.")


def main():
    # Set page config
    st.set_page_config(
        page_title="Predictive Maintenance Dashboard",
        page_icon="🔧",
        layout="wide"
    )

    # Header
    st.title("Predictive Maintenance System")
    st.markdown("Real-time monitoring of machine health and failure prediction")

    # Sidebar controls
    with st.sidebar:
        st.header("System Controls")

        # System configuration
        st.subheader("Configuration")
        data_path = st.text_input("Data Path", value="./data/processed")
        model_dir = st.text_input("Model Directory", value="./models/optimized")
        model_type = st.selectbox("Model Type", options=["tflite", "onnx"], index=0)
        threshold = st.slider("Failure Threshold (RUL)", min_value=10, max_value=100, value=30)
        delay = st.slider("Simulation Delay (seconds)", min_value=0.1, max_value=2.0, value=0.5)

        # Initialize button
        if st.button("Initialize System"):
            success = init_system(data_path, model_dir, model_type, threshold, delay)
            if success:
                st.success("System initialized successfully!")
            else:
                st.error("Failed to initialize system. Check logs for details.")

        # Start/Stop buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start System", disabled=not st.session_state.initialized):
                success = start_system()
                if success:
                    st.success("System started!")
                else:
                    st.error("Failed to start system.")

        with col2:
            if st.button("Stop System", disabled=not st.session_state.running):
                stop_system()
                st.success("System stopped!")

        # Status indicator
        if st.session_state.running:
            st.markdown('<div style="background-color: green; padding: 10px; border-radius: 5px; '
                        'color: white; text-align: center; font-weight: bold;">RUNNING</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div style="background-color: gray; padding: 10px; border-radius: 5px; '
                        'color: white; text-align: center; font-weight: bold;">STOPPED</div>',
                        unsafe_allow_html=True)

    # Main dashboard
    # Process any new results
    process_results()

    # Get latest result
    latest_result = None
    if st.session_state.history['timestamps']:
        latest_idx = len(st.session_state.history['timestamps']) - 1
        latest_result = {
            'unit_nr': st.session_state.history['unit_nrs'][latest_idx],
            'cycle': st.session_state.history['cycles'][latest_idx],
            'timestamp': st.session_state.history['timestamps'][latest_idx].isoformat(),
            'predicted_rul': st.session_state.history['predicted_ruls'][latest_idx],
            'smoothed_rul': st.session_state.history['predicted_ruls'][latest_idx],
            'true_rul': st.session_state.history['true_ruls'][latest_idx],
            'status': st.session_state.history['statuses'][latest_idx],
            'threshold': threshold if st.session_state.initialized else 30
        }

    # Create dashboard sections
    # Main dashboard components (status, unit info, RUL chart)
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Status")
        create_status_indicator(latest_result)
        st.subheader("Unit Information")
        create_unit_info(latest_result)

    with col2:
        # RUL Chart
        rul_chart = create_rul_chart()
        st.plotly_chart(rul_chart, use_container_width=True)

    # Performance metrics
    st.subheader("Performance Metrics")
    create_performance_metrics()

    # Sensor data visualization
    st.subheader("Sensor Data")
    show_sensor_data(latest_result)

    # Refresh the dashboard
    if st.session_state.running:
        time.sleep(0.1)  # Small delay to prevent high CPU usage
        st.experimental_rerun()


if __name__ == "__main__":
    main()