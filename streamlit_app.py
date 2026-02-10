import streamlit as st
import numpy as np
import pandas as pd
import psutil
import os

# Optional sklearn import
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ----------------------------
# Monitoring Utility
# ----------------------------
def display_performance_metrics():
    """Captures and displays real-time resource usage."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Resource Monitor")
    m1, m2 = st.sidebar.columns(2)
    m1.metric("CPU Load", f"{cpu_percent}%")
    m2.metric("RAM Usage", f"{mem_mb:.1f} MB")

# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(page_title="Genomic Annotation Reproducibility", layout="wide")
st.title("Biomedical Annotation Reliability Explorer")

st.markdown("""
### Instructions
1. **Configure Parameters**: Use the sidebar to adjust the number of mutations and researchers for the simulation.
2. **Review Consistency**: Analyze the **Intraclass Correlation Coefficient (ICC)** to determine inter-rater reliability.
3. **Evaluate Impact**: Observe the chart at the bottom to see how increasing the number of annotators reduces model error.
""")

# ----------------------------
# Sidebar: Controls
# ----------------------------
st.sidebar.header("Control Panel")

# File upload with tooltip
uploaded_file = st.sidebar.file_uploader(
    "Upload Annotation CSV", 
    type=["csv"],
    help="Upload a CSV where rows are mutations and columns are researcher scores. Ensure data is anonymized."
)

# Simulation controls
num_samples = st.sidebar.slider("Number of Mutations", 20, 500, 100, help="Total biological samples to be annotated.")
raters = st.sidebar.slider("Number of Researchers", 2, 10, 5, help="Number of independent annotators providing scores.")
noise_scale = st.sidebar.slider("Annotation Noise Level", 0.01, 0.5, 0.1, help="Simulates subjectivity or inconsistency among raters.")

# Model selection
models = ["NumPy Linear Regression"]
if SKLEARN_AVAILABLE:
    models.append("Random Forest (scikit-learn)")
model_choice = st.sidebar.selectbox("Predictive Model", models, help="Select the algorithm used to evaluate the consensus labels.")

display_performance_metrics()

# ----------------------------
# Data Preparation
# ----------------------------
if uploaded_file is not None:
    annotation_data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(annotation_data.head())
else:
    np.random.seed(42)
    data = {"Mutation_ID": np.arange(1, num_samples + 1)}
    for i in range(1, raters + 1):
        data[f"Researcher_{i}"] = np.random.normal(0.5, noise_scale, num_samples)
    annotation_data = pd.DataFrame(data)
    st.subheader("Simulated Dataset Preview")
    st.info("The table below shows simulated researcher scores for each mutation.")
    st.dataframe(annotation_data.head())

numeric_data = annotation_data.select_dtypes(include=[np.number])
if numeric_data.shape[1] < 2:
    st.error("Dataset must have at least one ID column and one annotator column.")
    st.stop()

# ----------------------------
# ICC Calculation
# ----------------------------
def icc(data):
    mean_annotations = data.mean(axis=1)
    rater_var = data.var(axis=1).mean()
    total_var = mean_annotations.var()
    return (total_var - rater_var) / total_var if total_var > 0 else 0

icc_value = icc(numeric_data.iloc[:, 1:])
st.subheader("Inter-Rater Reliability Analysis")
st.metric(
    label="Intraclass Correlation Coefficient (ICC)", 
    value=f"{icc_value:.4f}",
    help="ICC measures the reliability of ratings or measurements. Values closer to 1.0 indicate high agreement."
)

# ----------------------------
# Annotator Mean Chart
# ----------------------------
mean_scores = numeric_data.iloc[:, 1:].mean()
st.subheader("Average Scores per Researcher")
st.bar_chart(mean_scores)

# ----------------------------
# Predictive Modeling logic
# ----------------------------
def numpy_linear_regression(X_train, y_train, X_test):
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]
    beta = np.linalg.pinv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train
    return X_test_bias @ beta

st.subheader("Predictive Performance vs. Annotation Consensus")
st.write("This analysis demonstrates how increasing the number of independent annotators impacts the Mean Squared Error (MSE) of the model.")

errors = []
num_raters_list = range(1, numeric_data.shape[1])

for num in num_raters_list:
    X = numeric_data.iloc[:, 1:num+1].values
    y = X.mean(axis=1)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_choice == "NumPy Linear Regression":
        preds = numpy_linear_regression(X_train, y_train, X_test)
        mse = np.mean((y_test - preds) ** 2)
    elif model_choice == "Random Forest (scikit-learn)":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
    errors.append(mse)

results_df = pd.DataFrame({"Annotators": list(num_raters_list), "MSE": errors})
st.line_chart(results_df, x="Annotators", y="MSE")

st.markdown("""
---
### Scientific Context
The **Intraclass Correlation Coefficient (ICC)** is a descriptive statistic that can be used when quantitative measurements are made on units that are organized into groups. It describes how strongly units in the same group resemble each other. In this app, it quantifies the degree to which different researchers agree on the mutation scores. 


""")
