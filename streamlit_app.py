import streamlit as st
import numpy as np
import pandas as pd

# Optional sklearn import
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(page_title="Genomic Annotation Reproducibility", layout="wide")
st.title("🔬 Genetic Mutation Annotation Explorer")

# ----------------------------
# Sidebar: Controls
# ----------------------------
st.sidebar.header("Simulation / Upload Controls")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV of annotations (Mutations x Researchers). Do NOT upload private data.", 
    type=["csv"]
)

# Simulation controls (used if no file uploaded)
num_samples = st.sidebar.slider("Number of Mutations", 20, 500, 100)
raters = st.sidebar.slider("Number of Researchers", 2, 10, 5)
noise_scale = st.sidebar.slider("Annotation Noise Level", 0.01, 0.5, 0.1)

# Model selection
models = ["NumPy Linear Regression"]
if SKLEARN_AVAILABLE:
    models.append("Random Forest (scikit-learn)")
model_choice = st.sidebar.selectbox("Model", models)

# ----------------------------
# Data Preparation
# ----------------------------
if uploaded_file is not None:
    annotation_data = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(annotation_data.head())
else:
    np.random.seed(42)
    data = {"Mutation_ID": np.arange(1, num_samples + 1)}
    for i in range(1, raters + 1):
        data[f"Researcher_{i}"] = np.random.normal(0.5, noise_scale, num_samples)
    annotation_data = pd.DataFrame(data)
    st.subheader("📄 Simulated Data Preview")
    st.dataframe(annotation_data.head())

# Ensure numeric columns only for analysis
numeric_data = annotation_data.select_dtypes(include=[np.number])
if numeric_data.shape[1] < 2:
    st.error("Dataset must have at least 1 Mutation ID column and 1 annotator column.")
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
st.subheader("📊 Intraclass Correlation Coefficient (ICC)")
st.metric("ICC Score", f"{icc_value:.4f}")

# ----------------------------
# Annotator Mean Chart
# ----------------------------
mean_scores = numeric_data.iloc[:, 1:].mean()
st.subheader("📈 Average Annotation Scores by Researcher")
st.bar_chart(mean_scores)

# ----------------------------
# NumPy Linear Regression
# ----------------------------
def numpy_linear_regression(X_train, y_train, X_test):
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]
    beta = np.linalg.pinv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train
    return X_test_bias @ beta

# ----------------------------
# Model Training Across Annotators
# ----------------------------
st.subheader("🤖 Model Performance vs Number of Annotators")

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
st.dataframe(results_df)
