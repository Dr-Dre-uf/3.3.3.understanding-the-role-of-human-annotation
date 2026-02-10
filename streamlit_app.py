import streamlit as st
import numpy as np
import pandas as pd
import psutil
import os
import matplotlib.pyplot as plt

# Optional sklearn import
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
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
st.set_page_config(page_title="Radiology AI Reliability", layout="wide")
st.title("🩻 Radiology AI Reliability: Lung Cancer Detection")

# --- PRIVACY WARNING ---
st.warning("""
**DATA PRIVACY NOTICE**: 
This application is for demonstration and research simulation purposes only. 
**Do not upload** any Personally Identifiable Information (PII), Protected Health Information (PHI), or any confidential patient data. 
Ensure all datasets are fully anonymized before uploading.
""")

st.markdown("""
### Case Study: Lung Cancer Detection
**Context:** Radiologists label chest X-rays as either **Cancerous (1)** or **Benign (0)**. Inconsistencies among radiologists can confuse AI models.

**Objectives:**
1.  **Quantify Agreement:** Use ICC to measure how consistently radiologists rate the images.
2.  **Identify Ambiguity:** Find specific X-rays with high disagreement for potential re-review.
3.  **Optimize Annotators:** Determine how many radiologists are needed to train a reliable AI model.
""")

# ----------------------------
# Sidebar: Controls
# ----------------------------
st.sidebar.header("Simulation Settings")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Radiologist Annotations (CSV)", 
    type=["csv"],
    help="Upload a CSV where rows are Patient X-Rays and columns are Radiologist Diagnoses (0 or 1). STRICTLY NO PII/PHI."
)

# Simulation controls
num_samples = st.sidebar.slider("Number of X-Rays", 50, 1000, 200, help="Total number of chest X-ray images in the dataset.")
raters = st.sidebar.slider("Number of Radiologists", 2, 10, 5, help="Number of radiologists providing diagnoses.")
disagreement_rate = st.sidebar.slider("Disagreement Rate", 0.0, 0.5, 0.2, help="Probability that a radiologist disagrees with the 'ground truth'.")

# Model selection
models = ["Random Forest Classifier"] if SKLEARN_AVAILABLE else ["(Scikit-learn not found)"]
model_choice = st.sidebar.selectbox("AI Model", models, help="The machine learning model used to predict the diagnosis.")

display_performance_metrics()

# ----------------------------
# Data Preparation
# ----------------------------
if uploaded_file is not None:
    annotation_data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(annotation_data.head())
else:
    # Simulate Ground Truth (hidden from raters)
    np.random.seed(42)
    ground_truth = np.random.randint(0, 2, num_samples)
    
    data = {"Xray_ID": np.arange(1, num_samples + 1)}
    for i in range(1, raters + 1):
        # Generate ratings with some noise (disagreement)
        ratings = ground_truth.copy()
        # Flip labels based on disagreement rate
        noise_mask = np.random.rand(num_samples) < disagreement_rate
        ratings[noise_mask] = 1 - ratings[noise_mask] 
        data[f"Radiologist_{i}"] = ratings
        
    annotation_data = pd.DataFrame(data)
    st.subheader("Simulated Radiologist Diagnoses")
    st.info("0 = Benign, 1 = Cancerous. Rows represent individual X-rays.")
    st.dataframe(annotation_data.head())

# Extract numeric rating columns
rating_cols = [c for c in annotation_data.columns if "Radiologist" in c or "Rater" in c or "Res" in c]
if not rating_cols:
    rating_cols = annotation_data.select_dtypes(include=[np.number]).columns.tolist()
    if "Xray_ID" in rating_cols: rating_cols.remove("Xray_ID")

numeric_data = annotation_data[rating_cols]

# ----------------------------
# ICC & Agreement Analysis
# ----------------------------
def icc(data):
    # Simplified ICC formulation for binary/consistency check
    # Variance of mean ratings vs mean of variances
    mean_ratings = data.mean(axis=1)
    total_var = mean_ratings.var()
    within_item_var = data.var(axis=1).mean()
    return (total_var - within_item_var) / total_var if total_var > 0 else 0

icc_value = icc(numeric_data)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Inter-Rater Reliability")
    st.metric(
        label="ICC Score", 
        value=f"{icc_value:.4f}",
        help="A value near 1.0 means radiologists agree perfectly. Lower values indicate inconsistency."
    )

with col2:
    st.subheader("Diagnosis Distribution")
    # Show how many benign vs cancerous overall
    all_ratings = numeric_data.values.flatten()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(all_ratings, bins=[0, 0.5, 1], rwidth=0.8, color='skyblue')
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(['Benign (0)', 'Cancerous (1)'])
    st.pyplot(fig)

# ----------------------------
# Identify Ambiguous Images
# ----------------------------
st.markdown("---")
st.subheader("🔍 Review High-Disagreement Images")
st.write("The X-rays below have the highest variance in diagnosis (e.g., split decisions). These are candidates for expert re-review.")

# Calculate variance per row (image)
row_variance = numeric_data.var(axis=1)
annotation_data['Disagreement_Score'] = row_variance

# Sort by highest disagreement
high_disagreement = annotation_data.sort_values(by='Disagreement_Score', ascending=False).head(10)
st.dataframe(high_disagreement.style.background_gradient(subset=['Disagreement_Score'], cmap='Reds'))

# ----------------------------
# Model Performance Analysis
# ----------------------------
st.markdown("---")
st.subheader("🤖 AI Model Performance vs. Number of Radiologists")
st.write("This curve shows how adding more radiologists to the consensus label improves the AI's ability to detect cancer.")

if not SKLEARN_AVAILABLE:
    st.error("scikit-learn is not installed. Please install it to run the model simulation.")
else:
    accuracies = []
    num_raters_list = range(1, len(rating_cols) + 1)
    
    # We need a ground truth for testing. In simulation, we have it. 
    # In uploaded data, we usually assume the 'majority vote' of ALL raters is the best proxy for ground truth.
    if uploaded_file is None:
        y_true_proxy = ground_truth 
    else:
        # Majority vote across all available raters as proxy
        y_true_proxy = (numeric_data.mean(axis=1) > 0.5).astype(int)

    # Train loop
    progress_bar = st.progress(0)
    for i, num in enumerate(num_raters_list):
        # Select subset of raters
        subset_cols = rating_cols[:num]
        X_subset = numeric_data[subset_cols]
        
        # Create Consensus Label (Majority Vote) from this subset
        # This is what the AI learns from
        y_consensus = (X_subset.mean(axis=1) > 0.5).astype(int)
        
        # Features: For this simulation, we simulate "image features" that correlate with the true label
        # In a real app, you'd load image embeddings. Here we simulate features with noise.
        # We generate features based on the CONSENSUS label to simulate learning "what the doctors say"
        np.random.seed(i) # varying seed
        # Synthetic features: 10 features, some correlated with y_consensus
        X_features = np.random.normal(loc=y_consensus[:, None], scale=1.5, size=(num_samples, 10))
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_true_proxy, test_size=0.3, random_state=42)
        
        # Train
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        progress_bar.progress((i + 1) / len(num_raters_list))

    # Plot
    results_df = pd.DataFrame({"Count of Radiologists": list(num_raters_list), "Model Accuracy": accuracies})
    st.line_chart(results_df, x="Count of Radiologists", y="Model Accuracy")
    
    st.info("""
    **Interpretation:** * **Diminishing Returns:** Notice how the accuracy curve typically flattens out. The "elbow" of this curve suggests the optimal number of radiologists needed (cost-benefit).
    * **Noise Reduction:** More radiologists = stable consensus = better training data for the AI.
    """)

st.markdown("---")
st.markdown("### References")
st.markdown("""
* **Inter-rater Reliability:** 
* **Consensus Labeling:** Using majority vote or expert review to correct noisy labels.
""")
