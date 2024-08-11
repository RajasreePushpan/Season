import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

# Path to the example dataset
EXAMPLE_DATASET_PATH = 'https://raw.githubusercontent.com/RajasreePushpan/Season/main/weather_classification_data.csv'


# Helper function to plot confusion matrix
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)


# Helper function to plot ROC curve for binary classification
def plot_roc_curve_binary(y_test, y_prob, pos_label):
    # Ensure y_test is numeric
    y_test_numeric = (y_test == pos_label).astype(int)

    # Compute ROC curve and ROC area for the positive class
    fpr, tpr, _ = roc_curve(y_test_numeric, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)


# Helper function to plot ROC curve for multi-class classification
def plot_roc_curve_multi_class(y_test, y_prob):
    n_classes = y_prob.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)


# Title of the app
st.title('Classification Analysis Streamlit Application')

# Initialize data variable
data = None

# Sidebar for selecting example datasets
st.sidebar.title("Example Datasets")
example_dataset = st.sidebar.selectbox(
    "Select an example dataset",
    ["None", "Weather Classification Data"]
)

# Load example dataset if selected
if example_dataset == "Weather Classification Data":
    try:
        data = pd.read_csv(EXAMPLE_DATASET_PATH)
        # Remove rows with any null values
        data = data.dropna()
    except FileNotFoundError:
        st.error(f"Example dataset '{EXAMPLE_DATASET_PATH}' not found. Please upload your own dataset.")

# Upload Dataset
uploaded_file = st.file_uploader("Or upload your own dataset (CSV file)", type='csv')

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    # Remove rows with any null values
    data = data.dropna()

if data is None:
    st.warning("Please select or upload a dataset.")
    st.stop()

# Automatically select the last column as the target column
target_column = data.columns[-1]

X = data.drop(target_column, axis=1)
y = data[target_column]

# Initialize session state variables if not already initialized
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None

st.subheader("Data Integrity Checks")
if st.button("Check for Missing Values"):
    st.write("No missing values in the dataset.") if data.isnull().sum().sum() == 0 else st.write(data.isnull().sum())

if st.button("Show Dataset Shape"):
    st.write(f"Dataset Shape: {data.shape}")

if st.button("Show First Few Rows"):
    st.write(data.head())

st.subheader("Exploratory Data Analysis (EDA)")

if st.button("Generate Histograms"):
    fig, ax = plt.subplots(figsize=(10, 8))
    data.hist(ax=ax, bins=30)
    st.pyplot(fig)

if st.button("Generate Pair Plot"):
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        st.write("Pair plot is not supported as there are no numeric features in the dataset.")
    else:
        fig = sns.pairplot(numeric_data).fig
        st.pyplot(fig)

if st.button("Generate Correlation Heatmap"):
    # Convert categorical data to numeric
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        st.write("No numeric data available to plot correlation heatmap.")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

st.subheader("Model Selection and Training")

model_name = st.selectbox("Select Model", ["None", "Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])

scaler_option = st.selectbox("Select Scaler", ["None", "StandardScaler", "MinMaxScaler"])

if model_name != "None":
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    if len(categorical_cols) > 0:
        # Create a column transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(), categorical_cols)
            ],
            remainder='passthrough'
        )
    else:
        preprocessor = None

    if scaler_option == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_option == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = None

    # Apply preprocessing
    if preprocessor:
        X_preprocessed = preprocessor.fit_transform(X)
    else:
        X_preprocessed = X

    if scaler:
        X_preprocessed = scaler.fit_transform(X_preprocessed)

    if st.button('Split Dataset'):
        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
            X_preprocessed, y, test_size=0.2, random_state=42)
        st.write("Dataset has been split into training and testing sets.")

    if st.session_state.X_train is not None and st.session_state.y_train is not None:
        if st.button('Train Model'):
            if model_name == "Logistic Regression":
                st.session_state.model = LogisticRegression(max_iter=1000)
            elif model_name == "Decision Tree":
                st.session_state.model = DecisionTreeClassifier()
            elif model_name == "K-Nearest Neighbors":
                st.session_state.model = KNeighborsClassifier()

            st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
            st.write(f"Training complete. Model: {model_name}")

        if st.session_state.model:
            if st.button('Predict'):
                if st.session_state.X_test is not None and st.session_state.y_test is not None:
                    y_pred = st.session_state.model.predict(st.session_state.X_test)
                    accuracy = accuracy_score(st.session_state.y_test, y_pred)
                    st.write(f"Accuracy: {accuracy:.2f}")

                    st.write("Confusion Matrix:")
                    cm = confusion_matrix(st.session_state.y_test, y_pred)
                    plot_confusion_matrix(cm, np.unique(st.session_state.y_test))

                    st.write("Classification Report:")
                    st.text(classification_report(st.session_state.y_test, y_pred))

                    if hasattr(st.session_state.model, "predict_proba"):
                        y_prob = st.session_state.model.predict_proba(st.session_state.X_test)
                        if len(np.unique(st.session_state.y_test)) == 2:  # Binary Classification
                            positive_class = np.unique(st.session_state.y_test)[
                                1]  # Assuming the second class is positive
                            plot_roc_curve_binary(st.session_state.y_test, y_prob[:, 1], positive_class)
                        else:
                            y_test_binarized = label_binarize(st.session_state.y_test, classes=np.unique(y_pred))
                            plot_roc_curve_multi_class(y_test_binarized, y_prob)
                    else:
                        st.warning("The selected model does not support probability prediction.")

                    results_df = pd.DataFrame({'True Label': st.session_state.y_test, 'Predicted Label': y_pred})
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download Predictions", csv, "predictions.csv")
                else:
                    st.warning("Please split the dataset before making predictions.")
