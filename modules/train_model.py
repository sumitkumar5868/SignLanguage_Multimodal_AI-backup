"""Phase 6: Machine Learning Model Training and Evaluation Pipeline.

Pipeline:
CSV Dataset → Load → Validate → Separate Features/Labels → Train/Test Split →
Feature Scaling → Model Training → Evaluation → Save Model/Artifacts
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import joblib

# Paths
DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "sign_landmarks.csv"
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CONFUSION_MATRIX_PATH = DATA_DIR / "confusion_matrix.png"

# Model configuration
MODEL_PATH = MODEL_DIR / "sign_language_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Expected configuration
EXPECTED_FEATURES = 63
EXPECTED_TOTAL_COLUMNS = 64
LABEL_COLUMN = "label"
EXPECTED_CLASSES = ["HELLO", "I_LOVE_YOU", "I_HATE_YOU", "I_EAT", "THANK_YOU"]
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 200


def print_header():
    """Print the project header."""
    print("=" * 80)
    print(" SIGN LANGUAGE MULTIMODAL AI")
    print("=" * 80)
    print()
    print("PHASE 6: MACHINE LEARNING TRAINING")
    print()


def load_dataset(path=None):
    """Load the dataset from CSV file.
    
    Args:
        path: Path to CSV file. Defaults to DATASET_PATH.
    
    Returns:
        DataFrame containing the dataset, or None if loading fails.
    """
    target_path = Path(path) if path else DATASET_PATH
    
    print("Dataset:")
    print(f"{target_path}")
    print()
    
    if not target_path.exists():
        print("ERROR: Dataset file not found.")
        return None
    
    try:
        df = pd.read_csv(target_path)
        return df
    except Exception as exc:
        print(f"ERROR: Failed to load dataset: {str(exc)}")
        return None


def validate_dataset(df):
    """Validate dataset structure and content.
    
    Args:
        df: DataFrame to validate.
    
    Returns:
        True if valid, False otherwise.
    """
    print("-" * 80)
    print("DATASET VALIDATION")
    print("-" * 80)
    print()
    
    if df is None or df.empty:
        print("✗ Dataset is empty")
        return False
    
    print(f"✓ Dataset found")
    print(f"✓ {len(df)} samples loaded")
    
    # Check shape
    if df.shape[1] != EXPECTED_TOTAL_COLUMNS:
        print(f"✗ Expected {EXPECTED_TOTAL_COLUMNS} columns, got {df.shape[1]}")
        return False
    
    # Check features
    if df.shape[1] - 1 != EXPECTED_FEATURES:
        print(f"✗ Expected {EXPECTED_FEATURES} feature columns, got {df.shape[1] - 1}")
        return False
    
    print(f"✓ {EXPECTED_FEATURES} features detected")
    
    # Check label column
    if LABEL_COLUMN not in df.columns:
        print(f"✗ Label column '{LABEL_COLUMN}' not found")
        return False
    
    print(f"✓ Label column detected")
    
    # Check missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"✗ Found {missing_count} missing values")
        return False
    
    print(f"✓ No missing values")
    
    # Check classes
    classes = sorted(df[LABEL_COLUMN].unique().tolist())
    if len(classes) != len(EXPECTED_CLASSES):
        print(f"✗ Expected {len(EXPECTED_CLASSES)} classes, got {len(classes)}")
        print(f"  Found: {classes}")
        return False
    
    if set(classes) != set(EXPECTED_CLASSES):
        print(f"✗ Class mismatch")
        print(f"  Expected: {EXPECTED_CLASSES}")
        print(f"  Found: {classes}")
        return False
    
    print(f"✓ {len(classes)} classes detected")
    
    print()
    print("Classes:")
    for label in EXPECTED_CLASSES:
        count = len(df[df[LABEL_COLUMN] == label])
        print(f"- {label:15} {count:3} samples")
    
    print()
    return True


def separate_features_and_labels(df):
    """Separate features and labels from dataset.
    
    Args:
        df: DataFrame containing features and label.
    
    Returns:
        Tuple (X, y) containing features and labels, or (None, None) on error.
    """
    print("-" * 80)
    print("FEATURE AND LABEL SEPARATION")
    print("-" * 80)
    print()
    
    try:
        X = df.drop(columns=[LABEL_COLUMN]).values
        y = df[LABEL_COLUMN].values
        
        print(f"Feature shape: {X.shape}")
        print(f"Label shape: {y.shape}")
        print()
        
        return X, y
    except Exception as exc:
        print(f"ERROR: Failed to separate features and labels: {str(exc)}")
        return None, None


def split_train_test(X, y):
    """Split dataset into training and testing sets.
    
    Args:
        X: Features array.
        y: Labels array.
    
    Returns:
        Tuple (X_train, X_test, y_train, y_test) or (None, None, None, None) on error.
    """
    print("-" * 80)
    print("TRAIN / TEST SPLIT")
    print("-" * 80)
    print()
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print()
        
        # Show class distribution
        train_counts = {}
        test_counts = {}
        for label in EXPECTED_CLASSES:
            train_counts[label] = np.sum(y_train == label)
            test_counts[label] = np.sum(y_test == label)
        
        print("Class distribution:")
        for label in EXPECTED_CLASSES:
            print(f"  {label:15} Train: {train_counts[label]:2}  Test: {test_counts[label]:2}")
        print()
        
        return X_train, X_test, y_train, y_test
    except Exception as exc:
        print(f"ERROR: Failed to split data: {str(exc)}")
        return None, None, None, None


def scale_features(X_train, X_test):
    """Scale features using StandardScaler.
    
    Args:
        X_train: Training features.
        X_test: Testing features.
    
    Returns:
        Tuple (scaler, X_train_scaled, X_test_scaled) or (None, None, None) on error.
    """
    print("-" * 80)
    print("FEATURE SCALING")
    print("-" * 80)
    print()
    
    try:
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("✓ StandardScaler fitted on training data")
        print("✓ Training and testing data scaled")
        print()
        
        return scaler, X_train_scaled, X_test_scaled
    except Exception as exc:
        print(f"ERROR: Failed to scale features: {str(exc)}")
        return None, None, None


def train_model(X_train_scaled, y_train):
    """Train RandomForestClassifier model.
    
    Args:
        X_train_scaled: Scaled training features.
        y_train: Training labels.
    
    Returns:
        Trained model or None on error.
    """
    print("-" * 80)
    print("MODEL TRAINING")
    print("-" * 80)
    print()
    
    try:
        print(f"Model: RandomForestClassifier")
        print(f"Estimators: {N_ESTIMATORS}")
        print()
        print("Training...")
        
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        
        model.fit(X_train_scaled, y_train)
        
        print("✓ Model training completed")
        print()
        
        return model
    except Exception as exc:
        print(f"ERROR: Failed to train model: {str(exc)}")
        return None


def evaluate_model(model, X_test_scaled, y_test, le):
    """Evaluate model on test set.
    
    Args:
        model: Trained model.
        X_test_scaled: Scaled test features.
        y_test: Test labels (encoded).
        le: Label encoder for decoding labels.
    
    Returns:
        Tuple (accuracy, y_pred) or (None, None) on error.
    """
    print("-" * 80)
    print("MODEL EVALUATION")
    print("-" * 80)
    print()
    
    try:
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print()
        
        # Decode labels for classification report
        y_test_labels = le.inverse_transform(y_test)
        y_pred_labels = le.inverse_transform(y_pred)
        
        print("Classification Report:")
        print()
        print(classification_report(y_test_labels, y_pred_labels))
        
        return accuracy, y_pred, y_pred_labels, y_test_labels
    except Exception as exc:
        print(f"ERROR: Failed to evaluate model: {str(exc)}")
        return None, None, None, None


def generate_confusion_matrix(y_test, y_pred, le):
    """Generate and save confusion matrix.
    
    Args:
        y_test: True test labels (encoded).
        y_pred: Predicted labels (encoded).
        le: Label encoder for class names.
    
    Returns:
        True on success, False on error.
    """
    print("-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    print()
    
    try:
        cm = confusion_matrix(y_test, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set ticks
        tick_marks = np.arange(len(EXPECTED_CLASSES))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(EXPECTED_CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(EXPECTED_CLASSES)
        
        # Add labels
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Confusion Matrix - Sign Language Classification")
        
        # Add text annotations
        threshold = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )
        
        plt.tight_layout()
        
        # Create data directory if needed
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        plt.savefig(CONFUSION_MATRIX_PATH, dpi=100, bbox_inches="tight")
        print(f"✓ Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
        print()
        
        plt.close(fig)
        
        return True
    except Exception as exc:
        print(f"ERROR: Failed to generate confusion matrix: {str(exc)}")
        return False


def save_model_artifacts(model, scaler, le, X_train, y_train, accuracy):
    """Save trained model and related artifacts.
    
    Args:
        model: Trained model.
        scaler: Fitted scaler.
        le: Label encoder.
        X_train: Training features (for sample count).
        y_train: Training labels (for sample count).
        accuracy: Model accuracy.
    
    Returns:
        True on success, False on error.
    """
    print("-" * 80)
    print("SAVING MODEL FILES")
    print("-" * 80)
    print()
    
    try:
        # Create model directory
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, MODEL_PATH)
        print(f"✓ {MODEL_PATH.name}")
        
        # Save scaler
        joblib.dump(scaler, SCALER_PATH)
        print(f"✓ {SCALER_PATH.name}")
        
        # Save label encoder
        joblib.dump(le, LABEL_ENCODER_PATH)
        print(f"✓ {LABEL_ENCODER_PATH.name}")
        
        # Create and save metadata
        metadata = {
            "model": "RandomForestClassifier",
            "features": EXPECTED_FEATURES,
            "classes": EXPECTED_CLASSES,
            "training_samples": len(X_train),
            "testing_samples": int(len(X_train) / (1 - TEST_SIZE) * TEST_SIZE),
            "dataset_samples": len(X_train) + int(len(X_train) / (1 - TEST_SIZE) * TEST_SIZE),
            "random_state": RANDOM_STATE,
            "n_estimators": N_ESTIMATORS,
            "accuracy": float(accuracy),
        }
        
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ {METADATA_PATH.name}")
        
        # Confusion matrix (saved separately)
        if CONFUSION_MATRIX_PATH.exists():
            print(f"✓ confusion_matrix.png")
        
        print()
        return True
    except Exception as exc:
        print(f"ERROR: Failed to save model artifacts: {str(exc)}")
        return False


def test_model_loading():
    """Test that saved model and artifacts can be loaded.
    
    Returns:
        True if all files load successfully, False otherwise.
    """
    print("-" * 80)
    print("MODEL LOADING TEST")
    print("-" * 80)
    print()
    
    try:
        # Load model
        model = joblib.load(MODEL_PATH)
        print("✓ Model loaded successfully")
        
        # Load scaler
        scaler = joblib.load(SCALER_PATH)
        print("✓ Scaler loaded successfully")
        
        # Load label encoder
        le = joblib.load(LABEL_ENCODER_PATH)
        print("✓ Label encoder loaded successfully")
        
        # Load metadata
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print("✓ Metadata loaded successfully")
        
        print()
        return True
    except Exception as exc:
        print(f"ERROR: Failed to load model artifacts: {str(exc)}")
        print(f"Details: {str(exc)}")
        return False


def main():
    """Main training pipeline."""
    print_header()
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return 1
    
    # Validate dataset
    if not validate_dataset(df):
        return 1
    
    # Separate features and labels
    X, y = separate_features_and_labels(df)
    if X is None or y is None:
        return 1
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y_encoded)
    if X_train is None:
        return 1
    
    # Scale features
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    if scaler is None:
        return 1
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    if model is None:
        return 1
    
    # Evaluate model
    accuracy, y_pred, y_pred_labels, y_test_labels = evaluate_model(
        model, X_test_scaled, y_test, le
    )
    if accuracy is None:
        return 1
    
    # Generate confusion matrix
    if not generate_confusion_matrix(y_test, y_pred, le):
        return 1
    
    # Save model artifacts
    if not save_model_artifacts(model, scaler, le, X_train, y_train, accuracy):
        return 1
    
    # Test loading
    if not test_model_loading():
        return 1
    
    print("=" * 80)
    print(" PHASE 6 COMPLETED")
    print("=" * 80)
    print()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
