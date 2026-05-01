import os
import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Allows importing from src/preprocessing when running this file directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(SRC_DIR)

from preprocessing.mfcc_feature_extraction import extract_features_from_file


TRAIN_DIR = "data/training"
TEST_DIR = "data/testing"


def load_dataset(folder):
    X, y = [], []

    for fname in os.listdir(folder):
        if not fname.endswith(".wav"):
            continue

        if fname.startswith("cough"):
            label = 1
        elif fname.startswith("non_cough"):
            label = 0
        else:
            continue

        file_path = os.path.join(folder, fname)

        try:
            features = extract_features_from_file(file_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    return np.array(X), np.array(y)


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    f1_per_class = f1_score(y_test, y_pred, average=None)

    weighted_precision = precision_score(y_test, y_pred, average="weighted")
    weighted_recall = recall_score(y_test, y_pred, average="weighted")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix (%)")
    print("           Pred_Cough   Pred_NonCough")
    print(f"Actual_Cough     {cm_percent[1][1]:.1f}%        {cm_percent[1][0]:.1f}%")
    print(f"Actual_NonCough  {cm_percent[0][1]:.1f}%        {cm_percent[0][0]:.1f}%")

    print("\nF1 Score")
    print(f"Cough: {f1_per_class[1]:.2f}")
    print(f"Non-Cough: {f1_per_class[0]:.2f}")

    print("\nWeighted Metrics")
    print(f"AUC: {auc:.2f}")
    print(f"Precision: {weighted_precision:.2f}")
    print(f"Recall: {weighted_recall:.2f}")
    print(f"F1 Score: {weighted_f1:.2f}")


def main():
    X_train, y_train = load_dataset(TRAIN_DIR)
    X_test, y_test = load_dataset(TEST_DIR)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    evaluate_model(clf, X_test, y_test)


if __name__ == "__main__":
    main()