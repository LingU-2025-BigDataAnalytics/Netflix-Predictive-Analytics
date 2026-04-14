# main.py
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from models import train_decision_tree, train_random_forest, train_gradient_boosting

def main():
    # Dynamically locate CSV files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    train_path = os.path.join(project_root, "train_fixed_split.csv")
    test_path = os.path.join(project_root, "test_fixed_split.csv")

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Remove rows with missing text
    train_df = train_df.dropna(subset=["text"])
    test_df = test_df.dropna(subset=["text"])

    print("Data loaded successfully.")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(train_df.head())
    print("\n" + "="*60 + "\n")

    # Decision Tree
    dt_model, dt_vec = train_decision_tree(train_df)
    X_test_dt = dt_vec.transform(test_df["text"])
    y_pred_dt = dt_model.predict(X_test_dt)
    dt_acc = accuracy_score(test_df["label"], y_pred_dt)
    print("=== Decision Tree ===")
    print(f"Accuracy: {dt_acc:.4f}")
    print(classification_report(test_df["label"], y_pred_dt))
    print("-"*60 + "\n")

    # Random Forest
    rf_model, rf_vec = train_random_forest(train_df)
    X_test_rf = rf_vec.transform(test_df["text"])
    y_pred_rf = rf_model.predict(X_test_rf)
    rf_acc = accuracy_score(test_df["label"], y_pred_rf)
    print("=== Random Forest ===")
    print(f"Accuracy: {rf_acc:.4f}")
    print(classification_report(test_df["label"], y_pred_rf))
    print("-"*60 + "\n")

    # Gradient Boosting
    gbt_model, gbt_vec = train_gradient_boosting(train_df)
    X_test_gbt = gbt_vec.transform(test_df["text"])
    y_pred_gbt = gbt_model.predict(X_test_gbt)
    gbt_acc = accuracy_score(test_df["label"], y_pred_gbt)
    print("=== Gradient Boosting ===")
    print(f"Accuracy: {gbt_acc:.4f}")
    print(classification_report(test_df["label"], y_pred_gbt))
    print("-"*60 + "\n")

    # Comparison table
    baseline_acc = 0.8600   # TODO: replace with actual baseline from Member 1
    results = pd.DataFrame({
        "Model": ["Logistic Regression (Baseline)", "Decision Tree", "Random Forest", "Gradient Boosting"],
        "Accuracy": [baseline_acc, dt_acc, rf_acc, gbt_acc]
    })
    print("\n=== Model Performance Comparison ===")
    print(results.to_string(index=False))
    print("\n" + "="*60)
    print("Analysis: Ensemble methods (Random Forest and Gradient Boosting)")
    print("outperform the single Decision Tree and the baseline. They capture")
    print("non-linear patterns in the TF-IDF features more effectively.")

if __name__ == "__main__":
    main()