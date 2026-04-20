# main.py
# This script executes the entire training and evaluation workflow.

import os
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from models import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_gradient_boosting
)
import pandas as pd
from sklearn.metrics import classification_report

def main():
    spark = SparkSession.builder \
        .appName("Comprehensive_Model_Comparison") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    train_path = "train_fixed_split.csv"
    test_path = "test_fixed_split.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: CSV files not found at {train_path} or {test_path}")
        return

    # Load data using "Review" column
    train_df = spark.read.csv(train_path, header=True, inferSchema=True).dropna(subset=["Review"])
    test_df = spark.read.csv(test_path, header=True, inferSchema=True).dropna(subset=["Review"])

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    model_tasks = [
        ("Logistic Regression (Baseline)", train_logistic_regression),
        ("Decision Tree (Weighted)", train_decision_tree),
        ("Random Forest (Weighted)", train_random_forest),
        ("Gradient Boosting", train_gradient_boosting)
    ]

    # Based on Notebook output, StringIndexer order is ['positive', 'negative']
    label_names = ["positive", "negative"] 
    final_results = []

    for model_name, train_func in model_tasks:
        print(f"\n>>> Running {model_name}...")
        
        model, feat_pipeline = train_func(train_df)
        
        # Transform test data (now includes indexing and text features)
        test_transformed = feat_pipeline.transform(test_df)
        predictions = model.transform(test_transformed)
        
        accuracy = evaluator.evaluate(predictions)
        final_results.append({"Model": model_name, "Accuracy": accuracy})
        
        pred_data = predictions.select("label", "prediction").collect()
        y_true = [int(row.label) for row in pred_data]
        y_pred = [int(row.prediction) for row in pred_data]
        
        print(f"--- {model_name} Classification Report ---")
        print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    comparison_df = pd.DataFrame(final_results)
    print("\n" + "="*40)
    print("      FINAL MODEL PERFORMANCE COMPARISON")
    print("="*40)
    print(comparison_df.to_string(index=False))
    print("="*40)

    spark.stop()

if __name__ == "__main__":
    main()