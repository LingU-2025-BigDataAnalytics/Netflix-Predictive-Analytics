# models.py
# PySpark version: Decision Tree, Random Forest, Gradient Boosting
# Each function returns a trained model and the feature pipeline.

from pyspark.sql import DataFrame
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline

def _build_feature_pipeline(num_features=5000):
    """Build TF-IDF feature pipeline."""
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=num_features)
    idf = IDF(inputCol="raw_features", outputCol="features")
    return Pipeline(stages=[tokenizer, remover, hashingTF, idf])

def train_decision_tree(train_df: DataFrame, max_depth=10, seed=42):
    feature_pipeline = _build_feature_pipeline()
    pipeline_model = feature_pipeline.fit(train_df)
    train_features = pipeline_model.transform(train_df)
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features",
                                maxDepth=max_depth, seed=seed)
    model = dt.fit(train_features)
    return model, pipeline_model

def train_random_forest(train_df: DataFrame, num_trees=50, max_depth=10, seed=42):
    feature_pipeline = _build_feature_pipeline()
    pipeline_model = feature_pipeline.fit(train_df)
    train_features = pipeline_model.transform(train_df)
    rf = RandomForestClassifier(labelCol="label", featuresCol="features",
                                numTrees=num_trees, maxDepth=max_depth, seed=seed)
    model = rf.fit(train_features)
    return model, pipeline_model

def train_gradient_boosting(train_df: DataFrame, max_iter=50, max_depth=5, seed=42):
    feature_pipeline = _build_feature_pipeline()
    pipeline_model = feature_pipeline.fit(train_df)
    train_features = pipeline_model.transform(train_df)
    gbt = GBTClassifier(labelCol="label", featuresCol="features",
                        maxIter=max_iter, maxDepth=max_depth, seed=seed)
    model = gbt.fit(train_features)
    return model, pipeline_model