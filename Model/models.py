# models.py
# Updated to handle missing 'label' columns by indexing 'Sentiment' on the fly.
# All models utilize the same weighted pipeline to ensure fair comparison.

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    SQLTransformer,
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    StringIndexer,
)
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.sql import functions as F

def _add_weights(train_df: DataFrame):
    """
    Computes class weights to handle imbalance (equivalent to class_weight='balanced').
    Requires a numeric 'label' column to exist.
    """
    total_count = train_df.count()
    counts = train_df.groupBy("label").count().collect()
    num_classes = len(counts)
    
    # Create a dictionary for mapping: {label: weight}
    weight_dict = {row['label']: total_count / (num_classes * row['count']) for row in counts}
    
    # Create a conditional mapping column
    mapping_expr = F.create_map([F.lit(x) for x in sum(weight_dict.items(), ())])
    return train_df.withColumn("classWeight", mapping_expr[F.col("label")])

def _build_feature_pipeline(vocab_size=10000, min_df=2.0):
    """
    Feature engineering pipeline for text:
    Text cleaning -> Tokenization -> Stopword removal -> CountVectorization -> IDF.
    """
    clean_text_sql = SQLTransformer(
        statement="""
        SELECT *, lower(regexp_replace(Review, '[^a-zA-Z\\s]', ' ')) AS clean_text FROM __THIS__
        """
    )

    tokenizer = RegexTokenizer(
        inputCol="clean_text", 
        outputCol="tokens", 
        pattern="\\s+"
    )

    remover = StopWordsRemover(
        inputCol="tokens", 
        outputCol="filtered_tokens"
    )

    vectorizer = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        vocabSize=vocab_size,
        minDF=min_df
    )

    idf = IDF(inputCol="raw_features", outputCol="features")

    return Pipeline(stages=[clean_text_sql, tokenizer, remover, vectorizer, idf])

def train_logistic_regression(train_df: DataFrame, seed=42):
    # 1. Ensure 'label' column exists for weight calculation
    label_indexer = StringIndexer(inputCol="Sentiment", outputCol="label").fit(train_df)
    indexed_df = label_indexer.transform(train_df)
    
    # 2. Add weights
    train_weighted = _add_weights(indexed_df)
    
    # 3. Build text feature model
    feat_pipeline = _build_feature_pipeline()
    feat_model = feat_pipeline.fit(train_weighted)
    
    # 4. Prepare training features
    train_features = feat_model.transform(train_weighted)

    lr = LogisticRegression(
        labelCol="label",
        featuresCol="features",
        weightCol="classWeight",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0,
        predictionCol="prediction",
        family="binomial"
    )
    
    # Return the trained classifier and a combined model for the test set
    combined_feat_model = PipelineModel(stages=[label_indexer, feat_model])
    return lr.fit(train_features), combined_feat_model

def train_decision_tree(train_df: DataFrame, max_depth=10, seed=42):
    label_indexer = StringIndexer(inputCol="Sentiment", outputCol="label").fit(train_df)
    indexed_df = label_indexer.transform(train_df)
    
    train_weighted = _add_weights(indexed_df)
    feat_pipeline = _build_feature_pipeline()
    feat_model = feat_pipeline.fit(train_weighted)
    train_features = feat_model.transform(train_weighted)

    dt = DecisionTreeClassifier(
        labelCol="label",
        featuresCol="features",
        weightCol="classWeight",
        maxDepth=max_depth,
        seed=seed
    )
    combined_feat_model = PipelineModel(stages=[label_indexer, feat_model])
    return dt.fit(train_features), combined_feat_model

def train_random_forest(train_df: DataFrame, num_trees=50, max_depth=10, seed=42):
    label_indexer = StringIndexer(inputCol="Sentiment", outputCol="label").fit(train_df)
    indexed_df = label_indexer.transform(train_df)
    
    train_weighted = _add_weights(indexed_df)
    feat_pipeline = _build_feature_pipeline()
    feat_model = feat_pipeline.fit(train_weighted)
    train_features = feat_model.transform(train_weighted)

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        weightCol="classWeight",
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=seed
    )
    combined_feat_model = PipelineModel(stages=[label_indexer, feat_model])
    return rf.fit(train_features), combined_feat_model

def train_gradient_boosting(train_df: DataFrame, max_iter=50, max_depth=5, seed=42):
    label_indexer = StringIndexer(inputCol="Sentiment", outputCol="label").fit(train_df)
    indexed_df = label_indexer.transform(train_df)
    
    feat_pipeline = _build_feature_pipeline()
    feat_model = feat_pipeline.fit(indexed_df)
    train_features = feat_model.transform(indexed_df)

    gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        maxIter=max_iter,
        maxDepth=max_depth,
        seed=seed
    )
    combined_feat_model = PipelineModel(stages=[label_indexer, feat_model])
    return gbt.fit(train_features), combined_feat_model