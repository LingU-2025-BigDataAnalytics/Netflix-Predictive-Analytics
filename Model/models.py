# models.py
# Contains three tree-based models for text sentiment classification:
# Decision Tree, Random Forest, and Gradient Boosting.
# Each function returns the trained model and its fitted TF-IDF vectorizer.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def _get_vectorizer(max_features=10000, min_df=2, ngram_range=(1, 2)):
    """Return a standard TF-IDF vectorizer with fixed parameters."""
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range
    )


def train_decision_tree(train_df, max_depth=10, random_state=42):
    """
    Train a Decision Tree classifier on the training data.

    Parameters:
        train_df (DataFrame): Must contain columns 'text' (str) and 'label' (int).
        max_depth (int): Maximum depth of the tree.
        random_state (int): Seed for reproducibility.

    Returns:
        model: Fitted DecisionTreeClassifier.
        vectorizer: Fitted TfidfVectorizer.
    """
    vectorizer = _get_vectorizer()
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model, vectorizer


def train_random_forest(train_df, n_estimators=50, max_depth=10, random_state=42, n_jobs=-1):
    """
    Train a Random Forest classifier on the training data.

    Parameters:
        train_df (DataFrame): Must contain columns 'text' (str) and 'label' (int).
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree.
        random_state (int): Seed for reproducibility.
        n_jobs (int): Number of parallel jobs (-1 uses all cores).

    Returns:
        model: Fitted RandomForestClassifier.
        vectorizer: Fitted TfidfVectorizer.
    """
    vectorizer = _get_vectorizer()
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    return model, vectorizer


def train_gradient_boosting(train_df, n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42):
    """
    Train a Gradient Boosting classifier on the training data.

    Parameters:
        train_df (DataFrame): Must contain columns 'text' (str) and 'label' (int).
        n_estimators (int): Number of boosting stages.
        max_depth (int): Maximum depth of each tree.
        learning_rate (float): Shrinks the contribution of each tree.
        random_state (int): Seed for reproducibility.

    Returns:
        model: Fitted GradientBoostingClassifier.
        vectorizer: Fitted TfidfVectorizer.
    """
    vectorizer = _get_vectorizer()
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model, vectorizer