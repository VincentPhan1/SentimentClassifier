# Models/sentiment_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer # Used for fitting/transforming

import sys
import os

# Add the project root to the system path to import modules
# Adjust the path as needed based on where you run the script from
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

# Now import your functions
try:
    # We still import vectorize_reviews_tfidf as it's used internally
    from Models.tfidf_vectorization import vectorize_reviews_tfidf
except ImportError as e:
    print(f"Error importing feature_extraction.py: {e}")
    print("Please ensure 'feature_extraction.py' is in the project root.")
    sys.exit("Required module not found.")


# Modified function to accept pre-split data
def train_and_evaluate_naive_bayes(X_train_raw: pd.Series, X_test_raw: pd.Series, y_train: pd.Series, y_test: pd.Series, max_features: int = 10000):
    """
    Vectorizes pre-split raw text data using TF-IDF, trains a Multinomial Naive Bayes
    classifier on the training data, and evaluates its performance on the test data.

    Args:
        X_train_raw (pd.Series): Pandas Series containing raw training text reviews.
        X_test_raw (pd.Series): Pandas Series containing raw testing text reviews.
        y_train (pd.Series): Pandas Series containing training labels.
        y_test (pd.Series): Pandas Series containing testing labels.
        max_features (int, optional): The maximum number of features (terms)
                                       for TF-IDF. Defaults to 10000.

    Returns:
        MultinomialNB: The trained Multinomial Naive Bayes classifier model.
    """
    print("\n--- Starting Multinomial Naive Bayes Model Process ---")
    print(f"Training data size: {len(X_train_raw)}")
    print(f"Testing data size: {len(X_test_raw)}")


    # --- TF-IDF Vectorization ---
    # We fit the vectorizer ONLY on the training data
    print(f"Initializing TfidfVectorizer with max_features={max_features}...")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

    print("Fitting TfidfVectorizer on training data and transforming...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
    print(f"Shape of training TF-IDF matrix: {X_train_tfidf.shape}")

    # We transform the test data using the vectorizer fitted on the training data
    print("Transforming testing data using the fitted vectorizer...")
    X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)
    print(f"Shape of testing TF-IDF matrix: {X_test_tfidf.shape}")


    # --- Model Training ---
    print("\n--- Starting Model Training ---")
    model = MultinomialNB(alpha=1.0) # Common alpha value

    # Train the model using the TF-IDF transformed training data
    model.fit(X_train_tfidf, y_train)
    print("--- Training Complete ---")

    # --- Model Evaluation ---
    print("\n--- Performing Final Evaluation on Test Set ---")
    # Make predictions on the test set using the transformed test data
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("--- Test Evaluation Complete ---")


    print("\n--- Multinomial Naive Bayes Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("--------------------------------------------------")
    print("Confusion Matrix Explained:")
    print("[[True Negatives, False Positives]")
    print(" [False Negatives, True Positives]]")
    print("--------------------------------------------------")

    print("\nReturning the trained NB model.")
    return model

# Example Usage
if __name__ == '__main__':
    print("--- Testing train_and_evaluate_naive_bayes function (requires data loading) ---")
    try:
        # This example will load data internally for testing purposes
        # In the main comparison script, data is loaded and split ONCE.
        print("Loading data for standalone test...")
        # Assuming amazon_dataset.py is accessible
        from Datasets.amazon_dataset import load_amazon_reviews
        review_texts, sentiment_labels = load_amazon_reviews()
        print("Data loaded.")

        # Perform a simple split for this standalone test
        print("Performing standalone train/test split...")
        X_train_raw_test, X_test_raw_test, y_train_test, y_test_test = train_test_split(
             review_texts, sentiment_labels, test_size=0.2, random_state=42, stratify=sentiment_labels
        )
        print(f"Standalone train size: {len(X_train_raw_test)}, test size: {len(X_test_raw_test)}")


        # Run the training and evaluation using the split data
        trained_nb_model = train_and_evaluate_naive_bayes(
            X_train_raw_test, X_test_raw_test, y_train_test, y_test_test, max_features=5000
        )

        print("\nStandalone test complete. Model object returned.")

    except Exception as e:
        print(f"\nStandalone test failed due to an error: {e}")
        print("Please ensure 'kagglehub', 'pandas', and 'scikit-learn' are installed.")
        print("Also, ensure load_amazon_reviews() and vectorize_reviews_tfidf() are accessible.")

    print("--- Standalone Test Complete ---")
