# text_vectorization.py

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy import sparse # Import sparse for type hinting
from typing import Tuple

def vectorize_reviews_tfidf(reviews: pd.Series, max_features: int = 10000) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
    """
    Vectorizes text reviews using TF-IDF.

    Args:
        reviews (pd.Series): A pandas Series containing the text reviews,
                             typically obtained from load_amazon_reviews().

    Returns:
        Tuple[sparse.csr_matrix, TfidfVectorizer]: A tuple containing:
            - tfidf_matrix: The TF-IDF matrix representation of the reviews
                            (a SciPy CSR sparse matrix).
            - tfidf_vectorizer : The fitted TfidfVectorizer instance. This
                                 object can be used later to transform new
                                 text data using the same learned vocabulary
                                 and IDF weights.
    """
    print("Initializing TfidfVectorizer...")
    # stop_words='english': Removes common English words (like 'the', 'a', 'is').
    # max_features=10000: Limits the number of unique words/terms to the most frequent 10,000.
    #                     This helps manage dimensionality and noise. Adjust as needed.
    # min_df=5: Ignore terms that appear in fewer than 5 documents. Removes very rare terms.
    # ngram_range=(1, 1): Use unigrams (single words). Can be changed to (1, 2) for unigrams+bigrams, etc.
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        min_df=5,
        # max_df=0.95, # Optional: ignore terms appearing in > 95% documents
        ngram_range=(1, 1)
    )

    print("Fitting and transforming reviews using TF-IDF...")
    # Fit the vectorizer to the review text data and transform it
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of features (unique terms after processing): {len(tfidf_vectorizer.vocabulary_)}")

    return tfidf_matrix, tfidf_vectorizer


# Example Usage
if __name__ == '__main__':
    try:
        from Datasets.amazon_dataset import load_amazon_reviews
    except ImportError:
        print("Error: Could not import load_amazon_reviews from amazon_dataset.py.")
        print("Please ensure amazon_dataset.py is in the same directory.")
        exit()

    print("--- Testing TF-IDF Vectorization ---")
    try:
        # Load the raw data using the function from amazon_dataset.py
        review_texts, sentiment_labels = load_amazon_reviews()

        # Perform TF-IDF vectorization
        tfidf_features, tfidf_vectorizer  = vectorize_reviews_tfidf(review_texts)

        print("\nVectorization successful.")
        print(f"Shape of the resulting TF-IDF feature matrix: {tfidf_features.shape}")
        print(f"Type of the resulting matrix: {type(tfidf_features)}")
        print(f"Number of learned features (terms): {len(tfidf_vectorizer.vocabulary_)}")

        # You can inspect some feature names
        # feature_names = fitted_vectorizer.get_feature_names_out()
        # print("\nFirst 10 learned feature names (terms):")
        # print(feature_names[:10].tolist())

        # Note: The output is a sparse matrix.
        # To see a dense representation (use only for very small data):
        # print("\nDense matrix representation (first 5 rows, first 100 columns):")
        # print(tfidf_features[:5, :100].toarray())

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        print("Please ensure you have scikit-learn installed (`pip install scikit-learn`)")
        print("And that amazon_dataset.py ran successfully.")

    print("--- Test complete ---")