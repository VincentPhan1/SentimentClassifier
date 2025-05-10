"""
Provides functionality to download, load, and preprocess the Amazon Fine Food Reviews dataset from Kaggle.
"""
__author__  = "Vincent Phan"
__email__   = "vdp21005@utdallas.edu"

import kagglehub
import pandas as pd
import os
from typing import Tuple

def load_amazon_reviews() -> Tuple[pd.Series, pd.Series]:
    """
    Downloads the Amazon Fine Food Reviews dataset from Kaggle,
    processes it, and returns review text and binary sentiment labels.

    Scores 4 and 5 are considered positive (1), while scores 1 and 2
    are considered negative (0). Reviews with a score of 3 (neutral)
    are excluded.

    Returns:
        Tuple[X, y]: A tuple containing:
            X - The review text data.
            Y - The binary sentiment labels (0 for negative, 1 for positive).
    """
    print("Downloading Amazon Fine Food Reviews dataset from Kaggle...")
    try:
        # Download the latest version of the dataset
        cache_root_path = "/data/amazon"
        download_path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
        print(f"Dataset downloaded to: {download_path}")

        # We need to find the actual CSV file path within the downloaded directory.
        csv_file_name = 'Reviews.csv'
        csv_file_path = os.path.join(download_path, csv_file_name)

        if not os.path.exists(csv_file_path):
             # Fallback: list files in the directory and pick the first CSV if the expected name isn't found
            csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
            if not csv_files:
                 raise FileNotFoundError(f"No CSV file found in the downloaded directory: {download_path}")
            csv_file_path = os.path.join(download_path, csv_files[0])
            print(f"'{csv_file_name}' not found, using '{csv_files[0]}' instead.")


        print(f"Loading data from {csv_file_path}...")
        # Load the dataset
        df = pd.read_csv(csv_file_path)

        # Drop rows with missing values in relevant columns if any, though often not an issue for this dataset
        df.dropna(subset=['Score', 'Text'], inplace=True)

        print(f"Original dataset shape: {df.shape}")

        # Filter out neutral reviews (Score == 3)
        df_filtered = df[df['Score'] != 3].copy() # Use .copy() to avoid SettingWithCopyWarning
        print(f"Shape after filtering out neutral reviews (Score=3): {df_filtered.shape}")


        # Map scores to binary sentiment: 1, 2 -> 0 (negative); 4, 5 -> 1 (positive)
        # Ensure the column is numeric before mapping
        df_filtered['Score'] = pd.to_numeric(df_filtered['Score'], errors='coerce')
        # Drop rows where Score couldn't be converted to numeric (should be none after dropna, but good practice)
        df_filtered.dropna(subset=['Score'], inplace=True)


        df_filtered['Sentiment'] = df_filtered['Score'].apply(lambda score: 1 if score > 3 else 0)

        # Extract text and sentiment labels
        reviews = df_filtered['Text']
        labels = df_filtered['Sentiment']

        print("Data processing complete. Returning text reviews and binary sentiment labels.")

        return reviews, labels

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the 'kagglehub' and 'pandas' libraries installed (`pip install kagglehub pandas`).")
        raise # Re-raise the exception after printing the message


if __name__ == '__main__':
    # You can run this script directly to test the function
    print("--- Testing load_amazon_reviews function ---")
    try:
        review_texts, sentiment_labels = load_amazon_reviews()

        print("\nSuccessfully loaded data.")
        print(f"Number of reviews: {len(review_texts)}")
        print(f"Number of labels: {len(sentiment_labels)}")
        print("\nFirst 5 reviews:")
        for i, text in enumerate(review_texts.head()):
            print(f"Review {i+1}: {text[:100]}...") # Print first 100 chars
        print("\nFirst 5 labels:")
        print(sentiment_labels.head().tolist()) # Convert to list for cleaner printing
        print("\nLabel distribution:")
        print(sentiment_labels.value_counts())


    except Exception as e:
        print(f"\nTest failed due to an error: {e}")

    print("--- Test complete ---")