"""
Provides functionality to download, load, and preprocess the Amazon Fine Food Reviews dataset from Kaggle.
"""
__author__  = "Vincent Phan"
__email__   = "vdp21005@utdallas.edu"

import kagglehub
import pandas as pd
import os
from typing import Tuple

def load_sentiment_tweets() -> Tuple[pd.Series, pd.Series]:
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
    dataset_identifier = "kazanova/sentiment140"
    csv_file_name = 'training.1600000.processed.noemoticon.csv'

    print(f"Downloading dataset '{dataset_identifier}' from Kaggle...")
    try:
        download_path = kagglehub.dataset_download(dataset_identifier)
        print(f"Dataset downloaded to: {download_path}")

        # Construct the full path to the CSV file
        csv_file_path = os.path.join(download_path, csv_file_name)

        # In case the file name is slightly different or there are multiple files
        if not os.path.exists(csv_file_path):
            print(f"Warning: Expected file '{csv_file_name}' not found directly.")
            csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError(f"No CSV file found in the downloaded directory: {download_path}")
            # Assuming the largest CSV is the main data file if the expected name isn't found
            csv_files.sort(key=lambda x: os.path.getsize(os.path.join(download_path, x)), reverse=True)
            csv_file_path = os.path.join(download_path, csv_files[0])
            print(f"Using largest CSV file found: '{csv_files[0]}'")

        print(f"Loading data from {csv_file_path}...")
        column_names = ['polarity', 'id', 'date', 'query', 'user', 'text']
        df = pd.read_csv(csv_file_path, encoding='latin-1', header=None)
        df.columns = column_names

        # Drop rows with missing values in relevant columns ('polarity', 'text')
        df.dropna(subset=['polarity', 'text'], inplace=True)

        print(f"Original dataset shape: {df.shape}")

        # Filter out neutral tweets (polarity == 2)
        df_filtered = df[df['polarity'] != 2].copy()
        print(f"Shape after filtering out neutral tweets (polarity=2): {df_filtered.shape}")

        # Map polarity to binary sentiment: 0 -> 0 (negative); 4 -> 1 (positive)
        df_filtered['polarity'] = pd.to_numeric(df_filtered['polarity'], errors='coerce')
        df_filtered.dropna(subset=['polarity'], inplace=True)

        # Map 4 to 1, and 0 remains 0 after filtering out 2
        df_filtered['Sentiment'] = df_filtered['polarity'].apply(lambda p: 1 if p == 4 else 0)

        # Extract text and sentiment labels
        tweets = df_filtered['text']
        labels = df_filtered['Sentiment']

        print("Data processing complete. Returning tweet text and binary sentiment labels.")

        return tweets, labels

    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "Please ensure you have the 'kagglehub' and 'pandas' libraries installed (`pip install kagglehub pandas`).")
        print("Also ensure you have the necessary Kaggle credentials configured if required for download.")
        raise  # Re-raise the exception after printing the message


if __name__ == '__main__':
    # You can run this script directly to test the function
    print("--- Testing load_sentiment_tweets function ---")
    try:
        tweet_texts, sentiment_labels = load_sentiment_tweets()

        print("\nSuccessfully loaded data.")
        print(f"Number of tweets: {len(tweet_texts)}")
        print(f"Number of labels: {len(sentiment_labels)}")
        print("\nFirst 5 tweets:")
        # Use zip to print text and corresponding label side-by-side
        for i, (text, label) in enumerate(zip(tweet_texts.head(), sentiment_labels.head())):
            print(f"Tweet {i + 1} [Label: {label}]: {text[:100]}...")  # Print first 100 chars

        print("\nLabel distribution:")
        print(sentiment_labels.value_counts())

    except Exception as e:
        print(f"\nTest failed due to an error: {e}")

    print("--- Test complete ---")