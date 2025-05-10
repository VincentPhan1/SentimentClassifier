# main_comparison.py

import pandas as pd
from sklearn.model_selection import train_test_split # Needed for splitting data ONCE
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Needed for Naive Bayes evaluation

import os
import sys

# Adjust the system path to import modules from the project root and Models directory
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir)) # Assuming this script is in the root
models_dir = os.path.join(project_root, 'Models')

sys.path.insert(0, project_root)
sys.path.insert(0, models_dir)

try:
    # Import data loading
    from Datasets.amazon_dataset import load_amazon_reviews

    # Import model training functions (now accept splits)
    from Models.naive_bayes import train_and_evaluate_naive_bayes # Naive Bayes function
    from Models.transformer import train_amazon_sentiment_transformer # Transformer function

except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure 'amazon_dataset.py' is in the project root.")
    print("Please ensure 'Models' directory contains 'sentiment_classifier.py' and 'sentiment_trainer_function.py'.")
    sys.exit("Required modules not found. Please check file paths and names.")


def run_sentiment_comparison(
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    nb_max_features: int = 10000,
    transformer_model_name: str = 'distilbert-base-uncased',
    transformer_epochs: int = 1,
    transformer_batch_size: int = 16,
    transformer_max_len: int = 128,
    transformer_output_dir: str = './amazon_sentiment_transformer_results',
):
    """
    Runs a comparison between Multinomial Naive Bayes (with TF-IDF)
    and a Hugging Face Transformer model for sentiment analysis
    on the Amazon Fine Food Reviews dataset, using the exact same data splits.

    Args:
        test_size (float): Proportion of data for the final test set.
        val_size (float): Proportion of data for the validation set (from train_val split).
        random_state (int): Seed for random splits for reproducibility.
        nb_max_features (int): Max features for TF-IDF in Naive Bayes.
        transformer_model_name (str): Hugging Face model name for Transformer.
        transformer_epochs (int): Number of epochs for Transformer training.
        transformer_batch_size (int): Batch size for Transformer training/evaluation.
        transformer_max_len (int): Max sequence length for Transformer tokenizer.
        transformer_output_dir (str): Output directory for Transformer results.
    """
    print("--- Starting Sentiment Model Comparison ---")

    # --- 1. Load the Dataset ---
    print("\nLoading dataset...")
    try:
        review_texts, sentiment_labels = load_amazon_reviews()
        print("Dataset loaded successfully.")
        print(f"Total samples: {len(review_texts)}")
        print(f"Label distribution:\n{sentiment_labels.value_counts()}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Cannot proceed with model comparison.")
        return # Exit the function if data loading fails

    # Ensure consistent indexing before splitting
    review_texts = review_texts.reset_index(drop=True)
    sentiment_labels = sentiment_labels.reset_index(drop=True)

    # --- 2. Perform Data Split ONCE ---
    print(f"\nPerforming data split (test_size={test_size}, val_size={val_size})...")
    # Split into train_val and test
    X_train_val_raw, X_test_raw, y_train_val, y_test = train_test_split(
        review_texts, sentiment_labels, test_size=test_size, random_state=random_state, stratify=sentiment_labels
    )

    # Split train_val into train and validation
    # Calculate val_size relative to the train_val split size
    val_size_relative = val_size / (1 - test_size)
    if val_size_relative >= 1:
         print(f"Error: Validation size ({val_size}) is too large relative to test size ({test_size}).")
         print("Validation size must be less than (1 - test_size).")
         return # Exit if split is invalid

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val_raw, y_train_val, test_size=val_size_relative, random_state=random_state, stratify=y_train_val
    )

    print(f"Train set size: {len(X_train_raw)}")
    print(f"Validation set size: {len(X_val_raw)}")
    print(f"Test set size: {len(X_test_raw)}")
    print("Data split complete.")


    # --- 3. Train and Evaluate Multinomial Naive Bayes (with TF-IDF) ---
    print("\n--- Running Multinomial Naive Bayes Model ---")
    try:
        # Pass the pre-split raw data to the Naive Bayes trainer
        naive_bayes_model = train_and_evaluate_naive_bayes(
            X_train_raw, X_test_raw, y_train, y_test, max_features=nb_max_features
        )
        print("Multinomial Naive Bayes process complete.")

        # The evaluation metrics are printed by the function itself.

    except Exception as e:
        print(f"An error occurred during Naive Bayes process: {e}")
        naive_bayes_model = None # Indicate failure


    # --- 4. Train and Evaluate Hugging Face Transformer ---
    print("\n--- Running Hugging Face Transformer Model ---")
    try:
        # Pass the pre-split raw data to the Transformer trainer
        transformer_params = {
            'X_train_raw': X_train_raw,
            'X_val_raw': X_val_raw,
            'X_test_raw': X_test_raw,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'model_name': transformer_model_name,
            'output_dir': transformer_output_dir,
            'num_train_epochs': transformer_epochs,
            'per_device_train_batch_size': transformer_batch_size,
            'per_device_eval_batch_size': transformer_batch_size, # Using same batch size for eval
            'max_len': transformer_max_len,
            # Add more parameters if needed
        }
        print(f"Using Transformer parameters: {transformer_params}")

        # The function prints training progress and final test evaluation results.
        transformer_model = train_amazon_sentiment_transformer(**transformer_params)
        print("Hugging Face Transformer process complete.")

    except Exception as e:
        print(f"An error occurred during Transformer process: {e}")
        transformer_model = None # Indicate failure
        # Clean up potentially partially created output directory if needed
        import shutil
        if os.path.exists(transformer_params['output_dir']):
            print(f"Attempting to clean up {transformer_params['output_dir']}")
            try:
                shutil.rmtree(transformer_params['output_dir'])
                print("Cleanup successful.")
            except Exception as cleanup_e:
                print(f"Cleanup failed: {cleanup_e}")


    print("\n--- Sentiment Model Comparison Complete ---")


# --- Main Execution Block ---
if __name__ == '__main__':
    # Define parameters for the comparison run
    comparison_params = {
        'test_size': 0.15,
        'val_size': 0.15,
        'random_state': 42,
        'nb_max_features': 10000, # Max features for Naive Bayes TF-IDF
        'transformer_model_name': 'distilbert-base-uncased', # Use DistilBERT for faster comparison
        'transformer_epochs': 1, # Keep low for initial test
        'transformer_batch_size': 16, # Adjust based on GPU memory
        'transformer_max_len': 128,
        'transformer_output_dir': './amazon_sentiment_comparison_results_distilbert', # Dedicated output dir for this comparison run
    }

    try:
        run_sentiment_comparison(**comparison_params)
    except Exception as e:
        print(f"\nAn unexpected error occurred during the comparison: {e}")

    print("\n--- Main Program Finished ---")
