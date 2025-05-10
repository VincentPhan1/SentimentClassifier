# Models/sentiment_trainer_function.py

import pandas as pd
import torch
from torch.utils.data import Dataset # Used implicitly by Hugging Face Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PreTrainedModel # Type hinting for return
)
from datasets import Dataset as HFDataset # Import Hugging Face's Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import os
import sys

# Add the project root to the system path to import modules
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)


# Tokenization Function for map()
def tokenize_function(examples, tokenizer, max_len):
    """Applies tokenizer to examples in a batch."""
    return tokenizer(
        examples["text"],
        padding="max_length", # Pad to max_length
        truncation=True,      # Truncate if longer than max_length
        max_length=max_len,
    )

# Define Metrics Calculation
def compute_metrics(eval_pred):
    """Computes classification metrics from predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Use 'binary' average for 2-class problems
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# --- Training Function ---

def train_amazon_sentiment_transformer(
    X_train_raw: pd.Series,
    X_val_raw: pd.Series,
    X_test_raw: pd.Series,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    model_name: str = 'bert-base-uncased',
    output_dir: str = './results_amazon_sentiment',
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    max_len: int = 128,
):

    """
    Trains a Hugging Face Transformer for sentiment analysis on pre-split data
    and returns the trained model.

    Args:
        X_train_raw (pd.Series): Pandas Series containing raw training text reviews.
        X_val_raw (pd.Series): Pandas Series containing raw validation text reviews.
        X_test_raw (pd.Series): Pandas Series containing raw testing text reviews.
        y_train (pd.Series): Pandas Series containing training labels.
        y_val (pd.Series): Pandas Series containing validation labels.
        y_test (pd.Series): Pandas Series containing testing labels.
        model_name (str): Name of the pre-trained model from Hugging Face Hub.
        output_dir (str): Directory to save training checkpoints and results.
        num_train_epochs (int): Total number of training epochs.
        per_device_train_batch_size (int): Batch size per GPU/CPU for training.
        per_device_eval_batch_size (int): Batch size per GPU/CPU for evaluation.
        max_len (int): Maximum sequence length for tokenizer padding/truncation.
        # Add more args matching TrainingArguments parameters

    Returns:
        PreTrainedModel: The trained Hugging Face Transformer model.
    """
    print(f"\n--- Starting Transformer Training Process ---")
    print(f"Model: {model_name}, Output Dir: {output_dir}, Epochs: {num_train_epochs}")
    print(f"Training data size: {len(X_train_raw)}")
    print(f"Validation data size: {len(X_val_raw)}")
    print(f"Testing data size: {len(X_test_raw)}")


    # Convert pandas Series to Hugging Face Dataset format
    print("\nCreating Hugging Face Datasets from splits...")
    train_dict = {'text': X_train_raw.tolist(), 'label': y_train.tolist()}
    val_dict = {'text': X_val_raw.tolist(), 'label': y_val.tolist()}
    test_dict = {'text': X_test_raw.tolist(), 'label': y_test.tolist()} # Keep test set as HF dataset too

    train_dataset = HFDataset.from_dict(train_dict)
    val_dataset = HFDataset.from_dict(val_dict)
    test_dataset = HFDataset.from_dict(test_dict)

    print("Hugging Face Datasets created.")


    # Initialize Tokenizer
    print(f"\nLoading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # Apply tokenization to datasets using map()
    print("Tokenizing datasets...")
    # Use lambda to pass tokenizer and max_len to tokenize_function
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_len),
        batched=True,
        remove_columns=["text"] # Remove original text column
    )
    tokenized_val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_len),
        batched=True,
        remove_columns=["text"] # Remove original text column
    )
    tokenized_test_dataset = test_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_len),
        batched=True,
        remove_columns=["text"] # Remove original text column
    )
    print("Tokenization complete.")


    # --- Model Loading ---
    print(f"\nLoading pre-trained model: {model_name} for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print("Model loaded.")

    # --- Training Arguments ---
    print("\nDefining Training Arguments...")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,          # Output directory for checkpoints and results
        eval_strategy="epoch",          # Evaluate every epoch
        save_strategy="epoch",          # Save checkpoint every epoch
        save_total_limit=2,             # Limit the number of checkpoints to keep (e.g., keep last 2)
        num_train_epochs=num_train_epochs, # Total number of training epochs
        per_device_train_batch_size=per_device_train_batch_size, # Batch size per GPU/CPU for training
        per_device_eval_batch_size=per_device_eval_batch_size,  # Batch size per GPU/CPU for evaluation
        # Add other common arguments or make them function parameters
        learning_rate=2e-5,             # Default learning rate
        warmup_steps=500,               # Default warmup steps
        weight_decay=0.01,              # Default weight decay
        logging_dir=f'{output_dir}/logs',# Directory for storing logs
        logging_steps=10,               # Log every N update steps
        fp16=torch.cuda.is_available(), # Enable mixed precision training if CUDA is available
        load_best_model_at_end=True,    # Load the best model found during training at the end
        metric_for_best_model="accuracy", # Metric to monitor for best model saving
        greater_is_better=True,         # Whether the metric is better when greater
        report_to="none"                # Prevent reporting to external services by default
    )
    print("Training Arguments defined.")

    # --- Initialize the Trainer ---
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print("Trainer initialized.")

    # --- Train the Model ---
    print("\n--- Starting Model Training ---")
    trainer.train()
    print("--- Training Complete ---")

    # --- Final Evaluation on Test Set ---
    print("\n--- Performing Final Evaluation on Test Set ---")
    # Evaluate on the test set that was held out
    test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
    print(f"Transformer Test Results: {test_results}")
    print("--- Test Evaluation Complete ---")


    # Return the trained model (loaded with best weights if load_best_model_at_end=True)
    print("\nReturning the trained Transformer model.")
    return trainer.model

# Example Usage (kept for testing this file individually if needed)
if __name__ == '__main__':
    print("--- Testing train_amazon_sentiment_transformer function (requires data loading) ---")
    try:
        # This example will load data internally for testing purposes
        # In the main comparison script, data is loaded and split ONCE.
        print("Loading data for standalone test...")
        # Assuming amazon_dataset.py is accessible
        from Datasets.amazon_dataset import load_amazon_reviews
        review_texts, sentiment_labels = load_amazon_reviews()
        print("Data loaded.")

        # Perform a simple split for this standalone test
        print("Performing standalone train/val/test split...")
        X_train_val_test, X_test_raw_test, y_train_val_test, y_test_test = train_test_split(
            review_texts, sentiment_labels, test_size=0.15, random_state=42, stratify=sentiment_labels
        )
        X_train_raw_test, X_val_raw_test, y_train_test, y_val_test = train_test_split(
            X_train_val_test, y_train_val_test, test_size=0.1765, random_state=42, stratify=y_train_val_test
        )
        print(f"Standalone train size: {len(X_train_raw_test)}, val size: {len(X_val_raw_test)}, test size: {len(X_test_raw_test)}")

        # Define parameters for the standalone test run
        standalone_params = {
            'X_train_raw': X_train_raw_test,
            'X_val_raw': X_val_raw_test,
            'X_test_raw': X_test_raw_test,
            'y_train': y_train_test,
            'y_val': y_val_test,
            'y_test': y_test_test,
            'model_name': 'distilbert-base-uncased',
            'output_dir': './standalone_transformer_test_results',
            'num_train_epochs': 1,
            'per_device_train_batch_size': 32,
            'per_device_eval_batch_size': 32,
            'max_len': 128,
        }

        # Run the training and evaluation using the split data
        trained_transformer_model = train_amazon_sentiment_transformer(**standalone_params)

        print("\nStandalone test complete. Model object returned.")

    except Exception as e:
        print(f"\nStandalone test failed due to an error: {e}")
        print("Please ensure 'kagglehub', 'pandas', 'scikit-learn', 'torch', 'transformers', 'datasets' are installed.")
        print("Also, ensure load_amazon_reviews() is accessible.")

    print("--- Standalone Test Complete ---")
