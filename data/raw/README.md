# Raw Data Directory

This directory is designated for the raw, unmodified dataset used in the project:
**Credit Card Fraud Detection (ULB)**.

## How to Place the Dataset

1. Download the dataset from Kaggle:
   [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Extract the archive if downloaded as a `.zip` file.
3. Place the extracted CSV file at:
   `data/raw/creditcard.csv`

## Dataset Details
- **Filename**: `creditcard.csv`
- **Total Transactions**: 284,807
- **Features**: 31 columns (`Time`, `V1` through `V28` PCA components, `Amount`, and `Class`)
- **Target**: `Class` (0 = Legitimate transaction, 1 = Fraudulent transaction)

> **Important**: Never edit, overwrite, or commit the raw dataset file to version control.
