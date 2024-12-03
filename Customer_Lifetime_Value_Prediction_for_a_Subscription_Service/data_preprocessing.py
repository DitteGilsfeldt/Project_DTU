''' 
Purpose: Define functions to clean and prepare the data for modeling.

Content:
Handling Missing Values: Functions to fill or drop missing values.
Feature Engineering: Define functions to create new features, such as purchase_frequency or engagement_score.
Scaling / Normalization: Functions for scaling or normalizing numerical features if needed.
Data Transformation: Any additional data transformation steps (e.g., one-hot encoding for categorical variables).
'''

import pandas as pd
import numpy as np

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    file_path = 'Project_DTU/Customer_Lifetime_Value_Prediction_for_a_Subscription_Service/online_retail_II.xlsx'
    return pd.read_csv(file_path)

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    print(f"Loading file from: {file_path}")
    return pd.read_csv(file_path)


def clean_data(df):
    """Performs data cleaning such as removing cancellations and handling missing values."""
    # Remove rows where InvoiceNo starts with 'C' to exclude cancellations
    df = df[~df['InvoiceNo'].str.startswith('C')]
    # Drop rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    return df

def create_features(df):
    """Creates features for CLV analysis such as Recency, Frequency, and Monetary Value (RFM)."""
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    # Calculate monetary value for each transaction
    df['MonetaryValue'] = df['Quantity'] * df['UnitPrice']
    
    # Aggregating RFM by CustomerID
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'count',
        'MonetaryValue': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'MonetaryValue']
    
    # Filtering out customers with only one purchase if necessary
    rfm = rfm[rfm['Frequency'] > 1]
    
    return rfm
