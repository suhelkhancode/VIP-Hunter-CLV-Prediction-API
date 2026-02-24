import pandas as pd

def engineer_customer_features(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms cleaned transaction rows into customer-level features."""
    df = cleaned_df.copy()
    
    # 1. Calculate the revenue for each individual transaction row
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    # 2. Group by CustomerID and aggregate the exact features the model needs
    customer_features = df.groupby('CustomerID').agg(
        TotalQty=('Quantity', 'sum'),
        AvgUnitPrice=('UnitPrice', 'mean'),  # Your exact logic: mean of the unit prices
        Monetary_Value=('Revenue', 'sum')
    ).reset_index()
    
    # 3. Ensure the columns are ordered perfectly for the final output
    final_features = customer_features[['CustomerID', 'TotalQty', 'AvgUnitPrice', 'Monetary_Value']]
    
    return final_features