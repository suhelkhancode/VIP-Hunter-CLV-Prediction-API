import pandas as pd

def clean_raw_retail_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw online retail dataset and enforces data types."""
    df = raw_df.copy()
    
    # 1. Drop rows missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    
    # 2. Remove Cancelled Orders (InvoiceNo starts with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    # 3. ENFORCE PROPER DATA FORMATS
    # Convert dates to actual Pandas Datetime objects
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # CustomerID imports as a float (e.g., 17850.0) because of empty rows.
    # Now that empties are gone, we convert to int, then to string for a clean ID.
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    
    # Lock in the math columns
    df['Quantity'] = df['Quantity'].astype(int)
    df['UnitPrice'] = df['UnitPrice'].astype(float)
    
    # Lock in the categorical columns
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['Country'] = df['Country'].astype(str)
    
    # 4. Remove zero or negative quantities 
    df = df[df['Quantity'] > 0]
    
    # 5. Remove zero or negative prices
    df = df[df['UnitPrice'] > 0]
    
    return df