import pandas as pd
import typer
import re
from datetime import datetime
from sklearn.impute import KNNImputer

def clean_text(value):
    if isinstance(value, str):
        return re.sub(r'[^a-zA-Z0-9\s]', '', value).strip()
    return value

def standardize_date(date):
    if pd.isna(date) or not isinstance(date, str):
        return None
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%b %d %Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def preprocess_data(file_path: str):
    df = pd.read_csv(file_path)
    
    # Trim and clean text columns
    df = df.applymap(clean_text)
    
    # Standardizing dates
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = df[col].apply(standardize_date)
    
    # Handling missing values
    df.fillna({col: "Unknown" for col in df.select_dtypes(include='object').columns}, inplace=True)
    df.fillna({col: 0 for col in df.select_dtypes(include=['int64', 'float64']).columns}, inplace=True)
    
    # Converting negative ages to NaN
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df.loc[df['Age'] < 0, 'Age'] = None
    
    # Ensure Quantity is an integer
    if 'Quantity' in df.columns:
        knn_imputer = KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean")
        df['Quantity'] = knn_imputer.fit_transform(df["Quantity"])
    
    output_path = file_path.replace(".csv", "_cleaned.csv")
    df.to_csv(output_path, index=False)
    print(f"Preprocessed file saved as: {output_path}")

if __name__ == "__main__":
    typer.run(preprocess_data)
