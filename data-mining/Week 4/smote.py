import typer
import pandas as pd
import numpy as np
import xgboost as xgb
import random
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def smote(x, y, target_class, N=100, k=5):
    X_minority = x[y == target_class].values
    n_minority = len(X_minority)
    N = int((N / 100) * n_minority)
    
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_minority)
    synthetic_samples = []
    
    for _ in range(N):
        idx = np.random.randint(0, n_minority)
        sample = X_minority[idx]
        neighbors = nbrs.kneighbors([sample], return_distance=False)[0]
        neighbor_idx = np.random.choice(neighbors[1:])
        neighbor = X_minority[neighbor_idx]
        diff = neighbor - sample
        gap = np.random.rand()
        synthetic_sample = sample + gap * diff
        synthetic_samples.append(synthetic_sample)
    
    synthetic_samples = np.array(synthetic_samples)
    X_new = np.vstack((x.values, synthetic_samples))
    y_new = np.hstack((y.values, [target_class] * N))
    
    return X_new, y_new

def undersample(x, y, majority_class, target_size):
    X_majority = x[y == majority_class].values
    y_majority = y[y == majority_class].values
    
    indices = np.random.choice(len(X_majority), target_size, replace=False)
    X_majority_new = X_majority[indices]
    y_majority_new = y_majority[indices]
    
    return X_majority_new, y_majority_new

def main(file_path: str):
    df = pd.read_csv(file_path)
    file_name = file_path.split('/')[-1].split('.')[0]
    
    x = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    numerical_columns = df.select_dtypes(include=["int64", "float64"])
    x_num = numerical_columns.drop(columns=['is_fraud'])
    y_num = numerical_columns['is_fraud']
    
    class_counts = y_num.value_counts()
    target_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    X_oversampled, y_oversampled = smote(x_num, y_num, target_class, N=200, k=5)
    target_size = sum(y_oversampled == target_class)
    X_undersampled, y_undersampled = undersample(pd.DataFrame(X_oversampled, columns=x_num.columns), 
                                                  pd.Series(y_oversampled), majority_class, target_size)
    X_final = np.vstack((X_undersampled, X_oversampled[y_oversampled == target_class]))
    y_final = np.hstack((y_undersampled, y_oversampled[y_oversampled == target_class]))
    
    df_balanced = pd.DataFrame(X_final, columns=x_num.columns)
    df_balanced['is_fraud'] = y_final
    output_file = f"{file_name}_smote.csv"
    df_balanced.to_csv(output_file, index=False)
    print(f"Data setelah SMOTE disimpan sebagai: {output_file}")
    
if __name__ == "__main__":
    typer.run(main)
