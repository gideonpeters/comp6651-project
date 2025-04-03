import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def compute_gower_distance(df: pd.DataFrame) -> np.ndarray:
    """
    Computes the Gower distance matrix for a DataFrame with mixed data types.
    Normalizes numeric columns using min-max normalization and uses 0/1 distance for categorical columns.
    """
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Normalize numeric columns (min-max scaling)
    df_numeric = df[numeric_cols].copy()
    if not df_numeric.empty:
        df_numeric = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())
    
    df_categorical = df[categorical_cols].copy()
    
    n = df.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = 0
            count = 0
            # Numeric distance contribution
            if numeric_cols:
                diff = np.abs(df_numeric.iloc[i].values - df_numeric.iloc[j].values)
                d += np.sum(diff)
                count += len(numeric_cols)
            # Categorical distance contribution
            if categorical_cols:
                diff_cat = (df_categorical.iloc[i] != df_categorical.iloc[j]).astype(int)
                d += np.sum(diff_cat)
                count += len(categorical_cols)
            d_avg = d / count if count > 0 else 0
            D[i, j] = d_avg
            D[j, i] = d_avg
    return D



def compute_gower_distance_variant(x, y, num_ranges, num_indices, cat_indices):
    total_diff = 0
    # Process numerical attributes
    for i in num_indices:
        # Normalize absolute difference by the range
        total_diff += abs(x[i] - y[i]) / num_ranges[i]
    
    # Process categorical attributes
    for j in cat_indices:
        total_diff += 0 if x[j] == y[j] else 1
    
    # Combine and normalize by the total number of attributes
    return total_diff / (len(num_indices) + len(cat_indices))


def compute_gower_distance_matrix(df, num_cols, cat_cols):
    """
    Compute a pairwise Gower distance matrix for the DataFrame `df`
    using the provided numerical and categorical column lists.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataframe (with no missing values).
        num_cols (list): List of numerical column names.
        cat_cols (list): List of categorical column names.
        
    Returns:
        np.ndarray: A symmetric matrix of pairwise dissimilarities.
    """
    # Get the column indices for numerical and categorical columns
    num_indices = [df.columns.get_loc(col) for col in num_cols]
    cat_indices = [df.columns.get_loc(col) for col in cat_cols]
    
    # Compute the range for each numerical column (avoid division by zero)
    num_ranges = {}
    for col in num_cols:
        idx = df.columns.get_loc(col)
        col_range = df[col].max() - df[col].min()
        num_ranges[idx] = col_range if col_range != 0 else 1

    n = len(df)
    dist_matrix = np.zeros((n, n))
    
    # Compute pairwise distances using the provided compute_gower_distance_variant function
    for i in range(n):
        for j in range(i, n):
            d = compute_gower_distance_variant(
                    df.iloc[i].values, 
                    df.iloc[j].values, 
                    num_ranges, num_indices, cat_indices)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d  # matrix is symmetric
    return dist_matrix



def visualize_clusters(X, labels, title, output_dir='./../results/images', saveAs=None):
    """
    Visualize clusters using a scatter plot. If X has more than 2 dimensions,
    PCA is used to reduce it to 2D.
    
    Parameters:
        X (np.ndarray): Feature array.
        labels (np.ndarray): Cluster labels.
        title (str): Title for the plot.
    """
    # Reduce dimensionality if needed.
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
    else:
        X_vis = X
    
    plt.figure()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        # Color noise points (label -1) in black.
        if label == -1:
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1], c='k', marker='x', label='Noise')
        else:
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1], label=f'Cluster {label}')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)

    os.makedirs(output_dir, exist_ok=True)

    if saveAs is None:
        saveAs = title.replace(" ", "_")
    
    # Create a filename based on the title.
    file_name = saveAs + '.png'
    file_path = os.path.join(output_dir, file_name)
    
    plt.savefig(file_path, bbox_inches='tight')

    plt.show()

