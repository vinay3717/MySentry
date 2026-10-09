import os
import numpy as np
import pandas as pd

def generate_clean_dataset(n_samples: int = 200, n_features: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a clean synthetic binary classification dataset.
    Features: Feature_1 .. Feature_N
    Label: 0 or 1 based on linear boundary with noise margin.
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    # Decision boundary primarily guided by first two features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    columns = [f'Feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['Label'] = y
    return df

def poison_dataset(df: pd.DataFrame, poison_type: str = 'label_flip', contamination: float = 0.20, random_state: int = 42) -> pd.DataFrame:
    """
    Inject synthetic data poisoning attacks into the dataset.
    Supported types:
    - 'label_flip': Invert labels for random samples (P0 attack)
    - 'outlier': Inject extreme feature values (P1 attack)
    - 'backdoor': Inject backdoor trigger pattern into features forced to class 1 (P1 attack)
    """
    np.random.seed(random_state)
    df_poison = df.copy()
    n_poison = int(len(df) * contamination)
    idx = np.random.choice(len(df), n_poison, replace=False)
    feature_cols = [c for c in df.columns if c.startswith('Feature_')]

    if poison_type == 'label_flip':
        df_poison.loc[idx, 'Label'] = 1 - df_poison.loc[idx, 'Label']
    elif poison_type == 'outlier':
        # Extreme feature deviation away from normal distribution
        df_poison.loc[idx, feature_cols] += np.random.choice([-1, 1], size=(n_poison, len(feature_cols))) * np.random.uniform(6.0, 10.0, size=(n_poison, len(feature_cols)))
    elif poison_type == 'backdoor':
        # Backdoor trigger pattern on first two features tied to target class 1
        df_poison.loc[idx, feature_cols[:2]] = 5.0
        df_poison.loc[idx, 'Label'] = 1
    else:
        raise ValueError(f"Unsupported poison_type: {poison_type}")

    return df_poison

def generate_and_save_datasets(data_dir: str = None):
    """
    Generate and save clean_dataset.csv and poisoned_dataset.csv.
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(data_dir, exist_ok=True)

    clean_df = generate_clean_dataset(n_samples=200, n_features=5, random_state=42)
    poisoned_df = poison_dataset(clean_df, poison_type='label_flip', contamination=0.20, random_state=42)

    clean_path = os.path.join(data_dir, 'clean_dataset.csv')
    poisoned_path = os.path.join(data_dir, 'poisoned_dataset.csv')

    clean_df.to_csv(clean_path, index=False)
    poisoned_df.to_csv(poisoned_path, index=False)

    print(f"Saved clean dataset ({len(clean_df)} rows) to: {clean_path}")
    print(f"Saved poisoned dataset ({len(poisoned_df)} rows) to: {poisoned_path}")

if __name__ == '__main__':
    generate_and_save_datasets()
