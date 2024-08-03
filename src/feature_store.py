import pandas as pd
import joblib

def save_features(X, y, feature_names, path='../feature_store/'):
    pd.DataFrame(X, columns=feature_names).to_parquet(path + 'features.parquet')
    pd.DataFrame(y, columns=['charges']).to_parquet(path + 'target.parquet')
    joblib.dump(feature_names, path + 'feature_names.joblib')

def load_features(path='../feature_store/'):
    X = pd.read_parquet(path + 'features.parquet')
    y = pd.read_parquet(path + 'target.parquet').squeeze()
    
    return X, y
