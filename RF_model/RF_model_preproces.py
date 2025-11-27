import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

RANDOM = 123

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_eda(df):
    print("Shape:", df.shape)
    print("\nTarget distribution:\n", df['TenYearCHD'].value_counts(dropna=False))
    print("\nMissing values per column:\n", df.isna().sum())

def prepare_xy(df, target='TenYearCHD'):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def simple_impute(X_train, X_test):

    # imputacja medianą na liczbach

    imputer = SimpleImputer(strategy='median')

    # Dopasowuje tylko na train

    imputer.fit(X_train)
    X_train_imp = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_imp, X_test_imp, imputer

def train_test_split_stratified(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM)
