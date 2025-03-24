import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, label_encoders=None, preprocessor=None, is_new_data=False):
    df = df.dropna().drop_duplicates()
    if label_encoders is None:
        label_encoders = {}
        for col in ["disease", "gender", "severity", "drug"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    else:
        for col in ["disease", "gender", "severity", "drug"]:
            df[col] = label_encoders[col].transform(df[col])
    
    X = df.drop(columns=["drug"])
    y = df["drug"]
    
    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), ["age"]), ('cat', "passthrough", X.columns.difference(["age"]))]
        )
        X = preprocessor.fit_transform(X)
    else:
        X = preprocessor.transform(X)
    
    return X, y, label_encoders, preprocessor