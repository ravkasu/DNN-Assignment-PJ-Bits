import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_RAW = os.path.join("data", "raw", "heart.csv")
DATA_PROCESSED_X_TRAIN = os.path.join("data", "processed", "X_train.csv")
DATA_PROCESSED_Y_TRAIN = os.path.join("data", "processed", "y_train.csv")
DATA_PROCESSED_X_TEST = os.path.join("data", "processed", "X_test.csv")
DATA_PROCESSED_Y_TEST = os.path.join("data", "processed", "y_test.csv")

def load_raw_dataframe(path=DATA_RAW):
    df = pd.read_csv(path)
    return df

def prepare_and_save_test_train(test_size=0.2, random_state=42):
    df = load_raw_dataframe()
    if "target" not in df.columns:
        raise ValueError("No 'target' column in raw data")
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # save processed datasets
    X_train.to_csv(DATA_PROCESSED_X_TRAIN, index=False)
    X_test.to_csv(DATA_PROCESSED_X_TEST, index=False)
    y_train.to_csv(DATA_PROCESSED_Y_TRAIN, index=False)
    y_test.to_csv(DATA_PROCESSED_Y_TEST, index=False)

    print("Processed data saved to data/processed/")
    return True

if __name__ == "__main__":
    prepare_and_save_test_train()
