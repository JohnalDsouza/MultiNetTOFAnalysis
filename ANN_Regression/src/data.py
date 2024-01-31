from sklearn.model_selection import train_test_split
import pandas as pd

def train_val_data(train_path = "path to train", label_path = "path to train label"):
    """Load train and valid data"""

    data = pd.read_excel(train_path)
    
    X_train = data.iloc[:, :-1]
    X_train = X_train.astype(float)

    y_train = pd.read_excel(label_path)
    y_train = y_train.astype(float)

    


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    # Print the shapes of the datasets
    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("X_val shape:", X_val.shape)
    # print("y_val shape:", y_val.shape)

    return X_train, X_val, y_train, y_val

def test_data(data_path = "path to test data", label_path = "path to test label"):
    """ Load test data"""

    data1 = pd.read_excel(data_path)
    X_test = data1.iloc[:, :-1]
    X_test = X_test.astype(float)

    y_test = pd.read_excel(label_path)

    return X_test, y_test

