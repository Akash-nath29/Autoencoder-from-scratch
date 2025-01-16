import pandas as pd

def load_data(training_data_path: str, test_data_path: str) -> pd.DataFrame:
    train_data = pd.read_csv(training_data_path)
    test_data = pd.read_csv(test_data_path)

    train_data = train_data.drop('label', axis=1)
    test_data = test_data.drop('label', axis=1)

    train_data = train_data.values.reshape(-1, 784)
    test_data = test_data.values.reshape(-1, 784)
    
    train_data = train_data / 255.0
    test_data = test_data / 255.0
    
    return train_data, test_data