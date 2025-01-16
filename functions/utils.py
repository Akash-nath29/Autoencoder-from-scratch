import numpy as np

def init_params(input_size, output_size):
    W1 = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((output_size, 1))
    return W1, b1

def dense(x, w, b):
    # if len(x.shape) > 1:
    #     b = np.repeat(b, x.shape[0], axis=1).T
    return np.dot(x, w) + b

def calculate_accuracy(y_true, y_pred, threshold=0.1):
    diff = np.abs(y_true - y_pred)
    correct_features = (diff < threshold).astype(int)
    sample_accuracy = np.mean(correct_features, axis=1)
    overall_accuracy = np.mean(sample_accuracy) * 100
    return overall_accuracy