import numpy as np
from model.autoencoder import Autoencoder
import json
from model.optimizer import AdamOptimizer
from dataloader import load_data
import matplotlib.pyplot as plt

def load_model(file_path):
    with open(file_path, 'r') as file:
        model = json.load(file)
        
    input_size = model['input_size']
    hidden_size = model['hidden_size']
    optimizer = AdamOptimizer(lr=model['learning_rate'])
    
    autoencoder = Autoencoder(input_size, hidden_size, optimizer)
    autoencoder.encoder.W1 = np.array(model['encoder']['W1'])
    autoencoder.encoder.b1 = np.array(model['encoder']['b1'])
    autoencoder.decoder.W2 = np.array(model['decoder']['W2'])
    autoencoder.decoder.b2 = np.array(model['decoder']['b2'])
    
    return autoencoder

autoencoder = load_model("model_weights.json")
# print(autoencoder.encoder.W1)
# print(autoencoder.encoder.b1)
# print(autoencoder.decoder.W2)
# print(autoencoder.decoder.b2)


_, test_data = load_data('data/mnist_train.csv', 'data/mnist_test.csv')


test_loss = autoencoder.evaluate(test_data, test_data)
print(f'Test Loss: {test_loss}')

n = 10 
plt.figure(figsize=(20, 6))

for i in range(n):
    # Original Image
    ax = plt.subplot(3, n, i + 1)
    plt.title("Original")
    plt.imshow(test_data[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Encoded Image
    encoded_data = autoencoder.encoder.forward(test_data)
    ax = plt.subplot(3, n, i + 1 + n) 
    plt.title("Encoded")
    plt.imshow(encoded_data[i].reshape(16, 16))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed Image
    reconstructed_data = autoencoder.predict(test_data)
    ax = plt.subplot(3, n, i + 1 + 2 * n) 
    plt.title("Reconstructed")
    plt.imshow(reconstructed_data[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()
