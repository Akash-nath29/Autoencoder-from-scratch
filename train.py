from model.autoencoder import Autoencoder
from model.optimizer import AdamOptimizer
from dataloader import load_data

# Hyperparameters

input_size = 784
hidden_size = 256
output_size = 784
learning_rate = 0.001
batch_size = 64
epochs = 300

train_data, test_data = load_data('data/mnist_train.csv', 'data/mnist_test.csv')

optimizer = AdamOptimizer(lr=learning_rate)            
autoencoder = Autoencoder(784, 256, optimizer)
autoencoder.summary()
autoencoder.train(train_data, train_data, epochs, batch_size)

test_loss = autoencoder.evaluate(test_data, test_data)
print(f"Test Loss: {test_loss}")

autoencoder.save("model_weights.json")