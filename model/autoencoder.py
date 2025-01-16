import numpy as np
from functions.utils import dense, init_params, calculate_accuracy
from functions.loss_functions import bce_loss
from model.encoder import Encoder
from model.decoder import Decoder
import matplotlib.pyplot as plt
import json

class Autoencoder:
    def __init__(self, input_size, hidden_size, optimizer):
        self.encoder = Encoder(input_size, hidden_size, optimizer)
        self.decoder = Decoder(hidden_size, input_size, optimizer)
        self.optimizer = optimizer
        self.loss_history = []
        self.total_params = self.calculate_total_params()
        
    def calculate_total_params(self):
        encoder_params = np.prod(self.encoder.W1.shape) + np.prod(self.encoder.b1.shape)
        decoder_params = np.prod(self.decoder.W2.shape) + np.prod(self.decoder.b2.shape)
        return encoder_params + decoder_params
    
    def summary(self):
        # Header
        print("------------------------------------------------------------")
        print(f"{'Layer (Type)':<20} {'Output Shape':<20} {'Param #':<10}")
        print("============================================================")
        
        # Encoder details
        encoder_params = np.prod(self.encoder.W1.shape) + np.prod(self.encoder.b1.shape)
        print(f"Encoder (Dense):{'':<9} ({self.encoder.hidden_size},)       {encoder_params:<10}")
        
        # Decoder details
        decoder_params = np.prod(self.decoder.W2.shape) + np.prod(self.decoder.b2.shape)
        print(f"Decoder (Dense):{'':<9} ({self.decoder.output_size},)      {decoder_params:<10}")
        
        # Footer
        print("============================================================")
        print(f"Total Parameters: {self.total_params}")
        print(f"Trainable Parameters: {self.total_params}")
        print(f"Non-trainable Parameters: 0")
        print("------------------------------------------------------------")
        
    def forward(self, x):
        self.encoded = self.encoder.forward(x)
        self.decoded = self.decoder.forward(self.encoded)
        return self.decoded
    
    def backward(self, y):
        self.decoder.backward(y)
        self.decoder.update()
        self.encoder.backward(y)
        self.encoder.update()
        
    def train(self, x, y, epochs, batch_size, threshold=0.1):
        m = x.shape[0]
        self.accuracy_history = []

        for epoch in range(epochs):
            epoch_loss = 0
            all_predictions = []
            all_true_values = []

            for i in range(0, m, batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                
                self.forward(x_batch)
                self.backward(y_batch)

                batch_loss = bce_loss(y_batch, self.decoded)
                epoch_loss += batch_loss

                all_predictions.append(self.decoded)
                all_true_values.append(y_batch)

            avg_epoch_loss = epoch_loss / (m // batch_size)
            self.loss_history.append(avg_epoch_loss)

            all_predictions = np.vstack(all_predictions)
            all_true_values = np.vstack(all_true_values)

            accuracy = calculate_accuracy(all_true_values, all_predictions, threshold)
            self.accuracy_history.append(accuracy)

            print("-----------------------------------------------")
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy}%')

        self.plot_metrics()

            
    def predict(self, x):
        return self.forward(x)
    
    def evaluate(self, x, y):
        predictions = self.predict(x)
        loss = bce_loss(y, predictions)
        return loss
    
    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label="Training Loss", color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(self.accuracy_history, label="Training Accuracy", color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy Over Epochs')
        plt.legend()
        plt.grid()
        plt.show()
        
    def save(self, path):
        model = {
            "encoder": {
                "W1": self.encoder.W1.tolist(),
                "b1": self.encoder.b1.tolist()
            },
            "decoder": {
                "W2": self.decoder.W2.tolist(),
                "b2": self.decoder.b2.tolist()
            },
            "learning_rate": float(self.optimizer.lr),
            "total_params": int(self.total_params),
            "input_size": int(self.encoder.input_size),
            "hidden_size": int(self.encoder.hidden_size),
            "output_size": int(self.decoder.output_size)
        }
        
        with open(path, 'w') as f:
            json.dump(model, f)