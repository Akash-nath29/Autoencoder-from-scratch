# Autoencoder from Scratch

This project implements an autoencoder from scratch using NumPy. The autoencoder is trained on the MNIST dataset to compress and reconstruct images.

## Project Structure

```
autoencoder_from_scratch/
│
├── data/
│   ├── mnist_train.csv
│   ├── mnist_test.csv
│
├── functions/
│   ├── activations.py
│   ├── loss_functions.py
│   ├── utils.py
│
├── model/
│   ├── autoencoder.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── optimizer.py
│
├── train.py
├── dataloader.py
├── model.ipynb
├── README.md
```

## Hyperparameters

The hyperparameters for the model can be adjusted in `train.py`:
```python
input_size = 784
hidden_size = 256
output_size = 784
learning_rate = 0.001
batch_size = 64
epochs = 300
```

## Model Performance

### Loss Over Epochs

The loss over epochs shows how the model's loss decreases as training progresses. This indicates that the model is learning to reconstruct the input images more accurately over time.

![loss](https://github.com/user-attachments/assets/75b85b89-55e6-4524-8a1c-527038bd651c)

### Accuracy Over Epochs

The accuracy over epochs shows how the model's accuracy improves as training progresses. This indicates that the model is becoming better at reconstructing the input images.

![accuracy](https://github.com/user-attachments/assets/e04cc88a-2d16-46b3-9ebc-e812a5db8d30)

### Output

The output section shows examples of the original, encoded, and reconstructed images. This demonstrates the effectiveness of the autoencoder in compressing and reconstructing the input images.

![output](https://github.com/user-attachments/assets/fbad69a4-5d22-460b-9d32-45c81022dfbd)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/autoencoder_from_scratch.git
    cd autoencoder_from_scratch
    ```

2. Install the required packages:
    ```sh
    pip install numpy pandas matplotlib
    ```

3. Ensure the MNIST dataset CSV files are in the `data/` directory.

## Usage

### Training the Autoencoder

To train the autoencoder, run the `train.py` script:
```sh
python train.py
```

### Jupyter Notebook

You can also explore the project using the provided Jupyter Notebook `model.ipynb`.

### Evaluating the Model

After training, the model can be evaluated on the test dataset. The evaluation script is included in `train.py` and `model.ipynb`.

### Visualizing Results

The notebook and script include code to visualize the original, encoded, and reconstructed images.

## Saving the Model

The trained model weights are saved to `model_weights.json` after training.

## License

This project is licensed under the MIT License.
