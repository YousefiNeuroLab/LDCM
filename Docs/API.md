# Documentation for Repository

## CNN1D

### Description
A 1D Convolutional Neural Network for classifying time-series data. The architecture consists of convolutional layers followed by max-pooling and fully connected layers, making it suitable for binary classification tasks.

### Class: `CNN1D`

#### `__init__(self, input_channels, input_time_steps)`
Initializes the CNN1D model with specified input dimensions.

- **Parameters:**
  - `input_channels` (*int*): Number of features (input channels).
  - `input_time_steps` (*int*): Number of time steps in the input sequence.
- **Output:**
  - Instantiates the CNN model with convolutional, pooling, and fully connected layers.

#### `forward(self, x)`
Performs the forward pass through the network.

- **Parameters:**
  - `x` (*torch.Tensor*): Input tensor with shape `(batch_size, input_channels, input_time_steps)`.
- **Returns:**
  - *torch.Tensor*: Output tensor with shape `(batch_size, 2)` for binary classification logits.

#### `calculate_fc_input_size(self, input_channels, input_time_steps)`
Calculates the size of the flattened tensor after convolution and pooling layers, which feeds into the fully connected layer.

- **Parameters:**
  - `input_channels` (*int*): Number of input channels.
  - `input_time_steps` (*int*): Number of time steps in the input sequence.
- **Returns:**
  - *int*: Size of the flattened tensor.

---

### Utility Functions

#### `split_data(x, labels, train_size=0.6, val_size=0.2)`
Splits the dataset into training, validation, and test sets, maintaining class balance.

- **Parameters:**
  - `x` (*array-like*): Feature data with samples as rows and features as columns.
  - `labels` (*array-like*): Corresponding labels.
  - `train_size` (*float*): Fraction of data for training.
  - `val_size` (*float*): Fraction of data for validation.
- **Returns:**
  - *tuple*: Splits of the form ((`x_train`, `y_train`), (`x_val`, `y_val`), (`x_test`, `y_test`)).

#### `create_data_loader(x, y, batch_size=2, shuffle=False)`
Wraps features and labels into a DataLoader for batch processing.

- **Parameters:**
  - `x` (*torch.Tensor*): Input features of shape `(num_samples, num_features)`.
  - `y` (*torch.Tensor*): Labels of shape `(num_samples,)`.
  - `batch_size` (*int*): Number of samples per batch.
  - `shuffle` (*bool*): Whether to shuffle data at each epoch.
- **Returns:**
  - *torch.utils.data.DataLoader*: DataLoader object.

---

### Model Training and Evaluation

#### `train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)`
Trains the CNN model using the provided DataLoaders.

- **Parameters:**
  - `model` (*torch.nn.Module*): Model to train.
  - `train_loader` (*DataLoader*): Training data loader.
  - `val_loader` (*DataLoader*): Validation data loader.
  - `criterion` (*loss function*): Loss function for training.
  - `optimizer` (*torch.optim.Optimizer*): Optimizer for updating parameters.
  - `epochs` (*int*): Number of epochs.
- **Output:**
  - Prints training and validation loss, along with validation accuracy.

#### `evaluate_model(model, loader, criterion=None)`
Evaluates the model on a dataset and computes accuracy.

- **Parameters:**
  - `model` (*torch.nn.Module*): Model to evaluate.
  - `loader` (*DataLoader*): DataLoader for the dataset.
  - `criterion` (*loss function, optional*): Loss function for computing the loss.
- **Returns:**
  - *tuple*: If criterion is provided, returns (`loss`, `accuracy`). Otherwise, only `accuracy` is returned.

---

### Example Usage

```python
# Generate synthetic data
x = torch.randn(100, 25, 80)  # (samples, time_steps, dimensions)
x = x.permute(0, 2, 1)
labels = torch.randint(0, 2, (100,))

# Split the data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, labels)

# Create DataLoaders
train_loader = create_data_loader(x_train, y_train, shuffle=True)
val_loader = create_data_loader(x_val, y_val)
test_loader = create_data_loader(x_test, y_test)

# Initialize the CNN model
model = CNN1D(input_channels=80, input_time_steps=25)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# Evaluate on test set
test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")
