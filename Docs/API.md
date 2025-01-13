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

# State Class

## Description
The `State` class encapsulates the parameters and behaviors of a state-space model, supporting dynamic simulations of state and observation vectors. The class provides flexibility for different modes of simulation (e.g., random walk and generalized).

---

## Class: `State`

### `__init__(self, dim_state, dim_obs, mode, R, R0, n_steps)`
Initializes the state-space model with the specified parameters.

- **Parameters:**
  - `dim_state` (*int*): Dimension of the state vector.
  - `dim_obs` (*int*): Dimension of the observation vector.
  - `mode` (*int*): Simulation mode (1 for random walk, 2 for generalized).
  - `R` (*float*): Observation noise level.
  - `R0` (*float*): Initial noise level.
  - `n_steps` (*int*): Number of time steps for the simulation.
- **Output:**
  - An instance of the `State` class, pre-initialized with matrices and parameters for simulations.

---

## Simulation Functions

### `create_simulation_params(self)`
Generates the initial parameters for the simulation, including covariance matrices, eigenvalues, and observation model matrices.

- **Returns:**
  - A tuple containing:
    - `x_init` (*np.ndarray*): Initial state vector with Gaussian noise.
    - `mean` (*np.ndarray*): Observation mean vector.
    - `chol_cov_matrix` (*np.ndarray*): Cholesky decomposition of the covariance matrix.
    - `A`, `B` (*np.ndarray*): Transition matrices for state dynamics.
    - `C`, `D` (*np.ndarray*): Observation model matrices.
    - `eigenvalue`, `eigenvectors` (*np.ndarray*): Eigenvalues and eigenvectors of the covariance matrix.
    - `cov_matrix` (*np.ndarray*): Covariance matrix with low correlations.
    - `w` (*np.ndarray*): Transformed eigenvectors for simulation.
    - `Q` (*np.ndarray*): Process noise covariance matrix.

### `run_simulation(dim_state, dim_obs, mode, R, R0, n_steps, num_trials)`
Runs the simulation for multiple trials, generating state and observation data.

- **Parameters:**
  - `dim_state`, `dim_obs` (*int*): Dimensions of the state and observation vectors.
  - `mode` (*int*): Simulation mode (1 for random walk, 2 for generalized).
  - `R`, `R0` (*float*): Noise levels.
  - `n_steps` (*int*): Number of time steps in each trial.
  - `num_trials` (*int*): Number of trials to simulate.
- **Returns:**
  - `X_all_trials` (*np.ndarray*): State vectors for all trials.
  - `Y_all_trials` (*np.ndarray*): Observation vectors for all trials.
  - `state` (*State*): The initialized `State` object.

### `generate_simulation_data(state, n_steps=99)`
Generates state and observation sequences for a specified number of steps.

- **Parameters:**
  - `state` (*State*): Instance of the `State` class.
  - `n_steps` (*int*): Number of time steps to simulate.
- **Returns:**
  - `x` (*list of np.ndarray*): State vectors over time.
  - `y` (*list of np.ndarray*): Observation vectors over time.

### `visualize_simulation(y, n)`
Visualizes the first `n` components of the observation vector over time.

- **Parameters:**
  - `y` (*list of np.ndarray*): Observation vectors over time.
  - `n` (*int*): Number of components to visualize.

### `visualize_x(x, n)`
Visualizes the first `n` components of the state vector over time.

- **Parameters:**
  - `x` (*list of np.ndarray*): State vectors over time.
  - `n` (*int*): Number of components to visualize.

---

## Utility Functions

### `mvnrand(mean, covariance_matrix, dim, n=1)`
Generates random samples from a multivariate normal distribution.

- **Parameters:**
  - `mean` (*float or np.ndarray*): Mean vector (or scalar for 1D case).
  - `covariance_matrix` (*np.ndarray or float*): Covariance matrix.
  - `dim` (*int*): Dimensionality of the distribution.
  - `n` (*int*): Number of samples.
- **Returns:**
  - *np.ndarray*: Samples from the specified distribution.

### `generate_low_correlation_cov_matrix(M, var_range=(1, 2), corr_range=(-0.1, 0.1))`
Generates an `MxM` covariance matrix with low correlations between variables.

- **Parameters:**
  - `M` (*int*): Size of the covariance matrix.
  - `var_range` (*tuple*): Range for diagonal variances.
  - `corr_range` (*tuple*): Range for correlations.
- **Returns:**
  - *np.ndarray*: Covariance matrix.

---

## Example Usage

```python
# Initialize State object
state = State(dim_state=3, dim_obs=2, mode=1, R=0.5, R0=0.1, n_steps=100)

# Generate simulation data
x, y = generate_simulation_data(state, n_steps=100)

# Visualize results
visualize_simulation(y, n=2)
visualize_x(x, n=3)

# Run multi-trial simulation
X_all_trials, Y_all_trials, state = run_simulation(
    dim_state=3,
    dim_obs=2,
    mode=1,
    R=0.5,
    R0=0.1,
    n_steps=100,
    num_trials=5
)
