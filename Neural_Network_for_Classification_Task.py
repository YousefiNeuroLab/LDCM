
class CNN1D(nn.Module):
    """
    A 1D Convolutional Neural Network (CNN) designed for time-series data classification.

    Attributes:
        input_channels (int): Number of input channels (e.g., features or signals).
        input_time_steps (int): Number of time steps in the input data.
    """
    def __init__(self, input_channels, input_time_steps):
        """
        Initializes the CNN1D model with convolutional, batch normalization, pooling,
        dropout, and fully connected layers.

        Args:
            input_channels (int): Number of input channels.
            input_time_steps (int): Number of time steps in the input data.
        """
        super(CNN1D, self).__init__()

        # First convolutional block: Captures local patterns with a small kernel size
        self.conv1 = nn.Conv1d(input_channels, out_channels=8, kernel_size=80, padding=1)
        self.bn1 = nn.BatchNorm1d(8)

        # Second convolutional block: Increases feature depth for higher-level patterns
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=80, padding=1)
        self.bn2 = nn.BatchNorm1d(16)

        # Third convolutional block: Smaller kernel size to focus on finer details
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16, padding=3)
        self.bn3 = nn.BatchNorm1d(32)

        # Fourth convolutional block: Larger kernel for capturing broader patterns
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=32, padding=1)
        self.bn4 = nn.BatchNorm1d(64)

        # Pooling layer: Reduces dimensionality while retaining important features
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        # Dropout: Prevents overfitting by randomly dropping neurons during training
        self.dropout = nn.Dropout(p=0.3)

        # Calculate the flattened size for the input to fully connected layers
        flattened_size = self.calculate_fc_input_size(input_channels, input_time_steps)

        # Fully connected layers: Map extracted features to output labels
        self.fc1 = nn.Linear(flattened_size, 64)  # First FC layer with 64 neurons
        self.fc2 = nn.Linear(64, 2)  # Final FC layer with 2 output neurons for binary classification

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor with shape (batch_size, input_channels, input_time_steps).

        Returns:
            Tensor: Output probabilities with shape (batch_size, 2).
        """
        # First convolutional block: Convolution -> BatchNorm -> LeakyReLU -> Pooling -> Dropout
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        # Second convolutional block
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        # Third convolutional block
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        # Fourth convolutional block
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)

        # Flatten the tensor for input to fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.leaky_relu(self.fc1(x))  # First FC layer
        x = self.dropout(x)

        x = F.softmax(self.fc2(x), dim=1)  # Output probabilities
        return x

    def calculate_fc_input_size(self, input_channels, input_time_steps):
        """
        Calculate the size of the flattened input to the first fully connected layer.

        Args:
            input_channels (int): Number of input channels.
            input_time_steps (int): Number of time steps in the input data.

        Returns:
            int: Size of the flattened tensor.
        """
        # Create a dummy input tensor with the same shape as the input data
        x = torch.zeros(1, input_channels, input_time_steps)

        # Pass the dummy tensor through the convolutional and pooling layers
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))

        # Flatten the tensor and return its size
        return x.view(1, -1).size(1)

def split_data(x, labels, train_size=0.8, val_size=0.2):
    """
    Split data into training and testing sets.

    Args:
        x (Tensor or ndarray): Input data (features).
        labels (Tensor or ndarray): Corresponding labels for the data.
        train_size (float): Proportion of the data to include in the training set.
        val_size (float): Proportion of the data to include in the validation set (not used in this version).

    Returns:
        tuple: A tuple containing:
            - (x_train, y_train): Training data and labels.
            - (x_test, y_test): Testing data and labels.
    """
    # Split the dataset into training and testing sets, stratified by labels for balanced distribution
    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, train_size=train_size, stratify=labels, random_state=42
    )
    return (x_train, y_train), (x_test, y_test)


def create_data_loader(x, y, batch_size=2, shuffle=False):
    """
    Create a DataLoader for the given data.

    Args:
        x (Tensor): Input data (features).
        y (Tensor): Corresponding labels for the data.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data before each epoch.

    Returns:
        DataLoader: A PyTorch DataLoader instance for batching and iteration.
    """
    # Combine the features and labels into a dataset
    dataset = TensorDataset(x, y)

    # Create a DataLoader for batching and optional shuffling
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    """
    Train the CNN model on the given dataset.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader providing training data batches.
        criterion (torch.nn.Module): Loss function used for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        epochs (int): Number of training epochs.

    Returns:
        None
    """
    # Loop through the specified number of training epochs
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0  # Initialize running loss for the epoch

        # Iterate over all batches in the training DataLoader
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Clear gradients from the previous step
            outputs = model(inputs)  # Forward pass to compute model predictions
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass to compute gradients
            optimizer.step()  # Update model weights using the optimizer
            train_loss += loss.item() * inputs.size(0)  # Accumulate batch loss

        # Calculate average loss over the entire dataset
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

# Generate synthetic data
x = torch.randn(100, 360, 1)  # (samples, time_steps, dimensions)
x = x.permute(0, 2, 1)
labels = torch.randint(0, 2, (100, 1)).squeeze()  # (samples, )

# Split data
(x_train, y_train), (x_test, y_test) = split_data(x, labels)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# Create DataLoaders
train_loader = create_data_loader(x_train, y_train, shuffle=True)
test_loader = create_data_loader(x_test, y_test)

# Initialize model, loss function, and optimizer
input_channels = 1  # Example: 40 particles * 2 dimensions
input_time_steps = 360
model_temp = CNN1D(input_channels, input_time_steps)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_temp.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization with weight decay

# Train the model
train_model(model_temp, train_loader, criterion, optimizer)
