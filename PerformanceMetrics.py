
# Performance Metrics
# This module provides functions to evaluate model performance using metrics such as accuracy, ROC curves, and AUC.

import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def evaluate_model_accuracy(model, loader, criterion=None):
    """
    Evaluate the model on a given dataset and compute accuracy.

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader providing the dataset to evaluate.
        criterion (torch.nn.Module, optional): Loss function to compute the loss during evaluation.

    Returns:
        float or tuple: Accuracy of the model on the dataset. If `criterion` is provided, returns a tuple of 
                        average loss and accuracy.
    """
    # Set the model to evaluation mode (disable training-specific behaviors like dropout)
    model.eval()

    # Initialize variables to track total loss and correct predictions
    loss = 0.0
    correct = 0

    # Disable gradient computation during evaluation to save memory and computation
    with torch.no_grad():
        for inputs, labels in loader:
            # Pass inputs through the model to get predictions
            outputs = model(inputs)

            # Debugging: Print outputs for inspection (can be removed in production)
            print("outputs", outputs)

            # If a loss criterion is provided, compute the batch loss and accumulate it
            if criterion:
                loss += criterion(outputs, labels).item() * inputs.size(0)

            # Determine the predicted class by finding the index with the highest value in the output
            _, predicted = torch.max(outputs, 1)

            # Count the number of correct predictions
            correct += (predicted == labels).sum().item()

    # Calculate accuracy as the ratio of correct predictions to the total dataset size
    accuracy = correct / len(loader.dataset)

    # If a loss criterion is provided, calculate the average loss and return it with the accuracy
    if criterion:
        loss /= len(loader.dataset)  # Normalize the loss by the number of samples
        return loss, accuracy

    # If no loss criterion is provided, return only the accuracy
    return accuracy

def evaluate_model_roc_curve_multi_particle(model, loader, criterion=None):
    """
    Evaluate the model for multi-particle trials, calculate the ROC curve, and compute AUC.

    Parameters:
        model (torch.nn.Module): Trained model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader providing the test dataset.
        criterion (torch.nn.Module, optional): Loss function to compute the loss during evaluation.

    Returns:
        float or tuple: AUC score if no criterion is provided, or a tuple of loss and AUC score if criterion is provided.
    """
    # Set the model to evaluation mode (disable training-specific behaviors like dropout)
    model.eval()
    
    # Initialize variables to store loss, labels, and model outputs
    loss = 0.0
    all_labels = []  # Store true labels from all batches
    all_outputs = []  # Store model predictions for the positive class

    # Disable gradient computation during evaluation to save memory and improve efficiency
    with torch.no_grad():
        for inputs, labels in loader:
            # Reshape the input tensor for multi-particle data
            # Original shape: (batch_size, particles, time_steps, channels)
            batch_size, particles, time_steps, channels = inputs.size()
            inputs = inputs.view(batch_size * particles, channels, time_steps)

            # Reshape the labels to match the flattened input dimension
            # Each particle in a batch will get the same label
            labels = labels.repeat_interleave(particles)

            # Pass inputs through the model to get predictions
            outputs = model(inputs)

            # If a criterion is provided, compute the batch loss and accumulate it
            if criterion:
                loss += criterion(outputs, labels).item() * inputs.size(0)

            # Extract probabilities for the positive class (assumed to be at index 1)
            all_outputs.append(outputs[:, 1].cpu().numpy())

            # Append true labels to the list
            all_labels.append(labels.cpu().numpy())

    # Concatenate all predictions and labels into single arrays for evaluation
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    # Compute the ROC curve: false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)

    # Compute the AUC (Area Under the Curve) for the ROC curve
    auc_score = roc_auc_score(all_labels, all_outputs)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Reference line for random guessing
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()

    # If a criterion was used, return both the average loss and AUC score
    if criterion:
        loss /= len(loader.dataset)  # Normalize the loss by the number of samples
        return loss, auc_score

    # If no criterion was used, return only the AUC score
    return auc_score
