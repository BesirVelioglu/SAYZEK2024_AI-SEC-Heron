import torch
import yaml
from models import Model  # Import the custom model class from the models directory
from utils import load_data, save_model  # Import utility functions for data loading and model saving
from torch.optim import AdamW  # Import AdamW optimizer for weight decay regularization
import argparse  # Import for command-line argument parsing

# Argument parser to dynamically select the configuration YAML file at runtime
parser = argparse.ArgumentParser(description='Training script for SAYZEK Datathon 2024')
parser.add_argument('--config', type=str, required=True, help='Path to the dataset config file (YAML)')
args = parser.parse_args()  # Parse the arguments from the command line

# Load the dataset configuration file
# This config file should contain paths to training and validation data, as well as hyperparameters like number of classes
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)  # Safely load the YAML file contents into a dictionary

# Model initialization based on the number of classes (nc) defined in the config file
# 'Model' should be a custom-defined neural network architecture
model = Model(num_classes=config['nc'])

# Define optimizer (AdamW) with weight decay (L2 regularization), learning rate can be adjusted as needed
optimizer = AdamW(model.parameters(), lr=0.001)

# Define the loss function. GIoULoss is appropriate for object detection tasks involving bounding boxes.
# Ensure that your task uses a compatible loss function.
criterion = torch.nn.GIoULoss()  

# Load the training and validation data using paths and parameters from the config
# `load_data` should handle data preprocessing, batching, and augmentation
train_loader, val_loader = load_data(config['train'], config['val'], batch_size=16)

# Training loop setup
# `num_epochs` can also be dynamically adjusted through the config file or hardcoded as needed
num_epochs = 300  # Total number of epochs to train the model

# Loop through each epoch for training
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    # Iterate over each batch in the training data
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients before backpropagation

        # Forward pass: compute model output on the training batch
        outputs = model(images)

        # Compute the loss between the predicted outputs and ground truth labels
        loss = criterion(outputs, labels)

        # Backward pass: compute the gradients of the loss w.r.t. the model parameters
        loss.backward()

        # Step the optimizer to update the model's weights based on the computed gradients
        optimizer.step()

    # Optionally, add validation logic here to evaluate the model on validation data after each epoch

    # Print the current epoch and loss for monitoring
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the trained model's weights to a file after the final epoch
save_model(model, 'models/best_model.pth')
