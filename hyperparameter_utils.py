from early_stopping import EarlyStopping
from torch.utils.data import DataLoader

import random
import torch

def sample_hyperparameters(param_space):
    """
    Randomly samples hyperparameters from the provided parameter space.
    
    Args:
        param_space (dict): Dictionary of hyperparameter options.
        
    Returns:
        dict: A randomly sampled hyperparameter configuration.
    """
    return {
        'hidden_layers': random.choice(param_space['hidden_layers']),
        'dropout_rate': random.choice(param_space['dropout_rate']),
        'learning_rate': random.choice(param_space['learning_rate']),
        'batch_size': random.choice(param_space['batch_size']),
    }

def train_and_evaluate(params, model_class, input_dim, train_dataset, validate_dataset, criterion, num_epochs=10):
    """
    Trains and evaluates a model with the given hyperparameters.
    
    Args:
        params (dict): Hyperparameter configuration.
        model_class (class): Model class to instantiate.
        input_dim (int): Number of input features.
        train_dataset (Dataset): PyTorch Dataset for training.
        validate_dataset (Dataset): PyTorch Dataset for validation.
        criterion (Loss): Loss function for optimization.
        num_epochs (int): Number of training epochs.

    Returns:
        float: Best validation loss achieved.
        dict: The hyperparameter configuration.
    """
    
    print(f"Testing configuration: {params}")
    hidden_layers = params['hidden_layers']
    dropout_rate = params['dropout_rate']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    
    model = model_class(input_dim=input_dim, hidden_layers=hidden_layers, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size)
    
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    best_validate_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        validate_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in validate_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                validate_loss += loss.item()
        validate_loss /= len(validate_loader)
        
        if validate_loss < best_validate_loss:
            best_validate_loss = validate_loss

        early_stopping.check(validate_loss)
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Best Validation Loss for configuration {params}: {best_validate_loss:.4f}\n")

    return best_validate_loss, params