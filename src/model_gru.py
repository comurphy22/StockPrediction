"""
GRU Model for Stock Prediction

Implements Gated Recurrent Unit (GRU) networks for sequence-based
stock movement prediction. GRU is a simpler alternative to LSTM with
fewer parameters, which may generalize better on limited data.

Paper Justification:
The paper proposes "gated recurrent units, which are designed to capture
temporal dependencies."

GRU vs LSTM:
- GRU has 2 gates (update, reset) vs LSTM's 3 (input, forget, output)
- Fewer parameters → faster training, less overfitting risk
- Often performs similarly to LSTM on shorter sequences
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

# Import sequence preparation from LSTM module
from model_lstm import prepare_sequences


class StockGRU(nn.Module):
    """
    GRU model for binary stock movement prediction.
    
    Architecture:
    - Input: [batch_size, sequence_length, n_features]
    - GRU layers with dropout regularization
    - Fully connected output layer
    - Sigmoid activation for binary classification
    
    Parameters:
    -----------
    input_size : int
        Number of features per time step
    hidden_size : int, default=64
        Number of hidden units in GRU layers
    num_layers : int, default=2
        Number of stacked GRU layers
    dropout : float, default=0.3
        Dropout probability (applied if num_layers > 1)
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(StockGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch_size, sequence_length, n_features]
        
        Returns:
        --------
        torch.Tensor
            Output probabilities of shape [batch_size, 1]
        """
        # GRU output: (batch_size, seq_len, hidden_size)
        gru_out, _ = self.gru(x)
        
        # Take output from last time step
        last_output = gru_out[:, -1, :]
        
        # Fully connected layer + sigmoid
        out = self.fc(last_output)
        out = self.sigmoid(out)
        
        return out


def train_gru_model(X_train, y_train, X_test, y_test,
                   sequence_length=10, hidden_size=64, num_layers=2,
                   epochs=50, learning_rate=0.001, verbose=True):
    """
    Train GRU model for stock prediction.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features (n_samples, n_features)
    y_train : np.ndarray
        Training targets (n_samples,)
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets
    sequence_length : int, default=10
        Number of days to look back
    hidden_size : int, default=64
        GRU hidden size
    num_layers : int, default=2
        Number of GRU layers
    epochs : int, default=50
        Training epochs
    learning_rate : float, default=0.001
        Adam optimizer learning rate
    verbose : bool, default=True
        Print training progress
    
    Returns:
    --------
    model : StockGRU
        Trained model
    train_metrics : dict
        Training accuracy
    test_metrics : dict
        Test metrics (accuracy, precision, recall, f1)
    training_history : dict
        Loss history for visualization
    """
    if verbose:
        print(f"   Preparing sequences (length={sequence_length})...")
    
    start_time = time.time()
    
    # Prepare sequences
    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = prepare_sequences(X_test, y_test, sequence_length)
    
    if verbose:
        print(f"      Train: {X_train_seq.shape[0]} sequences")
        print(f"      Test:  {X_test_seq.shape[0]} sequences")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq).unsqueeze(1)
    
    # Initialize model
    input_size = X_train_seq.shape[2]
    model = StockGRU(input_size, hidden_size, num_layers)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    if verbose:
        print(f"   Training GRU ({epochs} epochs)...")
    
    training_history = {'loss': []}
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        training_history['loss'].append(loss.item())
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"      Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        # Training accuracy
        train_pred = (model(X_train_tensor) > 0.5).float()
        train_acc = (train_pred == y_train_tensor).float().mean().item()
        
        # Test predictions
        test_pred = (model(X_test_tensor) > 0.5).float()
        test_acc = (test_pred == y_test_tensor).float().mean().item()
        
        # Additional metrics
        y_pred = test_pred.squeeze().numpy()
        y_true = y_test_tensor.squeeze().numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
    
    runtime = time.time() - start_time
    
    if verbose:
        print(f"   [OK] GRU trained in {runtime:.1f}s")
    
    train_metrics = {'accuracy': train_acc}
    test_metrics = {
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return model, train_metrics, test_metrics, training_history


def evaluate_gru_model(model, X, y, sequence_length=10):
    """
    Evaluate trained GRU model.
    
    Parameters:
    -----------
    model : StockGRU
        Trained model
    X : np.ndarray
        Features
    y : np.ndarray
        True labels
    sequence_length : int, default=10
        Sequence length used in training
    
    Returns:
    --------
    dict : Evaluation metrics
    """
    # Prepare sequences
    X_seq, y_seq = prepare_sequences(X, y, sequence_length)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)
    
    # Predict
    model.eval()
    with torch.no_grad():
        predictions = (model(X_tensor) > 0.5).float()
        accuracy = (predictions == y_tensor).float().mean().item()
        
        y_pred = predictions.squeeze().numpy()
        y_true = y_tensor.squeeze().numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing GRU implementation...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 25
    
    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples) > 0.5).astype(int)
    
    # Train/test split
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train GRU
    print("\nTraining GRU on synthetic data...")
    model, train_metrics, test_metrics, history = train_gru_model(
        X_train, y_train, X_test, y_test,
        sequence_length=10,
        epochs=20,
        verbose=True
    )
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_metrics['accuracy']:.2%}")
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.2%}")
    print(f"  Test Precision: {test_metrics['precision']:.2%}")
    print(f"  Test Recall:    {test_metrics['recall']:.2%}")
    print(f"  Test F1:        {test_metrics['f1']:.2%}")
    
    print(f"\n[OK] GRU implementation working correctly")

