import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Data Preprocessing ---
def preprocess_data(df):
    """
    Preprocess the HPC job data for GNN training
    """
    # Encode categorical variables
    le_type = LabelEncoder()
    df['job_type_encoded'] = le_type.fit_transform(df['job_type'])
    
    # Feature selection
    gnn_features = [
        'num_nodes_alloc', 'cores_per_task', 'cores_per_node',
        'shared', 'priority', 'memory_gb', 'estimated_runtime',
        'job_type_encoded'
    ]
    
    # Create job size and runtime groups for stratification
    df['size_group'] = pd.qcut(df['num_nodes_alloc'], q=5, labels=['vs', 's', 'm', 'l', 'vl'])
    df['runtime_group'] = pd.qcut(df['run_time'], q=5, labels=['vs', 's', 'm', 'l', 'vl'])
    df['strata'] = df['size_group'].astype(str) + '_' + df['runtime_group'].astype(str)
    
    return df, gnn_features

def prepare_gnn_data(df, features, target=['mean_node_power'], feature_scaler=None, target_scaler=None):
    """
    Prepare data for GNN including feature scaling and neighbor calculation
    """
    # Scale features
    if feature_scaler is None:
        feature_scaler = StandardScaler()
        X = feature_scaler.fit_transform(df[features])
    else:
        X = feature_scaler.transform(df[features])
    
    # Scale target
    if target_scaler is None:
        target_scaler = StandardScaler()
        y = target_scaler.fit_transform(df[target])
    else:
        y = target_scaler.transform(df[target])
    
    # Calculate neighbors based on job characteristics
    knn = NearestNeighbors(n_neighbors=6)  # k=5 + self
    knn.fit(X)
    distances, indices = knn.kneighbors(X)
    
    return X, y, indices[:, 1:], feature_scaler, target_scaler

def create_graph_batch(features, neighbor_indices, targets, batch_size=128):
    """
    Create graph batches with edges based on KNN
    """
    graphs = []
    num_samples = len(features)
    
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_features = features[i:batch_end]
        batch_neighbors = neighbor_indices[i:batch_end]
        batch_targets = targets[i:batch_end]
        
        edge_index = []
        for job_idx, neighbors in enumerate(batch_neighbors):
            for neighbor_idx in neighbors:
                local_neighbor_idx = neighbor_idx - i
                if 0 <= local_neighbor_idx < len(batch_features):
                    edge_index.append([job_idx, local_neighbor_idx])
                    edge_index.append([local_neighbor_idx, job_idx])
        
        if not edge_index:
            continue
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        
        graph = Data(
            x=torch.FloatTensor(batch_features),
            edge_index=edge_index,
            y=torch.FloatTensor(batch_targets)
        )
        graphs.append(graph)
    
    return graphs

class PowerPredictionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(PowerPredictionGNN, self).__init__()
        
        # Initial node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Power prediction layers
        self.power_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index):
        # Node embedding
        x = self.node_embed(x)
        
        # Graph convolutions with residual connections
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index))) + x1
        
        # Power prediction
        return self.power_pred(x2)

def train_model(model, train_graphs, val_graphs, epochs=100, lr=0.001, patience=15):
    """
    Train the GNN model with early stopping
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for graph in train_graphs:
            optimizer.zero_grad()
            out = model(graph.x, graph.edge_index)
            loss = F.mse_loss(out, graph.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for graph in val_graphs:
                out = model(graph.x, graph.edge_index)
                val_loss += F.mse_loss(out, graph.y).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_graphs):.4f}, "
                  f"Val Loss = {val_loss/len(val_graphs):.4f}")

def evaluate_model(model, test_graphs, target_scaler):
    """
    Evaluate the model on test data
    """
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for graph in test_graphs:
            pred = model(graph.x, graph.edge_index)
            predictions.extend(target_scaler.inverse_transform(pred.numpy()))
            true_values.extend(target_scaler.inverse_transform(graph.y.numpy()))
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    metrics = {
        'mse': mean_squared_error(true_values, predictions),
        'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
        'mae': mean_absolute_error(true_values, predictions),
        'r2': r2_score(true_values, predictions)
    }
    
    return metrics, predictions, true_values