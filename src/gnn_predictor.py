import torch
import pickle
import numpy as np
from typing import Dict, Any
from gnn import PowerPredictionGNN

class GNNPowerPredictor:
    def __init__(self, model_path: str, scalers_path: str):
        """
        Initialize the GNN power predictor with saved model and scalers
        """
        # Load scalers
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
            self.feature_scaler = scalers['feature_scaler']
            self.target_scaler = scalers['target_scaler']
        
        # Define features list
        self.features = [
            'num_nodes_alloc', 'cores_per_task', 'cores_per_node',
            'shared', 'priority', 'memory_gb', 'estimated_runtime',
            'job_type_encoded'
        ]
        
        # Initialize and load model
        self.model = PowerPredictionGNN(input_dim=len(self.features))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict_power(self, job: Dict[str, Any]) -> float:
        """
        Predict power consumption for a single job
        """
        # Extract features in correct order
        features = np.array([[
            job['num_nodes_alloc'],
            job['cores_per_task'],
            job['cores_per_node'],
            job['shared'],
            job['priority'],
            job['memory_gb'],
            job['estimated_runtime'],
            job['job_type_encoded']
        ]])
        
        # Scale features
        scaled_features = self.feature_scaler.transform(features)
        
        # Convert to tensor
        x = torch.FloatTensor(scaled_features)
        
        # Create a simple edge index for single node (no edges)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Get prediction
        with torch.no_grad():
            pred = self.model(x, edge_index)
        
        # Inverse transform to get actual power value
        power = self.target_scaler.inverse_transform(pred.numpy())[0][0]
        
        return power

def prepare_jobs_with_gnn_power(df, gnn_predictor):
    """
    Prepare job dataframe by adding GNN power predictions
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    try:
        # Add job type if missing
        if 'job_type_encoded' not in df.columns:
            df['job_type_encoded'] = 0
        
        # Add other required columns with defaults if missing
        required_columns = {
            'cores_per_task': 1,
            'cores_per_node': 1,
            'shared': 0,
            'priority': 1,
            'memory_gb': 4,
            'estimated_runtime': 3600
        }
        
        for col, default in required_columns.items():
            if col not in df.columns:
                df[col] = default
        
        # Predict power for each job
        powers = []
        for _, job in df.iterrows():
            power = gnn_predictor.predict_power(job.to_dict())
            powers.append(power)
        
        # Replace mean_node_power with predictions
        df['mean_node_power'] = powers
        
        # Update power_kw based on new predictions
        df['power_kw'] = df['mean_node_power'] * df['num_nodes_alloc'] / 1000
        
    except Exception as e:
        print(f"Error in prepare_jobs_with_gnn_power: {str(e)}")
        # Fallback to a simple power model if GNN fails
        df['mean_node_power'] = df['num_nodes_alloc'] * 200  # 200W per node
        df['power_kw'] = df['mean_node_power'] / 1000
    
    return df