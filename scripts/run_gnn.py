import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
from gnn import (
    preprocess_data, prepare_gnn_data, create_graph_batch,
    PowerPredictionGNN, train_model, evaluate_model
)

# Set matplotlib backend to 'Agg' for non-interactive environments
plt.switch_backend('Agg')

# 1. Load your data
df = pd.read_csv('/home/abrar/Desktop/Code/Temporal HPC/hpc_simulator/synthetic_hpc_jobs_2020.csv')  # Replace with your data path

# 2. Preprocess the data
processed_df, gnn_features = preprocess_data(df)

# 3. Split the data
train_val_idx, test_idx = train_test_split(
    processed_df.index, 
    test_size=0.2, 
    stratify=processed_df['strata'],
    random_state=42
)

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.2,
    stratify=processed_df.loc[train_val_idx, 'strata'],
    random_state=42
)

train_data = processed_df.loc[train_idx]
val_data = processed_df.loc[val_idx]
test_data = processed_df.loc[test_idx]

# 4. Prepare data for GNN
train_X, train_y, train_neighbors, feature_scaler, target_scaler = prepare_gnn_data(
    train_data, gnn_features, ['mean_node_power']
)

val_X, val_y, val_neighbors, _, _ = prepare_gnn_data(
    val_data, gnn_features, ['mean_node_power'],
    feature_scaler, target_scaler
)

test_X, test_y, test_neighbors, _, _ = prepare_gnn_data(
    test_data, gnn_features, ['mean_node_power'],
    feature_scaler, target_scaler
)

# 5. Create graph batches
train_graphs = create_graph_batch(train_X, train_neighbors, train_y)
val_graphs = create_graph_batch(val_X, val_neighbors, val_y)
test_graphs = create_graph_batch(test_X, test_neighbors, test_y)

# 6. Initialize and train the model
model = PowerPredictionGNN(input_dim=len(gnn_features))
print("Training model...")
train_model(
    model,
    train_graphs,
    val_graphs,
    epochs=100,
    lr=0.001,
    patience=15
)

# 7. Evaluate the model
print("\nEvaluating model...")
metrics, predictions, true_values = evaluate_model(model, test_graphs, target_scaler)

# Create directory for results
import os
results_dir = os.path.join(os.getcwd(), 'gnn_results')
os.makedirs(results_dir, exist_ok=True)

# Save prediction data
prediction_df = pd.DataFrame({
    'true_values': true_values.ravel(),
    'predictions': predictions.ravel()
})
prediction_df.to_csv(os.path.join(results_dir, 'predicted_vs_actual.csv'), index=False)

# Save residuals data
residuals = predictions.ravel() - true_values.ravel()
residuals_df = pd.DataFrame({
    'residuals': residuals
})
residuals_df.to_csv(os.path.join(results_dir, 'prediction_errors.csv'), index=False)

# Save error by job type data
if 'job_type' in test_data.columns:
    try:
        error_by_type_df = pd.DataFrame({
            'Job_Type': test_data['job_type'].values,
            'Absolute_Error': np.abs(residuals)
        })
        error_by_type_df.to_csv(os.path.join(results_dir, 'error_by_job_type.csv'), index=False)
    except Exception as e:
        print(f"Warning: Could not create job type error analysis data: {str(e)}")

# Now create the plots (plotting code remains the same)
plt.figure(figsize=(10, 6))
plt.scatter(true_values, predictions, alpha=0.5)
plt.plot([min(true_values), max(true_values)], 
         [min(true_values), max(true_values)], 
         'r--', label='Perfect Prediction')
plt.xlabel('Actual Power')
plt.ylabel('Predicted Power')
plt.title('Predicted vs Actual Power Usage')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'predicted_vs_actual.png'))
plt.close()

# Residuals plot
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Prediction Error (Watts)')
plt.ylabel('Count')
plt.title('Distribution of Prediction Errors')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'prediction_errors.png'))
plt.close()

# Error analysis by job type
if 'job_type' in test_data.columns:
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Job_Type', y='Absolute_Error', data=error_by_type_df)
        plt.xticks(rotation=45)
        plt.title('Prediction Error by Job Type')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'error_by_job_type.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create job type error analysis plot: {str(e)}")

# Save model and scalers in the results directory
print("Saving model and scalers...")
torch.save(model.state_dict(), os.path.join(results_dir, 'power_prediction_gnn.pt'))

# Save the scalers
with open(os.path.join(results_dir, 'scalers.pkl'), 'wb') as f:
    pickle.dump({
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }, f)

print("\nTraining and evaluation complete!")
print("Results saved in directory:", results_dir)
print("Files saved:")
print("- Model weights: power_prediction_gnn.pt")
print("- Scalers: scalers.pkl")
print("- Data: predicted_vs_actual.csv, prediction_errors.csv, error_by_job_type.csv")
print("- Plots: predicted_vs_actual.png, prediction_errors.png, error_by_job_type.png")


# Print detailed metrics
print("\nDetailed Metrics:")
print("-" * 50)
print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
print(f"R² Score: {metrics['r2']:.4f}")

# Additional error statistics
print("\nError Statistics:")
print("-" * 50)
print(f"Mean Error: {np.mean(residuals):.4f}")
print(f"Std Error: {np.std(residuals):.4f}")
print(f"Median Error: {np.median(residuals):.4f}")
print(f"90th Percentile Error: {np.percentile(np.abs(residuals), 90):.4f}")
print(f"95th Percentile Error: {np.percentile(np.abs(residuals), 95):.4f}")