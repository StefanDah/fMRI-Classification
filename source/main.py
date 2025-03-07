import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from load_data import load_data, preprocessing
from compute_roi_and_connectome import compute_roi_time_series, compute_correlation_matrix
from train_test_dataloader import create_train_test_dataloader
from model import meanMLP 
from train import train_model

print("Current working directory:", os.getcwd())

# Get the absolute path to main.py
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move up one directory from 'source' to the project root
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Build the path to the data folder
data_path = os.path.join(project_root, 'data')

# Load data and create dict with subject labels (=condition)
(subjects,subject_labels, data_path) = load_data(data_path)

# Perform preprocessing steps: co-registration, resampling to MIN space and smoothing
preprocessing(subjects, data_path)

# Compute ROI time series
data = compute_roi_time_series(subjects, data_path)
np.save("data.npy", data)
# Generate data and labels datasets
y = list(subject_labels.values())
labels = np.array(y)
np.save("data.npy", labels)

print(labels)

print(f"Data shape: {data.shape}: (n_samples, time_length, n_components)")
print(f"Labels shape: {labels.shape}, unique labels: {np.unique(labels)}")

(train_dataloader, test_dataloader) = create_train_test_dataloader(data, labels, test_size=0.2, batch_size=8)


### meanMLP model
# Set config 
default_model_cfg = {
    "dropout": 0.49,
    "hidden_size": 160,
    "input_size": data.shape[2],            # size of the input features per time point
    "output_size": np.unique(labels).shape[0] # number of classes
}
# Initialize 
model = meanMLP(default_model_cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(model)  

# Train 
history = train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    device=device,
    epochs=50,
    lr=0.001,
    weight_decay=1e-4,
    scheduler_type="ReduceLROnPlateau",  # Optional: Use "StepLR" or "None"
)

### SVC model
# comes here