import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from load_data import load_data, preprocessing
from compute_roi_and_connectome import compute_roi_time_series, compute_correlation_matrix
from train_test_dataloader import create_train_test_dataloader
from model import meanMLP 
from train import train_meanMLP_model, preprocess_and_train_svc
from sklearn.svm import SVC
from utils import plot_confusion_matrix


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
# preprocessing(subjects, data_path)

#Compute ROI time series
(data, X_roi_list) = compute_roi_time_series(subjects, data_path)
np.save("data.npy", data)

# Compute correlation matrix
X_correlation = compute_correlation_matrix(X_roi_list, subjects)

# Generate data and labels datasets
y = list(subject_labels.values())
labels = np.array(y)
np.save("data.npy", labels)

print(labels)

print(f"Data shape: {data.shape}: (n_samples, time_length, n_components)")
print(f"Labels shape: {labels.shape}, unique labels: {np.unique(labels)}")




### SVC model
model_svc = SVC(kernel='linear')

# Run the preprocessing, training, and evaluation
result = preprocess_and_train_svc(X_correlation, labels, kernel="rbf")

# Extract outputs
svm_model = result["best_model"]
metrics = result["evaluation_metrics"]

print(f"Best Model: {svm_model}")
print(f"Evaluation Metrics: {metrics}")

y_test = result["y_test"]
y_pred = result["y_pred"] 

plot_confusion_matrix(y_test, y_pred, save_path="figures/confusion_matrix.png")


### meanMLP model

(train_dataloader, test_dataloader) = create_train_test_dataloader(data, labels, test_size=0.2, batch_size=8)

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
history = train_meanMLP_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    device=device,
    epochs=50,
    lr=0.001,
    weight_decay=1e-4,
    scheduler_type="ReduceLROnPlateau",  # Optional: Use "StepLR" or "None"
)



