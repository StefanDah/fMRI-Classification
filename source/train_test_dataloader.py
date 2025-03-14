from sklearn.model_selection import  train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset


def create_train_test_dataloader(data, labels, test_size=0.2, batch_size=8):
    
    """
    Create train and test dataloaders from the given data and labels.

    Parameters:
    data (array-like): The input data.
    labels (array-like): The labels corresponding to the data.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    batch_size (int, optional): The number of samples per batch to load. Default is 8.

    Returns:
    DataLoader:
        - train_dataloader (DataLoader): DataLoader for the training data.
        - test_dataloader (DataLoader): DataLoader for the test data.
    """
    
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels,
        test_size=test_size,
        stratify=labels,
        random_state=42
    )
    
    train_data, test_data = torch.tensor(train_data, dtype=torch.float32), torch.tensor(test_data, dtype=torch.float32)
    train_labels, test_labels = torch.tensor(train_labels, dtype=torch.int64), torch.tensor(test_labels, dtype=torch.int64)

    train_dataloader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size, shuffle=False)

    print(f"Train Data Shape: {train_data.shape}")  # (train_samples, 100, num_rois)
    print(f"Test Data Shape: {test_data.shape}")

    print(f"Train dataloader: {len(train_dataloader)} batches")
    print(f"Test dataloader: {len(test_dataloader)} batches")
    
    return train_dataloader, test_dataloader


