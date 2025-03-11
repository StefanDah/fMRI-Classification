import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



def train_meanMLP_model(
    model,
    train_dataloader,
    test_dataloader,
    device,
    epochs=100,
    lr=0.001,
    weight_decay=1e-4,
    scheduler_type=None,
):
    """
    Train the meanMLP model and directly plot training history.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        test_dataloader (DataLoader): DataLoader for testing data.
        device (torch.device): Device to run the training on ('cuda' or 'cpu').
        epochs (int, optional): Number of epochs. Default is 100.
        lr (float, optional): Learning rate. Default is 0.001.
        weight_decay (float, optional): L2 weight decay. Default is 1e-4.
        scheduler_type (str, optional): Choose from ['StepLR', 'ReduceLROnPlateau']. Default is None.
    """

    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler (optional)
    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
    else:
        scheduler = None

    # Store training history
    train_losses, test_losses = [], []
    train_aucs, test_aucs = [], []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        ### Train Phase
        model.train()
        train_batch_losses, true_labels, preds = [], [], []

        for x, labels in train_dataloader:
            x, labels = x.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_batch_losses.append(loss.item())
            true_labels.append(labels.cpu())
            preds.append(F.softmax(logits, dim=-1).cpu().detach())

        train_losses.append(np.mean(train_batch_losses))
        true_labels = torch.hstack(true_labels)
        preds = torch.vstack(preds)
        train_aucs.append(roc_auc_score(true_labels, preds[:, 1]))

        ### Test Phase
        model.eval()
        test_batch_losses, true_labels, preds = [], [], []

        with torch.no_grad():
            for x, labels in test_dataloader:
                x, labels = x.to(device), labels.to(device)
                logits = model(x)
                loss = criterion(logits, labels)

                test_batch_losses.append(loss.item())
                true_labels.append(labels.cpu())
                preds.append(F.softmax(logits, dim=-1).cpu().detach())

        test_losses.append(np.mean(test_batch_losses))
        true_labels = torch.hstack(true_labels)
        preds = torch.vstack(preds)
        test_aucs.append(roc_auc_score(true_labels, preds[:, 1]))

        # Learning rate scheduling step
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(test_losses[-1])
        elif scheduler_type == "StepLR":
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f}, "
            f"Test Loss: {test_losses[-1]:.4f}, Train AUC: {train_aucs[-1]:.4f}, "
            f"Test AUC: {test_aucs[-1]:.4f}"
        )

    print("Training Complete!")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Plot
    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(test_losses, label="Test Loss")
    axes[0].set_title("Loss Over Epochs")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # AUC Plot
    axes[1].plot(train_aucs, label="Train AUC")
    axes[1].plot(test_aucs, label="Test AUC")
    axes[1].set_title("AUC Over Epochs")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("AUC")
    axes[1].legend()

    # Save the plot as PNG
    plt.savefig('figures/training_plot.png', dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "train_loss": train_losses,
        "test_loss": test_losses,
        "train_auc": train_aucs,
        "test_auc": test_aucs,
    }



def preprocess_and_train_svc(X, y, test_size=0.2, cv_folds=5, kernel="rbf"):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 

    param_grid = {
        "C": [0.1, 1, 10],  # Regularization strength
        "gamma": ["scale", "auto"] if kernel == "rbf" else ["auto"],  # Kernel coefficient for RBF
        "kernel": [kernel]
    }
       
    grid_search = GridSearchCV(SVC(), param_grid, cv=cv_folds, scoring="accuracy", refit=True)
    grid_search.fit(X_train_scaled, y_train)      

    best_model = grid_search.best_estimator_
    print(f"Best SVC Hyperparameters: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test_scaled)
    
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "f1_score": f1_score(y_test, y_pred, average="binary"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # Convert to list for easy JSON storage
    }
    
    print(f"Model Evaluation:\n{metrics}")
    
    # # Save the plot as PNG
    # plt.savefig('figures/svc_training_plot.png', dpi=300, bbox_inches="tight")
    # plt.close()
    
    return {
        "best_model": best_model,
        "evaluation_metrics": metrics,
        "y_test": y_test,  # Add y_test
        "y_train": y_train,  # Add y_train
        "y_pred": y_pred,  # Add y_pred to use in confusion matrix plotting
    }
    