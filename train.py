import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from utils.dataloader import CustomDataset
from config import data_path, used_model
from utils.plots import plot_training_loss

# Training hyper‑parameters
batch_size = 32
epochs = 150
lr = 1e-3               # initial learning rate
patience = 10           # early‑stopping patience: stop if val loss does not decrease for consecutive epochs
factor = 0.8            # learning rate decay factor: lr *= factor when validation loss plateaus
min_lr = 1e-6           # lower bound of learning rate to avoid excessive decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_all_seeds(seed):
    """Set all random seeds for reproducible experiments"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Initialize seed for each dataloader subprocess worker"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


split_seed = 42
set_all_seeds(split_seed)

# Prepare dataset
root_dir = data_path
dataset = CustomDataset(root_dir)
index_dir = "data_indices"
os.makedirs(index_dir, exist_ok=True)

train_idx_path = os.path.join(index_dir, "train_idx.npy")
val_idx_path = os.path.join(index_dir, "val_idx.npy")
test_idx_path = os.path.join(index_dir, "test_idx.npy")

# Only perform dataset split when index files do NOT exist, prevent repeated splitting
if not (os.path.exists(train_idx_path) and os.path.exists(val_idx_path) and os.path.exists(test_idx_path)):
    print("Index files not found, performing dataset split")
    train_idx, temp_idx = train_test_split(np.arange(len(dataset)), test_size=0.3, random_state=split_seed)
    test_idx, val_idx = train_test_split(temp_idx, test_size=0.5, random_state=split_seed)
    np.save(train_idx_path, train_idx)
    np.save(val_idx_path, val_idx)
    np.save(test_idx_path, test_idx)
else:
    print("Loading existing saved dataset indices, skip re‑splitting")
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    test_idx = np.load(test_idx_path)

# Build data loaders
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = used_model()
print(f"Model: {model._get_name()}")
print(f"Total samples: {len(train_idx) + len(val_idx) + len(test_idx)}")
print(f"Train samples: {len(train_idx)}")
print(f"Validation samples: {len(val_idx)}")
print(f"Test samples: {len(test_idx)}")

model = model.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# Learning rate scheduler compatible with lower PyTorch versions (verbose argument removed)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # Monitor the minimum value of validation loss
    factor=factor,       # Decay factor
    patience=5,          # Decay lr if validation loss does not drop for consecutive epochs
    min_lr=min_lr       # Lower bound for learning rate (supported in older PyTorch versions)
)

# Record last learning rate for printing decay information
last_lr = optimizer.param_groups[0]['lr']

# Main training loop
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"best_{model._get_name()}_model.pth")

loss_tracker = {"train": [], "val": []}
best_val_loss = float('inf')
patience_counter = 0  # Counter for early‑stopping

for epoch in range(epochs):
    start_time = time.time()
    model.train()

    total_loss = 0
    total_mse = 0
    total_mae = 0
    total_mape = 0
    train_true = []
    train_pred = []

    for xb, yb in train_loader:
        xb, yb = xb.to(device).float(), yb.to(device).float()
        optimizer.zero_grad()
        preds = model(xb)
        yb = yb.unsqueeze(1)

        loss_mse = nn.MSELoss()(preds, yb)
        loss_mae = nn.L1Loss()(preds, yb)
        epsilon = 1e-8
        loss_mape = torch.mean(torch.abs((yb - preds) / (yb + epsilon))) * 100
        loss = criterion(preds, yb)

        loss.backward()
        # Gradient clipping to avoid gradient explosion and stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_mae += loss_mae.item()
        total_mape += loss_mape.item()

        train_true.extend(yb.cpu().detach().numpy().flatten())
        train_pred.extend(preds.cpu().detach().numpy().flatten())

    avg_train_loss = total_loss / len(train_loader)
    avg_train_mse = total_mse / len(train_loader)
    avg_train_mae = total_mae / len(train_loader)
    avg_train_mape = total_mape / len(train_loader)
    avg_train_rmse = np.sqrt(avg_train_mse)
    train_r2 = r2_score(train_true, train_pred)
    loss_tracker["train"].append(avg_train_loss)

    # Validation phase
    model.eval()
    val_total_loss = 0
    val_total_mse = 0
    val_total_mae = 0
    val_total_mape = 0
    val_true = []
    val_pred = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device).float(), yb.to(device).float()
            preds = model(xb)
            yb = yb.unsqueeze(1)

            loss_mse = nn.MSELoss()(preds, yb)
            loss_mae = nn.L1Loss()(preds, yb)
            loss_mape = torch.mean(torch.abs((yb - preds) / (yb + 1e-8)) * 100)
            loss = criterion(preds, yb)

            val_total_loss += loss.item()
            val_total_mse += loss_mse.item()
            val_total_mae += loss_mae.item()
            val_total_mape += loss_mape.item()

            val_true.extend(yb.cpu().numpy().flatten())
            val_pred.extend(preds.cpu().numpy().flatten())

    avg_val_loss = val_total_loss / len(val_loader)
    avg_val_mse = val_total_mse / len(val_loader)
    avg_val_mae = val_total_mae / len(val_loader)
    avg_val_mape = val_total_mape / len(val_loader)
    avg_val_rmse = np.sqrt(avg_val_mse)
    val_r2 = r2_score(val_true, val_pred)
    loss_tracker["val"].append(avg_val_loss)

    # Update scheduler according to validation loss
    scheduler.step(avg_val_loss)

    # Manually print learning rate change (replaces verbose argument)
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr < last_lr:
        print(f"\nEpoch {epoch + 1}: learning rate decays to {current_lr:.6f}")
        last_lr = current_lr

    # Early‑stopping logic
    last_val_loss = float('inf')

    # Update best model checkpoint if current validation loss is lower
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_path)
        print(f"Updating best model, saved to {model_path}")

    # Increase patience counter when validation loss rises
    if avg_val_loss > last_val_loss:
        patience_counter += 1
        print(f"Early‑stop counter: {patience_counter}/{patience}")
    else:
        patience_counter = 0  # Reset counter if validation loss decreases

    # Trigger early‑stopping condition
    if patience_counter >= patience:
        print(f"\nValidation loss increases for {patience} consecutive epochs, early‑stop triggered!")
        break

    # Record validation loss of current epoch
    last_val_loss = avg_val_loss

    epoch_time = time.time() - start_time
    print(f"\nEpoch {epoch + 1}/{epochs} | Time: {epoch_time:.2f}s")
    print(
        f"Train | Loss: {avg_train_loss:.6f} | MAE: {avg_train_mae:.6f} | MSE: {avg_train_mse:.6f} | RMSE: {avg_train_rmse:.6f} | R²: {train_r2:.6f} | MAPE: {avg_train_mape:.6f}")
    print(
        f"Val   | Loss: {avg_val_loss:.6f} | MAE: {avg_val_mae:.6f} | MSE: {avg_val_mse:.6f} | RMSE: {avg_val_rmse:.6f} | R²: {val_r2:.6f} | MAPE: {avg_val_mape:.6f}")

# Plot training‑validation loss curve
plot_training_loss(loss_tracker["train"], filename=f"outputs/{model._get_name()}training_loss.png")
print("Training finish")
