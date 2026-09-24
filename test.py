import random
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from utils.Prediction_analysis import prediction_analysis
from utils.dataloader import CustomDataset
from config import data_path, used_model


def set_all_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Initialize random seed for each dataloader worker"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Build dataset instance
root_dir = data_path
print(f"Dataset directory: {root_dir}")
dataset = CustomDataset(root_dir)
index_dir = "data_indices"
test_idx = np.load(os.path.join(index_dir, "test_idx.npy"))

# Build test set dataloader
test_dataset = Subset(dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Number of test samples: {len(test_idx)}")

# Initialize model and load trained weights
model = used_model()
model_path = os.path.join("saved_models", f"best_{model._get_name()}_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Model: {model._get_name()}")
print(f"Model weight path: {model_path}")

# Run model inference on test set
true_values = []
pred_values = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device).float(), yb.to(device).float()
        preds = model(xb)
        true_values.extend(yb.cpu().numpy().flatten())
        pred_values.extend(preds.cpu().numpy().flatten())

# Calculate evaluation metrics: MAE, MSE, RMSE, R², MAPE
mae = mean_absolute_error(true_values, pred_values)
mse = mean_squared_error(true_values, pred_values)
rmse = np.sqrt(mse)
r2 = r2_score(true_values, pred_values)
mape = mean_absolute_percentage_error(true_values, pred_values)

print(f"Test | MAE: {mae:.6f} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f} | MAPE: {mape:.6f}")

# Generate prediction‑analysis visualization figures
prediction_analysis(
    true_values=true_values,
    pred_values=pred_values,
    model_name=model._get_name(),
    mae=mae,
    mse=mse,
    r2=r2,
    mape=mape,
    save_dir="outputs"
)
