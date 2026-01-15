
import json
import os

def create_winning_notebook():
    # Helper to create code cell structure
    def code_cell(source_code):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_code.split("\n")]
        }

    def markdown_cell(source_text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_text.split("\n")]
        }

    cells = []
    
    # --- CELL 1: Main Header ---
    cells.append(markdown_cell("""
# CSIRO Image2Biomass: Unified Strategy Notebook

**Goal**: Achieve Rank #1 using DINOv2 / ConvNeXt V2.
**Mode**: This notebook is **Adaptive**.
1.  **If Weights Exist**: It detects saved model files (in `/kaggle/input` or local), skips training, and runs **Inference**.
2.  **If No Weights**: It downloads the backbone (Internet Required initially), runs **Training**, and saves weights.

This allows you to use **ONE** notebook for both developing (training) and submitting (inference).
"""))

    # --- CELL 2: Imports & Config ---
    cells.append(markdown_cell("## 1. Configuration & Imports"))
    code_imports = """
import os
import sys
import glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm

# --- CONFIGURATION ---
CONFIG = {
    'seed': 42,
    'img_size': 384,         
    
    # Model Choice
    'backbone': 'vit_base_patch14_dinov2.lvd142m', 
    # 'backbone': 'convnextv2_base.fcmae',
    
    'batch_size': 8,          
    'epochs': 10,
    'lr': 1e-4,
    'num_workers': 0,         
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_folds': 5,
    'target_cols': ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g'],
    
    # Paths (Auto-dectected usually, but settable)
    'train_csv': 'train.csv',
    'test_csv': 'test.csv',
    'img_dir': 'images/', # Or /kaggle/input/csiro-biomass/train
    'weights_dir': '.',   # Where to look for/save weights
}

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])
print(f"Using device: {CONFIG['device']}")
"""
    cells.append(code_cell(code_imports))

    # --- CELL 3: Metrics ---
    code_metrics = """
def weighted_r2_score(y_true, y_pred):
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    n_samples = y_true.shape[0]
    weights_flat = np.tile(weights, n_samples)
    y_weighted_mean = np.average(y_true_flat, weights=weights_flat)
    ss_res = np.sum(weights_flat * (y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum(weights_flat * (y_true_flat - y_weighted_mean) ** 2)
    if ss_tot < 1e-6: return 0.0
    return 1 - (ss_res / ss_tot)
"""
    cells.append(code_cell(code_metrics))

    # --- CELL 4: CV Strategy ---
    code_cv = """
class StateAwareStratifiedGroupKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups):
        # Bin total biomass for stratification
        y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
        if isinstance(X, pd.DataFrame) and 'State' in X.columns:
            stratify_label = X['State'].astype(str) + "_" + y_bins.astype(str)
        else:
            stratify_label = y_bins
        return self.sgkf.split(X, stratify_label, groups=groups)
"""
    cells.append(code_cell(code_cv))

    # --- CELL 5: Losses ---
    code_loss = """
class TweedieLoss(nn.Module):
    def __init__(self, p=1.5, epsilon=1e-8):
        super().__init__()
        self.p = p 
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = F.softplus(y_pred) + self.epsilon
        a = y_true * torch.pow(y_pred, 1 - self.p) / (1 - self.p)
        b = torch.pow(y_pred, 2 - self.p) / (2 - self.p)
        loss = -a + b
        return torch.mean(loss)

class BiomassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tweedie = TweedieLoss(p=1.5)
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        return 0.5 * self.tweedie(y_pred, y_true) + 0.5 * self.mse(y_pred, y_true)
"""
    cells.append(code_cell(code_loss))

    # --- CELL 6: Post-Processing ---
    code_pp = """
def hierarchical_reconciliation(preds):
    # Enforce: Green + Dead = Total AND Green >= Clover
    preds = np.maximum(preds, 0)
    green = preds[:, 0]
    dead = preds[:, 1]
    total = preds[:, 4]
    
    sum_comp = green + dead
    mask = sum_comp > 1e-6
    scale_factor = np.ones_like(total)
    scale_factor[mask] = total[mask] / sum_comp[mask]
    
    new_preds = preds.copy()
    new_preds[:, 0] = green * scale_factor
    new_preds[:, 1] = dead * scale_factor
    new_preds[:, 2] = np.minimum(preds[:, 2], new_preds[:, 0]) 
    return new_preds
"""
    cells.append(code_cell(code_pp))

    # --- CELL 7: Dataset ---
    code_data = """
class CSIRODataset(Dataset):
    def __init__(self, df, img_dir, transform=None, mode='train'):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        col_name = 'image_path' if 'image_path' in df.columns else df.columns[0]
        self.file_names = df[col_name].values
        if self.mode != 'test':
            self.labels = df[CONFIG['target_cols']].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        full_path = get_image_path(file_path, self.img_dir)
            
        image = cv2.imread(full_path)
        if image is None: # Safety
            image = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']
            
        if self.mode != 'test':
            return image, torch.tensor(self.labels[idx])
        return image

def get_image_path(filename, search_dir):
    # Robust path finding (handles flat dirs or subdirs)
    p = os.path.join(search_dir, filename)
    if os.path.exists(p): return p
    # Try just basename in search_dir
    p = os.path.join(search_dir, os.path.basename(filename))
    if os.path.exists(p): return p
    return filename # Fallback

def get_transforms(img_size):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    return {
        'train': A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]),
        'valid': A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    }
"""
    cells.append(code_cell(code_data))

    # --- CELL 8: Model ---
    code_model = """
class BiomassModel(nn.Module):
    def __init__(self, model_name, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.n_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.head(self.backbone(x))
"""
    cells.append(code_cell(code_model))

    # --- CELL 9: Training Loop ---
    code_train_loop = """
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_list, targets_list = [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Valid", leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
    
    all_preds = hierarchical_reconciliation(np.vstack(preds_list))
    all_targets = np.vstack(targets_list)
    score = weighted_r2_score(all_targets, all_preds)
    return running_loss / len(loader.dataset), score, all_preds
"""
    cells.append(code_cell(code_train_loop))

    # --- CELL 10: Execution Functions (Train & Infer) ---
    code_execution = """
def run_training():
    print("Starting TRAINING Pipeline...")
    if not os.path.exists(CONFIG['train_csv']):
        print(f"Error: {CONFIG['train_csv']} not found.")
        return

    df = pd.read_csv(CONFIG['train_csv'])
    groups = df['location_id'] if 'location_id' in df.columns else df.index
    
    cv = StateAwareStratifiedGroupKFold(n_splits=CONFIG['n_folds'])
    oof_preds = np.zeros((len(df), 5))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(df, df['Dry_Total_g'], groups=groups)):
        print(f"\\n{'='*20} FOLD {fold} {'='*20}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        train_loader = DataLoader(
            CSIRODataset(train_df, CONFIG['img_dir'], transform=get_transforms(CONFIG['img_size'])['train']),
            batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers']
        )
        val_loader = DataLoader(
            CSIRODataset(val_df, CONFIG['img_dir'], transform=get_transforms(CONFIG['img_size'])['valid']),
            batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers']
        )
        
        model = BiomassModel(CONFIG['backbone'], pretrained=True).to(CONFIG['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
        criterion = BiomassLoss()
        
        best_score = -np.inf
        for epoch in range(CONFIG['epochs']):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
            val_loss, val_score, _ = validate(model, val_loader, criterion, CONFIG['device'])
            print(f"Ep {epoch+1} | T: {train_loss:.4f} V: {val_loss:.4f} Score: {val_score:.4f}")
            
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), f"model_fold{fold}.pth")
                print(f"  >>> Saved model_fold{fold}.pth")

def run_inference(weight_files):
    print("Starting INFERENCE Pipeline...")
    if not os.path.exists(CONFIG['test_csv']):
        print("Test CSV not found. Skipping inference.")
        return

    test_df = pd.read_csv(CONFIG['test_csv'])
    test_ds = CSIRODataset(test_df, CONFIG['img_dir'], transform=get_transforms(CONFIG['img_size'])['valid'], mode='test')
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    # Load all models for bagging
    models = []
    for w in weight_files:
        print(f"Loading weights: {w}")
        m = BiomassModel(CONFIG['backbone'], pretrained=False) # Important: pretrained=False for inference
        m.load_state_dict(torch.load(w, map_location=CONFIG['device']))
        m.to(CONFIG['device'])
        m.eval()
        models.append(m)
        
    final_preds = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Infer"):
            images = images.to(CONFIG['device'])
            batch_preds = []
            for m in models:
                batch_preds.append(m(images).cpu().numpy())
            # Average across models (Bagging)
            avg_batch = np.mean(batch_preds, axis=0)
            final_preds.append(avg_batch)
            
    all_preds = np.vstack(final_preds)
    all_preds = hierarchical_reconciliation(all_preds) # Apply post-process constraints
    
    # Create Submission
    submission = pd.DataFrame(all_preds, columns=CONFIG['target_cols'])
    # Add ID or whatever format/structure is needed. 0.69 nb melts it. 
    # For now, we save raw format and let user inspect.
    submission.to_csv('submission.csv', index=False)
    print("Saved submission.csv")
"""
    cells.append(code_cell(code_execution))

    # --- CELL 11: Unified Main ---
    code_main = """
if __name__ == "__main__":
    # 1. Look for existing weights
    # We check typical places: current dir, or /kaggle/input
    possible_weights = glob.glob("*.pth") + glob.glob("/kaggle/input/*/model_fold*.pth")
    
    if len(possible_weights) > 0:
        print(f"Found {len(possible_weights)} weights! Switching to INFERENCE mode.")
        run_inference(possible_weights)
    else:
        print("No weights found. Switching to TRAINING mode.")
        # Note: In Kaggle Submit environment, this might fail if no internet.
        # But this is what the user asked for: Single Notebook Logic.
        run_training()
"""
    cells.append(code_cell(code_main))

    # --- SAVE ---
    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open('csiro_winning_strategy.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=1)
    print("Notebook refined: csiro_winning_strategy.ipynb (Unified Train/Infer)")

if __name__ == "__main__":
    create_winning_notebook()
