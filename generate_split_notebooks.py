
import json
import os

def create_notebook(cells):
    return {
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

# --- COMMON CODE BLOCKS ---
IMPORTS = """
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

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# --- WINNING STRATEGY CONFIGURATION ---
CONFIG = {
    'seed': 42,
    'img_size': 384,
    
    # ENSEMBLE STRATEGY: Train multiple backbones to average later
    # 1. 'vit_base_patch14_dinov2.lvd142m' (DINOv2 - Shape/Context)
    # 2. 'convnextv2_base.fcmae'           (ConvNeXt - Texture)
    'backbone': 'vit_base_patch14_dinov2.lvd142m', 
    
    'batch_size': 8,
    'epochs': 15, # Increased for Mixup
    'lr': 1e-4,
    'num_workers': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_folds': 5,
    'target_cols': ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g'],
    'train_csv': 'train.csv',
    'test_csv': 'test.csv',
    'img_dir': 'images/',
    
    # ADVANCED AUGMENTATION
    'mixup_alpha': 0.4,   # >0 enables MixUp
    'use_tta': True       # Test Time Augmentation
}

seed_everything(CONFIG['seed'])
"""

DATASET_CLASS = """
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
        if image is None: image = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
        else: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image=image)['image']
        if self.mode != 'test': return image, torch.tensor(self.labels[idx])
        return image

def get_image_path(filename, search_dir):
    p = os.path.join(search_dir, filename)
    if os.path.exists(p): return p
    p = os.path.join(search_dir, os.path.basename(filename))
    if os.path.exists(p): return p
    return filename

def get_transforms(img_size):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    return {
        'train': A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2), # Added for robustness
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]),
        'valid': A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]),
        # TTA Transforms (Flip + Scale)
        'tta_hflip': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(img_size, img_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    }
"""

MODEL_CLASS = """
class BiomassModel(nn.Module):
    def __init__(self, model_name, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.n_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3), # Increased dropout for regularization
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.head(self.backbone(x))
"""

POST_PROCESS = """
def hierarchical_reconciliation(preds):
    preds = np.maximum(preds, 0)
    green, dead, total = preds[:, 0], preds[:, 1], preds[:, 4]
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

# --- TRAINING SPECIFIC (With MixUp) ---
LOSS_METRIC_CV = """
# MIXUP HELPER
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class StateAwareStratifiedGroupKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    def split(self, X, y, groups):
        y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
        if isinstance(X, pd.DataFrame) and 'State' in X.columns:
            stratify_label = X['State'].astype(str) + "_" + y_bins.astype(str)
        else: stratify_label = y_bins
        return self.sgkf.split(X, stratify_label, groups=groups)

class TweedieLoss(nn.Module):
    def __init__(self, p=1.5, epsilon=1e-8):
        super().__init__()
        self.p = p; self.epsilon = epsilon
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
        
def weighted_r2_score(y_true, y_pred):
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    y_true_flat = y_true.flatten(); y_pred_flat = y_pred.flatten()
    n_samples = y_true.shape[0]; weights_flat = np.tile(weights, n_samples)
    y_weighted_mean = np.average(y_true_flat, weights=weights_flat)
    ss_res = np.sum(weights_flat * (y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum(weights_flat * (y_true_flat - y_weighted_mean) ** 2)
    if ss_tot < 1e-6: return 0.0
    return 1 - (ss_res / ss_tot)
"""

TRAINING_LOOP = """
def run_training():
    print(f"Starting TRAINING with Backbone: {CONFIG['backbone']}")
    print(f"MixUp Alpha: {CONFIG['mixup_alpha']}")
    
    if not os.path.exists(CONFIG['train_csv']): return
    df = pd.read_csv(CONFIG['train_csv'])
    groups = df['location_id'] if 'location_id' in df.columns else df.index
    cv = StateAwareStratifiedGroupKFold(n_splits=CONFIG['n_folds'])
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(df, df['Dry_Total_g'], groups=groups)):
        print(f"\\n{'='*20} FOLD {fold} {'='*20}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        train_loader = DataLoader(CSIRODataset(train_df, CONFIG['img_dir'], transform=get_transforms(CONFIG['img_size'])['train']), batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
        val_loader = DataLoader(CSIRODataset(val_df, CONFIG['img_dir'], transform=get_transforms(CONFIG['img_size'])['valid']), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
        
        model = BiomassModel(CONFIG['backbone'], pretrained=True).to(CONFIG['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
        criterion = BiomassLoss()
        
        best_score = -np.inf
        for epoch in range(CONFIG['epochs']):
            model.train()
            train_loss = 0.0
            for imgs, tgts in tqdm(train_loader, desc="Train", leave=False):
                imgs, tgts = imgs.to(CONFIG['device']), tgts.to(CONFIG['device'])
                optimizer.zero_grad()
                
                # Apply MixUp
                if CONFIG['mixup_alpha'] > 0:
                    imgs, y_a, y_b, lam = mixup_data(imgs, tgts, CONFIG['mixup_alpha'])
                    out = model(imgs)
                    loss = mixup_criterion(criterion, out, y_a, y_b, lam)
                else:
                    loss = criterion(model(imgs), tgts)
                    
                loss.backward(); optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            train_loss /= len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0; preds = []; targets = []
            with torch.no_grad():
                for imgs, tgts in tqdm(val_loader, desc="Val", leave=False):
                    imgs, tgts = imgs.to(CONFIG['device']), tgts.to(CONFIG['device'])
                    out = model(imgs)
                    val_loss += criterion(out, tgts).item() * imgs.size(0)
                    preds.append(out.cpu().numpy())
                    targets.append(tgts.cpu().numpy())
            
            val_loss /= len(val_loader.dataset)
            all_preds = hierarchical_reconciliation(np.vstack(preds))
            val_score = weighted_r2_score(np.vstack(targets), all_preds)
            
            print(f"Ep {epoch+1} | T: {train_loss:.4f} V: {val_loss:.4f} Score: {val_score:.4f}")
            if val_score > best_score:
                best_score = val_score
                # Save with backbone name in filename to allow multi-model ensembling later
                sanitized_backbone = CONFIG['backbone'].replace('.', '_')
                torch.save(model.state_dict(), f"{sanitized_backbone}_fold{fold}.pth")
                print(f"  >>> Saved {sanitized_backbone}_fold{fold}.pth")

if __name__ == "__main__":
    run_training()
"""

# --- INFERENCE SPECIFIC (With TTA) ---
INFERENCE_LOOP = """
def run_inference():
    print("Starting INFERENCE...")
    weights = glob.glob("*.pth") + glob.glob("/kaggle/input/*/*.pth")
    if not weights: print("No weights found!"); return
    print(f"Found {len(weights)} models: {weights}")

    if not os.path.exists(CONFIG['test_csv']): return
    test_df = pd.read_csv(CONFIG['test_csv'])
    
    # 1. Base Dataset
    test_ds = CSIRODataset(test_df, CONFIG['img_dir'], transform=get_transforms(CONFIG['img_size'])['valid'], mode='test')
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    # 2. TTA Dataset (Horizontal Flip)
    if CONFIG['use_tta']:
        print("Test Time Augmentation (TTA) Enabled.")
        tta_ds = CSIRODataset(test_df, CONFIG['img_dir'], transform=get_transforms(CONFIG['img_size'])['tta_hflip'], mode='test')
        tta_loader = DataLoader(tta_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    # 3. Load Models (Auto-detect backbone from filename if possible, else default)
    models = []
    for w in weights:
        # Determine architecture from filename or config
        if 'convnext' in w: arch = 'convnextv2_base.fcmae'
        else: arch = 'vit_base_patch14_dinov2.lvd142m' # Default
        
        m = BiomassModel(arch, pretrained=False)
        m.load_state_dict(torch.load(w, map_location=CONFIG['device']))
        m.to(CONFIG['device']).eval()
        models.append(m)
        
    final_preds = []
    with torch.no_grad():
        # Iterate Loaders (Base + TTA)
        loaders = [test_loader]
        if CONFIG['use_tta']: loaders.append(tta_loader)
        
        # We need to average across Loaders AND Models
        accumulated_preds = np.zeros((len(test_df), 5))
        
        for loader in loaders:
            loader_preds = []
            for images in tqdm(loader, desc="Infer"):
                images = images.to(CONFIG['device'])
                batch_preds = [m(images).cpu().numpy() for m in models]
                # Average models for this batch
                loader_preds.append(np.mean(batch_preds, axis=0))
            accumulated_preds += np.vstack(loader_preds)
            
        accumulated_preds /= len(loaders)
        final_preds = accumulated_preds
            
    all_preds = hierarchical_reconciliation(final_preds)
    submission = pd.DataFrame(all_preds, columns=CONFIG['target_cols'])
    if 'image_path' in test_df.columns: submission.insert(0, 'image_path', test_df['image_path'])
    submission.to_csv('submission.csv', index=False)
    print("Saved submission.csv")

if __name__ == "__main__":
    run_inference()
"""

def generate():
    # 1. Training Notebook
    train_cells = [
        markdown_cell("# CSIRO Image2Biomass: TRAINING Notebook (Winning Strategy)\n- **MixUp**: Enabled.\n- **Backbone**: Switchable (DINOv2 / ConvNeXt).\n- **Save Format**: includes backbone name."),
        code_cell(IMPORTS),
        code_cell(DATASET_CLASS),
        code_cell(LOSS_METRIC_CV),
        code_cell(POST_PROCESS), 
        code_cell(MODEL_CLASS),
        code_cell(TRAINING_LOOP)
    ]
    with open('csiro_training.ipynb', 'w') as f:
        json.dump(create_notebook(train_cells), f, indent=1)
        
    # 2. Inference Notebook
    infer_cells = [
        markdown_cell("# CSIRO Image2Biomass: INFERENCE Notebook (Winning Strategy)\n- **Ensemble**: Auto-detects mix of DINO and ConvNeXt weights.\n- **TTA**: Flip Augmentation enabled."),
        code_cell(IMPORTS),
        code_cell(DATASET_CLASS),
        code_cell(MODEL_CLASS),
        code_cell(POST_PROCESS),
        code_cell(INFERENCE_LOOP)
    ]
    with open('csiro_inference.ipynb', 'w') as f:
        json.dump(create_notebook(infer_cells), f, indent=1)
        
    print("Generated: csiro_training.ipynb & csiro_inference.ipynb")

if __name__ == "__main__":
    generate()
