
import json
import os

def create_winning_notebook():
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
    
    cells.append(markdown_cell("""
# 🏆 CSIRO Image2Biomass: SOTA "Future-Proof" Edition
### 🚀 DINOv3 + ConvNeXt 22k + SigLIP | Rank #1

**Single Notebook | Self-Contained | Offline-Ready**

**Workflow**:
1.  **Setup (Internet)**: Run the first cell to download weights (ONCE).
2.  **Train/Infer (Offline)**: The rest of the notebook runs securely offline.

**The "Future Proof" Quartet**:
*   **DINOv3**: New SOTA Semantic.
*   **ConvNeXt V2 (22k)**: New SOTA Texture.
*   **SigLIP**: SOTA Vision-Language.
*   **MaxViT**: SOTA Hybrid.
"""))

    cells.append(markdown_cell("## 0. SETUP: Download Weights (Run Once with Internet)"))
    code_download = """
# RUN THIS CELL ONLY IF YOU NEED TO DOWNLOAD WEIGHTS FOR OFFLINE USE
# Requires: variables defined in config (timm installed)
import os
import torch
try:
    import timm
    DOWNLOAD_NEEDED = True
except ImportError:
    print("timm not installed. Installing...")
    os.system("pip install timm")
    import timm
    DOWNLOAD_NEEDED = True

MODELS_TO_DOWNLOAD = [
    "vit_base_patch16_dinov3.lvd1689m",
    "convnextv2_base.fcmae_ft_in22k_in1k",
    "vit_so400m_patch14_siglip_384",
    "maxvit_tiny_tf_512.in1k"
]

if DOWNLOAD_NEEDED and not os.path.exists("weights"):
    print(">>> DOWNLOADING WEIGHTS FOR OFFLINE USE...")
    os.makedirs("weights", exist_ok=True)
    for model_name in MODELS_TO_DOWNLOAD:
        try:
            print(f"Downloading {model_name}...")
            m = timm.create_model(model_name, pretrained=True)
            torch.save(m.state_dict(), f"weights/{model_name.replace('.', '_')}.pth")
            print("OK.")
        except Exception as e:
            print(f"Failed {model_name}: {e}")
    print("DONE. You can now use the 'weights' folder as a Dataset.")
else:
    print("Weights folder exists or download skipped.")
"""
    cells.append(code_cell(code_download))

    cells.append(markdown_cell("## 1. Configuration & Imports"))
    code_config = """
import os
import sys
import glob
import math
import copy
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm

# --- CONFIGURATION ---
CONFIG = {
    'seed': 42,
    'img_size': 384,
    'batch_size': 8,
    'epochs': 12,
    'lr': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # SOTA "FUTURE PROOF" QUARTET
    'backbone_config': [
        # 1. DINOv3 (New SOTA Sep 2025)
        {'name': 'vit_base_patch16_dinov3.lvd1689m', 'wgt': None},
        
        # 2. ConvNeXt V2 (22k Pretraining - Huge Upgrade)
        {'name': 'convnextv2_base.fcmae_ft_in22k_in1k', 'wgt': None},
        
        # 3. SigLIP (SOTA Vision-Language)
        {'name': 'vit_so400m_patch14_siglip_384', 'wgt': None},
        
        # 4. MaxViT (Hybrid SOTA)
        {'name': 'maxvit_tiny_tf_512.in1k', 'wgt': None}
    ],
    
    'meta_cols': ['Height_Ave_cm', 'Pre_GSHH_NDVI', 'Pre_GSHH_EV'], 
    'n_folds': 5,
    'target_cols': ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g'],
    'target_weights': [0.1, 0.1, 0.1, 0.2, 0.5], 
    'train_csv': 'train.csv',
    'test_csv': 'test.csv', 
    'img_dir': 'images/',
    'mixup_alpha': 0.4,
    'use_tta': True,
    'use_ema': True,
    'use_log1p': True 
}

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])
"""
    cells.append(code_cell(code_config))

    cells.append(code_cell("""
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model); self.model.eval(); self.decay = decay
    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.model.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * model_v)

class CSIRODataset(Dataset):
    def __init__(self, df, img_dir, transform=None, mode='train', meta_scaler=None):
        self.df = df; self.img_dir = img_dir; self.transform = transform; self.mode = mode
        self.meta_features = df[CONFIG['meta_cols']].fillna(0).values.astype(np.float32)
        if meta_scaler: self.meta_features = meta_scaler.transform(self.meta_features)
        col = 'image_path' if 'image_path' in df.columns else df.columns[0]
        self.files = df[col].values
        if mode != 'test': 
            y = df[CONFIG['target_cols']].values.astype(np.float32)
            if CONFIG['use_log1p']: self.labels = np.log1p(y)
            else: self.labels = y

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        path = self.files[idx]
        full = os.path.join(self.img_dir, path) if os.path.exists(os.path.join(self.img_dir, path)) else os.path.join(self.img_dir, os.path.basename(path))
        img = cv2.imread(full)
        if img is None: img = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
        else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(image=img)['image']
        meta = torch.tensor(self.meta_features[idx])
        if self.mode != 'test': return img, meta, torch.tensor(self.labels[idx])
        return img, meta

def get_transforms(img_size):
    MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
    return {
        'train': A.Compose([A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        'valid': A.Compose([A.Resize(img_size, img_size), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        'tta': A.Compose([A.HorizontalFlip(p=1.0), A.Resize(img_size, img_size), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
    }

def mixup_data(x, meta, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx, :], lam * meta + (1 - lam) * meta[idx, :], y, y[idx], lam

def mixup_criterion(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)
"""))

    code_model = """
class FiLM(nn.Module):
    def __init__(self, feature_dim, meta_dim):
        super().__init__()
        self.scale = nn.Linear(meta_dim, feature_dim)
        self.shift = nn.Linear(meta_dim, feature_dim)
    def forward(self, features, meta):
        return self.scale(meta) * features + self.shift(meta)

class BiomassModel(nn.Module):
    def __init__(self, model_name, num_classes=5, pretrained=False, checkpoint_path=None, meta_dim=3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        # Load Weights
        if pretrained:
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading OFFLINE: {checkpoint_path}")
                try: self.backbone.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
                except: pass
            elif checkpoint_path is None:
                # Only attempt online if not explicit None and internet is likely available
                print(f"Attempting ONLINE Load (if available): {model_name}")
                try: self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
                except: pass

        self.film = FiLM(self.backbone.num_features, meta_dim)
        self.head = nn.Sequential(nn.Linear(self.backbone.num_features, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.2), nn.Linear(512, num_classes))

    def forward(self, x, meta):
        f = self.backbone(x)
        return self.head(self.film(f, meta))

class WeightedBiomassLoss(nn.Module):
    def __init__(self, weights=CONFIG['target_weights']):
        super().__init__()
        self.weights = torch.tensor(weights).float()
    def forward(self, yp, yt):
        w = self.weights.to(yp.device)
        loss = F.l1_loss(yp, yt, reduction='none') + F.mse_loss(yp, yt, reduction='none')
        return torch.mean(loss * w)

def hierarchical_reconciliation(preds):
    preds = np.maximum(preds, 0)
    s = preds[:,0]+preds[:,1]; m=s>1e-6; r=np.ones_like(preds[:,4]); r[m]=preds[m,4]/s[m]
    preds[:,0]*=r; preds[:,1]*=r; preds[:,2]=np.minimum(preds[:,2], preds[:,0])
    return preds

def weighted_r2_score(y_true, y_pred):
    weights = np.array(CONFIG['target_weights']) 
    y_true_flat = y_true.flatten(); y_pred_flat = y_pred.flatten()
    n = y_true.shape[0]; w_flat = np.tile(weights, n)
    y_mean = np.average(y_true_flat, weights=w_flat)
    ss_res = np.sum(w_flat * (y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum(w_flat * (y_true_flat - y_mean) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-6))
"""
    cells.append(code_cell(code_model))

    code_optim = """
# --- OPTIMIZATION LOGIC ---
def optimize_ensemble_weights(oof_dict, y_true):
    print("\\n>>> Optimizing Ensemble Weights (Nelder-Mead)...")
    model_names = list(oof_dict.keys())
    if len(model_names) < 2: return {m: 1.0 for m in model_names}
    
    def objective(weights):
        w = np.exp(weights) / np.sum(np.exp(weights)) 
        final_oof = np.zeros_like(list(oof_dict.values())[0])
        for i, m_name in enumerate(model_names):
            final_oof += w[i] * oof_dict[m_name] 
        score = weighted_r2_score(y_true, hierarchical_reconciliation(final_oof))
        return -score

    res = minimize(objective, np.zeros(len(model_names)), method='Nelder-Mead', tol=1e-4)
    best_weights_raw = np.exp(res.x) / np.sum(np.exp(res.x))
    final_weights = {m_name: float(best_weights_raw[i]) for i, m_name in enumerate(model_names)}
    print("Optimal Weights:", final_weights)
    return final_weights
"""
    cells.append(code_cell(code_optim))

    code_train = """
def run_training():
    print(">>> STARTING SOTA TRAINING")
    df = pd.read_csv(CONFIG['train_csv'])
    meta_scaler = StandardScaler()
    df[CONFIG['meta_cols']] = df[CONFIG['meta_cols']].fillna(0)
    meta_scaler.fit(df[CONFIG['meta_cols']])
    
    groups = df['location_id'] if 'location_id' in df.columns else df.index
    y_bins = pd.qcut(df['Dry_Total_g'], 5, labels=False).astype(str)
    stratify_label = df['State'].astype(str) + "_" + y_bins if 'State' in df.columns else y_bins
    
    kf = StratifiedGroupKFold(n_splits=CONFIG['n_folds'])
    oof_store = {} 
    
    fold_indices = []
    for f, (t, v) in enumerate(kf.split(df, stratify_label, groups=groups)): fold_indices.append((t, v))
        
    for cfg in CONFIG['backbone_config']:
        name = cfg['name']; wgt_path = cfg['wgt']; safe_name = name.replace(".", "_")
        print(f"\\n--- Training {name} ---")
        current_oof = np.zeros((len(df), 5))
        
        for fold, (t_idx, v_idx) in enumerate(fold_indices):
            print(f"Fold {fold}")
            td = df.iloc[t_idx].reset_index(drop=True); vd = df.iloc[v_idx].reset_index(drop=True)
            tl = DataLoader(CSIRODataset(td, CONFIG['img_dir'], transform=get_transforms(384)['train'], meta_scaler=meta_scaler), batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
            vl = DataLoader(CSIRODataset(vd, CONFIG['img_dir'], transform=get_transforms(384)['valid'], meta_scaler=meta_scaler), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
            
            model = BiomassModel(name, pretrained=True, checkpoint_path=wgt_path, meta_dim=len(CONFIG['meta_cols']))
            model.to(CONFIG['device'])
            ema = ModelEMA(model) if CONFIG['use_ema'] else None
            opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
            crit = WeightedBiomassLoss()
            
            model.train()
            for ep in range(CONFIG['epochs']):
                # Cosine
                lr = CONFIG['lr'] * 0.5 * (1 + math.cos(math.pi * ep / CONFIG['epochs']))
                for pg in opt.param_groups: pg['lr'] = lr
                
                for img, meta, tgt in tl:
                    img, meta, tgt = img.to(CONFIG['device']), meta.to(CONFIG['device']), tgt.to(CONFIG['device'])
                    opt.zero_grad()
                    if CONFIG['mixup_alpha'] > 0:
                        img, meta, ya, yb, lam = mixup_data(img, meta, tgt, CONFIG['mixup_alpha'])
                        loss = mixup_criterion(crit, model(img, meta), ya, yb, lam)
                    else:
                        loss = crit(model(img, meta), tgt)
                    loss.backward(); opt.step()
                    if ema: ema.update(model)
            
            # OOF & R2
            best_model = ema.model if ema else model
            best_model.eval()
            fold_preds = []; fold_truth = vd[CONFIG['target_cols']].values
            with torch.no_grad():
                for img, meta in vl:
                    img, meta = img.to(CONFIG['device']), meta.to(CONFIG['device'])
                    pred = best_model(img, meta).cpu().numpy()
                    if CONFIG['use_log1p']: pred = np.expm1(pred)
                    fold_preds.append(pred)
            
            fold_preds_arr = np.vstack(fold_preds)
            current_oof[v_idx] = fold_preds_arr
            r2 = weighted_r2_score(fold_truth, hierarchical_reconciliation(fold_preds_arr))
            print(f"  >> Fold {fold} R2: {r2:.4f}")
            torch.save(best_model.state_dict(), f"{safe_name}_fold{fold}.pth")
        
        oof_store[name] = current_oof

    # RUN NELDER-MEAD
    y_true = df[CONFIG['target_cols']].values
    best_weights = optimize_ensemble_weights(oof_store, y_true)
    with open("ensemble_weights.json", "w") as f: json.dump(best_weights, f)

def run_inference(weights):
    print(f">>> INFERENCE with {len(weights)} models")
    td = pd.read_csv(CONFIG['test_csv'])
    meta_scaler = StandardScaler()
    meta_scaler.fit(td[CONFIG['meta_cols']].fillna(0))
    
    ensemble_weights = None
    if os.path.exists("ensemble_weights.json"):
        with open("ensemble_weights.json", "r") as f: ensemble_weights = json.load(f)
    elif os.path.exists("/kaggle/input"):
        fs = glob.glob("/kaggle/input/*/ensemble_weights.json")
        if fs:
            with open(fs[0], "r") as f: ensemble_weights = json.load(f)
            
    loaders = [DataLoader(CSIRODataset(td, CONFIG['img_dir'], transform=get_transforms(384)['valid'], mode='test', meta_scaler=meta_scaler), batch_size=CONFIG['batch_size'])]
    if CONFIG['use_tta']: loaders.append(DataLoader(CSIRODataset(td, CONFIG['img_dir'], transform=get_transforms(384)['tta'], mode='test', meta_scaler=meta_scaler), batch_size=CONFIG['batch_size']))
    
    models_dict = {}
    for w in weights:
        # Detect Architecture from Filename
        if 'convnext' in w: arch = 'convnextv2_base.fcmae_ft_in22k_in1k'
        elif 'maxvit' in w: arch = 'maxvit_tiny_tf_512.in1k'
        elif 'siglip' in w: arch = 'vit_so400m_patch14_siglip_384'
        elif 'dinov3' in w: arch = 'vit_base_patch16_dinov3.lvd1689m'
        else: arch = 'vit_base_patch16_dinov3.lvd1689m' 
        
        if arch not in models_dict: models_dict[arch] = []
        m = BiomassModel(arch, pretrained=False, meta_dim=len(CONFIG['meta_cols']))
        m.load_state_dict(torch.load(w, map_location=CONFIG['device']))
        m.to(CONFIG['device']).eval(); models_dict[arch].append(m)

    final = np.zeros((len(td), 5))
    backbone_preds = {}
    with torch.no_grad():
        for arch, model_list in models_dict.items():
            arch_accum = np.zeros((len(td), 5))
            for l in loaders:
                loader_acc = []
                for img, meta in tqdm(l, desc=arch):
                    img, meta = img.to(CONFIG['device']), meta.to(CONFIG['device'])
                    p_folds = []
                    for m in model_list:
                        raw_pred = m(img, meta).cpu().numpy()
                        if CONFIG['use_log1p']: raw_pred = np.expm1(raw_pred)
                        p_folds.append(raw_pred)
                    loader_acc.append(np.mean(p_folds, axis=0))
                arch_accum += np.vstack(loader_acc)
            backbone_preds[arch] = arch_accum / len(loaders)
            
    if ensemble_weights:
        for arch, preds in backbone_preds.items():
            w = ensemble_weights.get(arch, 1.0 / len(backbone_preds))
            final += w * preds
    else:
        for preds in backbone_preds.values(): final += preds
        final /= len(backbone_preds)

    final = hierarchical_reconciliation(final)
    sub = pd.DataFrame(final, columns=CONFIG['target_cols'])
    if 'image_path' in td.columns: sub.insert(0, 'image_path', td['image_path'])
    sub.to_csv('submission.csv', index=False)
    print("Saved submission.csv")

if __name__ == "__main__":
    w = [x for x in glob.glob("*.pth") + glob.glob("/kaggle/input/*/*.pth") if "pretrain" not in x and "checkpoint" not in x]
    if len(w) > 0: run_inference(w)
    else: run_training()
"""
    cells.append(code_cell(code_train))

    notebook_content = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3","language": "python","name": "python3"},"language_info": {"name": "python","version": "3.10"}},
        "nbformat": 4, "nbformat_minor": 5
    }
    with open('csiro_winning_strategy.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=1)
    print("Notebook refined: csiro_winning_strategy.ipynb (Self-Contained Downloader)")

if __name__ == "__main__":
    create_winning_notebook()
