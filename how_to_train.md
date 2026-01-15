# How to Train & Submit (4 Ways to Win)

## 🛠️ SETUP: How to Get Offline Weights (Critical)
Since Kaggle has **No Internet** during submission, you must upload the model weights as a Dataset.

1.  **Download Locally**:
    Run the helper script on your machine (internet required):
    ```bash
    pip install timm torch
    python download_offline_weights.py
    ```
    This will create a `weights/` folder with `dinov3.pth`, `convnext22k.pth`, etc.

2.  **Upload to Kaggle**:
    *   Go to [Kaggle Datasets](https://www.kaggle.com/datasets).
    *   Click **New Dataset**.
    *   Drag & Drop the `weights` folder you just created.
    *   Name it: `csiro-sota-weights`.
    *   Create.

3.  **Attach to Notebook**:
    *   Open `csiro_winning_strategy.ipynb` on Kaggle.
    *   Right sidebar -> **Add Input**.
    *   Select "Your Datasets" -> `csiro-sota-weights`.

4.  **Update Config**:
    Copy the paths from the sidebar and paste them into the notebook's `CONFIG`:
    ```python
    'backbone_config': [
        {'name': 'vit_base_patch16_dinov3.lvd1689m', 'wgt': '/kaggle/input/csiro-sota-weights/vit_base_patch16_dinov3_lvd1689m.pth'},
        ...
    ]
    ```

---

## Method 1: The "Split" (Recommended)
**Best for**: Speed & Safety. Train local, infer Kaggle. Uses simple notebooks.

## Method 2: The "Ensemble One-Stop" (Internet)

## Method 3: The "Offline One-Stop" (Basic)

## Method 4: The "Grandmaster SOTA 2025" (Current Default)
**Best for**: Maximizing Score (Rank #1 Attempt).

1.  **Log1p Target Scaling**: Handles skewed data naturally.
2.  **SOTA "Future Proof" Quartet**:
    *   **DINOv3 (New)**: `vit_base_patch16_dinov3` (Sep 2025).
    *   **ConvNeXt V2 (22k)**: `convnextv2_base-22k` (Sep 2025).
    *   **SigLIP**: Google's Language-Image Alignment SOTA.
    *   **MaxViT**: Google's SOTA Hybrid.
3.  **Advanced Logic**: Metric Hacking, FiLM, Nelder-Mead.
