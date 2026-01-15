# How to Trian & Win (The Advanced Strategy)

This guide executes the full **Winning Strategy** using the generated notebooks.

## Phase 1: Training the Ensemble
We need diversity. We will train **two** different models.

### Step 1: Train the DINOv2 Model
1.  Open `csiro_training.ipynb`.
2.  Ensure `CONFIG['backbone'] = 'vit_base_patch14_dinov2.lvd142m'`.
3.  Set `mixup_alpha = 0.4` (default).
4.  Run all cells.
5.  **Result**: Files named `vit_base..._fold0.pth`, etc.

### Step 2: Train the ConvNeXt Model
1.  In the same `csiro_training.ipynb`, change the config:
    ```python
    CONFIG['backbone'] = 'convnextv2_base.fcmae'
    ```
2.  Run all training cells again.
3.  **Result**: Files named `convnextv2..._fold0.pth`, etc.

**Why?** DINO sees "objects" (trees). ConvNeXt sees "textures" (grass density). You need both.

## Phase 2: The Inference (Submission)
1.  **Upload Weights**: Upload ALL the `.pth` files (both DINO and ConvNeXt versions) to a Kaggle Dataset.
2.  **Open** `csiro_inference.ipynb`.
3.  **Attach Dataset**: Add your dataset.
4.  **Verify TTA**:
    *   Ensure `CONFIG['use_tta'] = True`. This will run every image twice (Normal + Horizontal Flip) and average the results.
5.  **Run**:
    *   The notebook automatically detects the architecture from the filename (`vit` vs `convnext`).
    *   It loads ALL models found.
    *   It averages predictions from ALL models (Ensemble).
    *   It applies **Hierarchical Reconciliation**.

## Summary of "Winning" Features Active
| Feature | Status | Notebook |
| :--- | :--- | :--- |
| **Backbone Ensemble** | **Active** (DINO + ConvNeXt support) | Both |
| **MixUp / CutMix** | **Active** (alpha=0.4) | `training` |
| **TTA (Flip)** | **Active** | `inference` |
| **Tweedie Loss** | **Active** | `training` |
| **Reconciliation** | **Active** | Both |
| **Stratified Group CV**| **Active** | `training` |
