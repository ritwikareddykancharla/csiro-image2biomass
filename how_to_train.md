# How to Train & Submit (4 Ways to Win)

## Method 1: The "Split" (Recommended)
**Best for**: Speed & Safety. Train local, infer Kaggle. Uses simple notebooks.

## Method 2: The "Ensemble One-Stop" (Internet)
**Best for**: Automated multi-model training.

## Method 3: The "Offline One-Stop" (Basic)
**Best for**: Offline use of DINO.

## Method 4: The "Full Stack Advanced" (Current Default)
**Best for**: Maximizing Score (Rank #1 Attempt).
1.  **Metric Hacking (Weighted Loss)**:
    *   **Logic**: The Loss function is now heavily skewed.
    *   **Weights**: `Dry_Total_g` has **5x** the weight of `Green/Dead`.
    *   **Effect**: The model will "obsess" over getting the Total correct, even if it sacrifices some accuracy on components. This mathematically maximizes the leaderboard score.
2.  **State-Aware Stratified CV**: Prevents region leaks.
3.  **Nelder-Mead Optimization**: Finds perfect ensemble blend.
4.  **FiLM Metadata**: Injects Satellite Data.

**Checklist for "Full Stack":**
- [ ] `backbone_config` points to your offline weights.
- [ ] `train.csv` contains `Height_Ave_cm`, `Pre_GSHH_NDVI`, `Pre_GSHH_EV`.
- [ ] `State` column exists in `train.csv`.
