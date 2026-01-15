# Ultimate Winning Strategy: From 0.79 to #1 in CSIRO Image2Biomass

To secure the top spot on the leaderboard, you must move beyond standard ensembles and exploit every nuance of the data and evaluation metric. This guide outlines a structured approach to maximize your score within the remaining 14 days.

## Table of Contents
1. [The Holy Grail: Cross-Validation Consistency](#1-the-holy-grail-cross-validation-consistency)
2. [Metric Hacking: Weighted R² Optimization](#2-metric-hacking-weighted-r-optimization)
3. [Model Architecture: The "Texture & Context" Ensemble](#3-model-architecture-the-texture--context-ensemble)
4. [Advanced Feature Engineering & Injection](#4-advanced-feature-engineering--injection)
5. [Loss Function Engineering](#5-loss-function-engineering)
6. [The "Golden" Post-Processing](#6-the-golden-post-processing)
7. [Training Stability & Tricks](#7-training-stability--tricks)
8. [Pseudo-Labeling (The Finisher)](#8-pseudo-labeling-the-finisher)
9. [14-Day Execution Plan](#9-14-day-execution-plan)

---

## 1. The Holy Grail: Cross-Validation Consistency
If your local CV doesn't match the LB, you are flying blind.
- **State-Aware Stratified Group K-Fold**: The data has a strong location (`State`) bias. Simple random split leaks information.
    - *Strategy*: Use `GroupKFold` on a location identifier (if available or derived from coordinates). If not, ensure `State` distribution in folds exactly matches the whole dataset.
    - *Binning*: Stratify not just by `State` but also by binned `Dry_Total_g` to ensure high-biomass samples are evenly distributed.

## 2. Metric Hacking: Weighted R² Optimization
The evaluation metric is a **Weighted R²** where each row has a weight based on the target type:
- **Weights**:
    - `Dry_Total_g`: **0.5** (This is HUGE. Half the score depends on this one column.)
    - `GDM_g`: **0.2**
    - `Dry_Green/Dead/Clover_g`: **0.1** each
- **Strategy**:
    - **Prioritize Total Biomass**: Your model *must* nail `Dry_Total_g`. During training, weight the loss for the `Total` head by 5x compared to the components.
    - **Custom Metric Callback**: Implement the exact weighted R² metric as a callback to monitor the *real* score during training, not just MSE.

## 3. Model Architecture: The "Texture & Context" Ensemble
Biomass estimation relies on "Greenness" (easy) and "Volume/Texture" (hard).
- **Core Ensemble Concept**:
    1.  **DINOv2 (ViT-g/14)**: Best for semantic understanding and global context.
    2.  **ConvNeXt V2 (Atto/Nano - Huge)**: CNNs are superior at capturing high-frequency texture details (critical for differentiating dead grass from dirt).
    3.  **MaxViT / CoAtNet**: Hybrids that capture both.
- **Input Strategy**:
    - **Multi-Resolution**: Train separate models on 224x224 (global structure) and 512x512 or higher (texture detail).
    - **2.5D Input**: Inject `Pre_GSHH_NDVI` into the MLP head directly or as a 4th channel.

## 4. Advanced Feature Engineering & Injection
Don't just rely on images. The tabular metadata is powerful but needs careful handling.
- **Sensor Fusion**: Instead of just concatenating metadata at the end, use **FiLM (Feature-wise Linear Modulation)** layers.
    - *How*: Use `Height_Ave_cm`, `Pre_GSHH_NDVI`, `State`, `Seasonality` to predict scale (`gamma`) and shift (`beta`) parameters that modulate the image feature maps before the pooling layer. This conditions the image extraction on the physical constraints.
- **Cyclical Time**: Use `sin` and `cos` of `DayOfYear` to capture growth cycles perfectly.

## 5. Loss Function Engineering
Standard MSE is suboptimal because of the zero-inflated and heavy-tailed nature of the targets.
- **Compound Loss**: `Loss = 0.5 * Tweedie(p=1.5) + 0.5 * MSE`
    - *Why*: Tweedie handles the exact zeros in `Dry_Clover_g` and `Dry_Dead_g` naturally.
- **Asymmetric Loss**: If the metric penalizes under-prediction more than over-prediction (or vice-versa), adjust your loss.
- **Log-Cosh**: Approximates Huber loss, robust to outliers (extremely high biomass samples).

## 6. The "Golden" Post-Processing
This is often the difference between Top 10 and #1.
- **Hierarchical Reconciliation (Optimization)**:
    - `Dry_Total_g` *must* approximately equal `Dry_Green_g + Dry_Dead_g`.
    - *Strategy*: Predict all targets and run a **Quadratic Programming (QP)** solver or simple linear projection for every sample to find the closest values that satisfy:
        1. `Green + Dead = Total`
        2. `Green >= Clover`
        3. `All >= 0`
- **Nelder-Mead Ensemble Optimization**:
    - Optimize the weights of your ensemble members (e.g., `0.3*DINO + 0.4*ConvNext + ...`) directly on the OOF predictions using `scipy.optimize.minimize` with the specific Weighted R² metric.

## 7. Training Stability & Tricks
- **EMA (Exponential Moving Average)**: Use Model EMA during training. It stabilizes weights and acts as a "free" ensemble.
- **Stochastic Depth**: High drop-path rates (0.2-0.4) for large ViT models to prevent overfitting on the small dataset.
- **Slide & Fuse**: For high-res inference, take random crops during training but use **Sliding Window** inference with 50% overlap during testing.

## 8. Pseudo-Labeling (The Finisher)
Once you have a strong model (0.79+):
1. Predict on the *entire* Test set.
2. Select samples where your ensemble members have **low variance** (high agreement).
3. Add these "confident" test samples to your training data.
4. Retrain your best single models.

## 9. 14-Day Execution Plan
With two weeks left, focus on high-ROI activities:
- **Days 1-3**: Fix CV strategy (GroupKFold) and implement Weighted R² metric monitoring. Train a baseline ConvNeXt V2.
- **Days 4-7**: Implement FiLM metadata injection and train the Ensemble components (DINOv2 + ConvNeXt).
- **Days 8-10**: Focus on Post-Processing (Nelder-Mead & Reconciliation). This requires no retraining, just OOF optimization.
- **Days 11-14**: Pseudo-labeling and final ensemble blending.
