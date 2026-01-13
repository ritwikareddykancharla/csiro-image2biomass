# Hackathon Notebook Strategies Explanation

This document analyzes and explains the machine learning strategies implemented in the provided notebooks for the CSIRO image-to-biomass hackathon.

## Overview

The repository contains three key notebooks representing a progression from a simple baseline to a sophisticated, high-performance ensemble.

| Notebook | Type | Key Strategy | Complexity |
| :--- | :--- | :--- | :--- |
| `csiro-simple.ipynb` | Baseline | EfficientNet Regression | Low |
| `csiro-comp-notebook.ipynb` | EDA / Analysis | Data Pivoting & Correlation | Low |
| `0-69-ensemble-3-models-embeddings.ipynb` | SOTA / Ensemble | SigLIP Features + DINOv2 + Mamba | High |

---

## 1. Baseline Strategy (`csiro-simple.ipynb`)

This notebook implements a standard deep learning regression pipeline.

### key Components:
*   **Data Handling**:
    *   Loads `train.csv`.
    *   **Pivoting**: Converts the "long" format (multiple rows per image, one for each target like `Dry_Green_g`, `Dry_Dead_g`) into a "wide" format (one row per image with columns for each target).
    *   **Augmentation**: Uses standard `torchvision.transforms` (Resize, Flip, ColorJitter).
*   **Model**:
    *   Uses `timm` to create an **EfficientNet-B2** (`efficientnet_b2`).
    *   Pre-trained on ImageNet.
    *   Modified output layer to predict 3 targets: `Dry_Green_g`, `Dry_Clover_g`, `Dry_Dead_g`.
    *   Note: `GDM_g` and `Dry_Total_g` are derived from the sum of these predictions, ensuring consistency.
*   **Training**:
    *   Loss Function: `SmoothL1Loss` (Huber Loss), which is robust to outliers compared to MSE.
    *   Optimizer: `AdamW` with `ReduceLROnPlateau` scheduler.
    *   Evaluation: Weighted R² score (custom competition metric).
    *   Cross-Validation: 5-Fold K-Fold.

## 2. Exploratory Data Analysis (`csiro-comp-notebook.ipynb`)

This notebook focuses on understanding the data distribution and relationships rather than building a complex model.

### Key Insights:
*   **Data Transformation**: Demonstrates how to pivot the raw CSV into a usable dataframe for modeling.
*   **Correlation Analysis**: Checks correlations between satellite/sensor metadata (`Pre_GSHH_NDVI`, `Height_Ave_cm`) and the ground truth biomass.
    *   *Finding*: Strong correlation between Height and Green Biomass.
*   **Consistency Checks**: Verifies if the provided Total Biomass equals the sum of its parts (Green + Dead + Clover).
*   **Visualization**: Uses Histograms and Boxplots to show the distribution of biomass values (often skewed/long-tailed).

## 3. Advanced Ensemble Strategy (`0-69-ensemble-3-models-embeddings.ipynb`)

This is a high-scoring solution (LB ~0.69) that employs a complex multi-stage ensemble strategy.

### Stage A: Feature Engineering & Gradient Boosting
Instead of just training a CNN on pixels, this stage extracts high-level semantic features.

1.  **SigLIP Embeddings**: Uses a pre-trained **SigLIP** (similar to CLIP) model (`siglip-so400m-patch14-384`) to extract visual embeddings from image patches.
2.  **Semantic Features (Zero-Shot)**:
    *   Defines text prompts for concepts like "bare soil", "lush green pasture", "dead grass", "clover", "weeds".
    *   Calculates the similarity between the image embeddings and these text feature vectors.
    *   This provides interpretability (e.g., "how much does this image look like 'dead grass'?").
3.  **Supervised Embedding Engine**:
    *   Reduces dimensionality using **PCA** (Principal Component Analysis) and **PLS** (Partial Least Squares).
    *   Adds clustering features using **GMM** (Gaussian Mixture Models).
4.  **Ensemble of Regressors**:
    *   Trains multiple tree-based models on these extracted features: **GradientBoosting**, **HistGradientBoosting**, **CatBoost**, and **LightGBM**.
    *   Averages their predictions.

### Stage B: Hybrid Deep Learning Architecture
Implements a custom end-to-end deep learning model named `CrossPVT_T2T_MambaDINO`.

1.  **Backbone**: **DINOv2** (Vision Transformer trained with self-supervision), known for excellent dense feature representations.
2.  **Tiling Strategy**: Splits high-resolution images into grids (e.g., 2x2, 4x4) to handle fine details like clover leaves.
3.  **Advanced Modules**:
    *   **T2T (Tokens-to-Token)**: Condenses visual info.
    *   **Mamba (State Space Model)**: Efficiently processes long sequences of visual tokens (alternative to Attention).
    *   **Cross-Scale Fusion**: Merges global (whole image) and local (tile) features.
4.  **Inference**:
    *   Uses **TTA (Test Time Augmentation)**: Averages predictions on original, horizontally flipped, and vertically flipped images.

### Stage C: Final Ensemble & Post-Processing
1.  **Weighted Averaging**: Combines predictions from Stage A and Stage B (e.g., `0.5 * StageA + 0.25 * ModelB1 + 0.25 * ModelB2`).
2.  **Matrix Reconciliation (`post_process_biomass`)**:
    *   The problem has a hierarchy: `Total = Green + Dead`, `Green = Grass + Clover` (simplified).
    *   Uses linear algebra (Projection Matrix) to force the independent predictions to satisfy these summation constraints mathematically.
    *   Clips negative values to 0.

## Recommendations for Your Hackathon

1.  **Start with `csiro-simple.ipynb`**: Get a submission pipeline working.
2.  **Implement Post-Processing**: Copy the `post_process_biomass` function from the ensemble notebook. It typically boosts scores for free by enforcing consistency.
3.  **Use Embeddings**: If you don't have compute for training huge ViTs, use the "Stage A" approach: extract embeddings (SigLIP or DINOv2) and train XGBoost/CatBoost. It's fast and effective.
4.  **Tiling is Key**: For biomass estimation, resolution matters. Splitting images into crops (as seen in the advanced notebook) usually helps performance.
