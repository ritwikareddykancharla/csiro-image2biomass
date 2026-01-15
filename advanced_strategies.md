# Winning Strategies: Bridging the Gap from 0.69 to 0.79

To jump from a solid 0.69 score to the winning 0.79 range, you need to move beyond standard training. These strategies are refined based on specific insights from the `csiro-comp-notebook.ipynb` EDA.

## 1. Insights from EDA & Immediate Actions (New!)
The analysis of `csiro-comp-notebook.ipynb` revealed three critical patterns you must exploit:

*   **The "Dead vs Green" Disconnect**: `Height_Ave_cm` has a strong correlation (0.65) with `Dry_Green_g` but **zero correlation** (-0.05) with `Dry_Dead_g`. `NDVI` behaves similarly.
    *   *Action*: Create **Interaction Features** like `Height * NDVI`. NDVI captures "greenness", Height captures "volume". Together they provide a 3D proxy.
    *   *Action*: Use **Specialized Heads**: Don't force one shared backbone to predict everything equally. Use a separate head for "Dead Biomass" that emphasizes texture (yellow/brown color) over height/NDVI.
*   **Clover Zero-Inflation**: The 25th percentile of `Dry_Clover_g` is 0.0. This is a sparse, zero-inflated target.
    *   *Action*: Use **Tweedie Loss** (power parameter $p \approx 1.5$). It handles zero-mass peaks much better than MSE/SmoothL1.
*   **State/Location Bias**: `NSW` samples have a mean biomass of ~71g, while `WA` samples coincide with ~31g.
    *   *Action*: **Stratify K-Fold by State**. If your validation set has a different State mix than the leaderboard, your CV score is meaningless.

## 2. Data Strategy & Validation (Critical)
*   **Stratified K-Fold on Bins**: Bin the continuous `Dry_Total_g` target into 5-10 bins and use `StratifiedKFold`. This ensures every fold has a representative distribution of high-biomass examples (which have the highest error impact).
*   **Adversarial Validation**: Train a classifier to distinguish `Train` vs `Test` images. Drop easy "train-like" samples from validation to mimic the hard test set.

## 3. Advanced Feature Engineering
*   **Seasonality Features**:
    *   Convert `Sampling_Date` to `Day of Year` (1-365).
    *   Add cyclic features: `sin(2*pi*day/365)`, `cos(2*pi*day/365)`.
*   **State/Location Encoding**: Use Target Encoding or One-Hot encoding for `State` to capture the "NSW vs WA" bias explicitly.
*   **External Climate Data**: If allowed, fetch historical rainfall/temperature data. Biomass growth lags rainfall by 2-4 weeks.

## 4. Stronger & Diverse Models
The current ensemble uses DINOv2 (ViT). You need Diversity.
*   **CNNs for Texture**: Add a **ConvNeXt V2 (Large/Huge)** or **EfficientNetV2-L**. These are often better at capturing the "texture" of dead grass vs green grass than ViTs.
*   **Swin Transformer V2**: Excellent at dense prediction and variable resolutions.
*   **MLP Heads**: Replace simple linear output layers with 2-layer MLPs (`Linear -> GELU -> Dropout -> Linear`) to capture non-linear feature interactions.

## 5. Training Tricks for Regression
*   **Mixup & Cutmix**: Forces the model to learn "volume" rather than memorizing images.
*   **Loss Function Tuning**:
    *   **Tweedie Loss**: (As mentioned above) Essential for Clover.
    *   **Log Cosh Loss**: Smoother than L1, less explosive than L2.
*   **Target Scaling**: The distributions are long-tailed. Use **Log Scaling** (`log1p(target)`) during training to stabilize variance.
*   **Pseudo-Labeling**: Train on Train+Test(Confident) to adapt to the Test domain.

## 6. Post-Processing Optimization
*   **Nelder-Mead Weight Optimization**: Optimize ensemble weights against OOF predictions.
*   **Constraint Satisfaction**: The notebook `post_process_biomass` enforces `Total = Green + Dead`. Check if `Green >= Clover` is always true. If `Clover` is a subset of `Green`, enforce `Green >= Clover`.

## 7. Resolution & Tiling
*   **Sliding Window Inference**: Use overlapping tiles (e.g., 50% overlap) and average predictions to remove edge artifacts.
*   **Multi-Scale Inference**: Predict at 1.0x, 1.25x, and 0.75x scales and average.
