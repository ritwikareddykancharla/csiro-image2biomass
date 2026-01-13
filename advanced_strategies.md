# Winning Strategies: Bridging the Gap from 0.69 to 0.79

To jump from a solid 0.69 score to the winning 0.79 range, you need to move beyond standard training. Here are specific, high-impact strategies tailored for this biomass estimation challenge.

## 1. Data Strategy & Validation (Critical)
The distribution of biomass is likely highly skewed (long tail). Random K-Fold often fails here.

*   **Stratified K-Fold on Bins**: Don't just `KFold`. Bin the continuous `Dry_Total_g` target into 5-10 bins and use `StratifiedKFold`. This ensures every fold has a representative distribution of high-biomass examples (which have the highest error impact).
*   **Adversarial Validation**: Train a classifier to distinguish `Train` vs `Test` images. If it's accurate, your validation set doesn't match the leaderboard. Drop easy "train-like" samples from validation to mimic the hard test set.

## 2. Advanced Feature Engineering
The "Comparison Notebook" showed metadata (`State`, `Date`) exists but isn't fully exploited.

*   **Seasonality Features**: Biomass depends heavily on the time of year.
    *   Convert `Sampling_Date` to `Day of Year` (1-365).
    *   Add cyclic features: `sin(2*pi*day/365)`, `cos(2*pi*day/365)`.
*   **State/Location Encoding**: Use Target Encoding or One-Hot encoding for `State`. Different regions have different grass species mixes.
*   **External Climate Data**: If rules allow, fetch historical rainfall/temperature data for those States/Dates. Biomass lag correlates with rainfall from 2-4 weeks prior.

## 3. Stronger & Diverse Models
The current ensemble uses DINOv2 (ViT). You need Diversity.

*   **CNNs are not dead**: Add a **ConvNeXt V2 (Large/Huge)** or **EfficientNetV2-L**. CNNs capture different texture patterns (grass blades vs clover leaves) than ViTs.
*   **Swin Transformer V2**: A hierarchical transformer that excels at dense prediction tasks like this. It handles variable resolutions better than standard ViT.
*   **Regression Heads**: Instead of a simple Linear Layer, try a small MLP head `(Linear -> GELU -> Dropout -> Linear)` on top of the backbone.

## 4. Training Tricks for Regression
*   **Mixup & Cutmix**: Essential. linearly interpolating two images AND their biomass labels forces the model to learn "volume" rather than just memorizing samples.
*   **Loss Function Tuning**:
    *   `Huber Loss` or `SmoothL1Loss` is good (used in the baseline), but try `Tweedie Loss` (great for zero-inflated, skewed positive data like biomass).
*   **Target Scaling**: The ensemble notebook uses **Max Scaling** (dividing by constant max) for some models. Consider **Log Scaling** (`log1p`), which is standard for biomass/count data to stabilize variance (making the model care equally about small and large error *ratios* rather than absolute values).
*   **Pseudo-Labeling (The "Leaderboard Climber")**:
    1.  Train your best ensemble (Score ~0.70).
    2.  Predict on the **Test Set**.
    3.  Take the most confident predictions (or all).
    4.  Add them to the **Train Set** and retrain.
    5.  This aligns the model domain with the test distribution.

## 5. Post-Processing Optimization
The ensemble notebook uses simple averaging or fixed weights.

*   **Nelder-Mead Weight Optimization**: Don't guess weights (`0.5 * A + 0.25 * B`). Treat the weights as parameters. Optimize them against the **OOF (Out-of-Fold)** predictions to maximize the competition metric directly.
*   **Constraint Satisfaction**: The notebook `post_process_biomass` enforces `Total = Green + Dead`. Check if `Green >= Clover` is always true. If `Clover` is a subset of `Green`, enforce `Green >= Clover`.

## 6. Resolution & Tiling (The "Zoom")
*   **Sliding Window Inference**: Instead of non-overlapping tiles, use overlapping tiles (e.g., 50% overlap) and average the predictions in the center. This removes edge artifacts where a plant is cut in half.
*   **Multi-Scale Inference**: Predict at 1.0x, 1.25x, and 0.75x scales and average.

## Recommended Action Plan
1.  **Immediate**: Implement **Stratified K-Fold based on Biomass Bins**. This stabilizes local CV scores.
2.  **Quick Win**: Add **Seasonality (Day of Year)** to the tabular features in the Gradient Boosting models.
3.  **Model**: Swap one of the tree models for a **ConvNeXt** to act as a texture specialist.
