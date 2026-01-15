# Implementation Plan - CSIRO Image2Biomass Winning Strategy

## Goal Description
Implement the core components of the "Winning Strategy" by modularizing the codebase. This moves away from monolithic notebooks to a structured `src/` library, enabling systematic experimentation with advanced techniques like State-Aware CV, Tweedie Loss, and Hierarchical Reconciliation.

## User Review Required
> [!IMPORTANT]
> This plan involves creating a **new python package structure** (`src/`) in your directory.
> You will need to make sure your notebooks can import from this directory (usually just ensuring it's in the path).

## Proposed Changes

### Structure
Create a new directory `src/` to hold all reusable logic.

### Cross-Validation
#### [NEW] [src/cv.py](file:///c:/Users/hp/Desktop/csiro-image2biomass/src/cv.py)
- Implement `StateAwareStratifiedGroupKFold`.
- Logic:
    - Groups samples by location (if available) or ensures State balance.
    - Stratifies by `binned_target` (Dry_Total_g) to ensure high-biomass representation in all folds.

### Metrics & Losses
#### [NEW] [src/metrics.py](file:///c:/Users/hp/Desktop/csiro-image2biomass/src/metrics.py)
- Implement `weighted_r2_score` exact competition metric.
- Add `MetricCallback` for popular libraries (PyTorch Lightning / Keras / Standard Loop) if applicable, or just a standalone function.

#### [NEW] [src/losses.py](file:///c:/Users/hp/Desktop/csiro-image2biomass/src/losses.py)
- Implement `TweedieLoss` (for zero-inflated accumulation).
- Implement `LogCoshLoss` (robust regression).
- Implement `CompoundLoss` wrapper.

### Post-Processing
#### [NEW] [src/postprocess.py](file:///c:/Users/hp/Desktop/csiro-image2biomass/src/postprocess.py)
- Implement `hierarchical_reconciliation(preds)` using `cvxpy` or strict linear algebra projection.
- Implement `optimize_ensemble_weights(oof_preds, OOF_TARGETS)` using `scipy.optimize.minimize(method='Nelder-Mead')`.

## Verification Plan

### Automated Tests
I will create a simple test script `tests/test_components.py` to verify:
1.  **CV Split**: Check if folds are non-overlapping and `Dry_Total_g` distribution is similar.
2.  **Metric**: Run `weighted_r2_score` on dummy data and compare with manual calculation.
3.  **Loss**: Ensure `TweedieLoss` is differentiable and handles zeros without NaN.
4.  **Reconciliation**: Ensure `Green + Dead == Total` after processing dummy predictions.

### Manual Verification
- **Notebook Integration**: I will create a small demo notebook `src_demo.ipynb` that imports these functions and runs a "dummy" training loop step to prove integration works.
