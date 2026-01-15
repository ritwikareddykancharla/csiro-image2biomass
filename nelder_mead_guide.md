# Nelder-Mead Ensemble Optimization: A Practical Guide

## What is Nelder-Mead?
The **Nelder-Mead method** (also known as the Simplex method) is a numerical method used to find the minimum or maximum of an objective function in a multidimensional space.

- **Derivative-Free**: Unlike Gradient Descent, it does *not* require the function to be differentiable (have gradients). It only needs to evaluate the function result.
- **Geometric**: It uses a "simplex" (a triangle in 2D, a tetrahedron in 3D) that crawls, expands, and contracts across the search space to find the optimal point.

## Why use it for Ensembling?
In Kaggle competitions (and this biomass task), we usually have predictions from multiple models:
- Model A (DINOv2)
- Model B (ConvNeXt)
- Model C (MaxViT)

We want to find the perfect mix:
$$ FinalPrediction = w_1 \cdot Pred_A + w_2 \cdot Pred_B + w_3 \cdot Pred_C $$

**The Problem**: The competition metric (Weighted R²) is complex, and standard Linear Regression (OLS) minimizes MSE, not the specific Weighted R². Also, we often want to impose constraints (e.g., weights must sum to 1, weights > 0).

**The Solution**: Treat the validation score as a "Black Box" function. Nelder-Mead will try different weight combinations, check the score, and converge on the best weights.

## Implementation Code

Here is how to implement it using `scipy` for the CSIRO competition.

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score

# 1. Prepare your OOF (Out-Of-Fold) Predictions
# Shape: (N_samples, N_targets, N_models)
# For simplicity, let's assume we are averaging predictions for a single target first
# or optimizing one set of scalar weights for all targets.

def calculate_competition_metric(y_true, y_pred):
    # Place the actual competition metric code here
    # Returns a FLOAT (higher is better)
    return weighted_r2_score(y_true, y_pred)

def objective_function(weights, oof_predictions, y_true):
    """
    The function we want to MINIMIZE.
    Since we want to MAXIMIZE R2, we return NEGATIVE R2.
    """
    # 1. Normalize weights (Softmax or Simple Sum) to ensure they sum to 1
    # This prevents the model from just scaling up predictions indefinitely
    coefs = np.array(weights)
    # Option A: Softmax (always positive, sums to 1)
    # coefs = np.exp(coefs) / np.sum(np.exp(coefs))
    
    # Option B: Simple constraint (handled by constraints param, but NM ignores constraints often)
    # So we often just re-normalize manually here
    coefs = coefs / np.sum(coefs)
    
    # 2. Weighted Blend
    # oof_predictions shape: (Models, Samples, Targets)
    final_pred = np.zeros_like(oof_predictions[0])
    for i, w in enumerate(coefs):
        final_pred += w * oof_predictions[i]
        
    # 3. Calculate Score
    score = calculate_competition_metric(y_true, final_pred)
    
    # 4. Return Negative Score (for minimization)
    return -score

# --- Execution ---

# Example: 3 Models
initial_weights = [0.33, 0.33, 0.33] 

# Lists of OOF predictions from your 3 models
oof_preds = [model1_oof, model2_oof, model3_oof] 

# Target values
y_true = train_df[TARGET_COLS].values

# Run Optimization
# Nelder-Mead is decent, but 'SLSQP' or 'L-BFGS-B' allow bounds (weights > 0)
# However, Nelder-Mead is very robust for non-smooth landscapes.
result = minimize(
    fun=objective_function,
    x0=initial_weights,
    args=(oof_preds, y_true),
    method='Nelder-Mead',
    options={'maxiter': 1000, 'disp': True}
)

best_weights = result.x / np.sum(result.x) # Re-normalize
print("Best Weights:", best_weights)
```

## Tips for Success
1.  **OOF is mandatory**: You must optimize on Out-Of-Fold predictions, *not* the training predictions (or you will overfit massively).
2.  **Boundaries**: If you get negative weights, switch to `method='SLSQP'` with `bounds=[(0, 1)]*N_models` and `constraints={'type':'eq', 'fun': lambda w: 1-sum(w)}`. Nelder-Mead doesn't support bounds natively, but the manual normalization inside the objective function usually handles it.
3.  **Separate Optimizers?**: You can optimize weights for *each target column* separately if the competition allows it. For CSIRO, optimizing one set of weights for `Dry_Total_g` and another for `Dry_Green_g` might be better than one global set.
