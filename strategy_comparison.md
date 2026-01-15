# Strategy Comparison: Trees vs. Deep Learning

You asked: *"Why did the 0.69 notebook use Trees (XGBoost/CatBoost) and what is my equivalent?"*

## 1. The "0.69 Notebook" Approach: Feature Extraction + Tabular Regression
That notebook treated Computer Vision as a **Tabular Math Problem**.

*   **Step 1 (Frozen Feature Extractor)**: It used SigLIP (or DINO) simply as a "Camera" to take a mathematical snapshot of the image. It extracted a vector (embedding) and **stopped**. It never updated the SigLIP model itself.
*   **Step 2 (The "Brain")**: It took those numbers (embeddings) and fed them into **Gradient Boosted Trees** (CatBoost, XGBoost).
    *   *Why?* Because Trees are the undisputed king of **fixed** tabular data. If your input features are static lists of numbers, Trees usually beat Neural Networks.

**The Logic**: "I can't afford to retrain a huge image model. I'll just extract features and let a Tree find the patterns."

## 2. Your "Winning Strategy" Approach: End-to-End Fine-Tuning
Your notebooks use **End-to-End Deep Learning**.

*   **The Equivalent of "Trees"**: Your **MLP Head** (The `nn.Linear` layers at the end of `BiomassModel`).
    *   Instead of a Tree deciding "If Feature 5 > 0.5, then Biomass = 10", your MLP layers do weighted matrix multiplications to output the value.
    *   *Why use an MLP instead of a Tree here?* Because an MLP allows **Gradients to flow backward**.

## 3. The Critical Difference: "Backpropagation"
This is why your strategy has a higher ceiling (Rank #1 potential).

*   **0.69 Notebook (Trees)**: The feature extractor (SigLIP) **NEVER LEARNS**. If SigLIP thinks "Dead Grass" looks like "Sand", the Tree can't fix that basic perception error. It just deals with the bad data.
*   **Your Strategy (End-to-End)**: When your MLP makes an error, it sends a signal **all the way back** through the DINOv2/ConvNeXt backbone.
    *   Your model **updates the "eyes"** (Conv/Attention layers) to better distinguish dead grass from sand.
    *   It **customizes the feature extractor** for this specific CSIRO dataset.

| Feature | 0.69 Notebook Strategy | Your Winning Strategy |
| :--- | :--- | :--- |
| **Vision Model** | Frozen Feature Extractor (SigLIP) | **Fine-Tuned Backbone** (DINOv2 / ConvNeXt) |
| **"The Brain"** | Gradient Boosted Trees (CatBoost) | **Differentiable MLP Head** (Linear Layers) |
| **Learning Scope** | Only learns how to map *existing* features | Learns to **create better features** AND map them |
| **Data Usage** | fast, low resource | Slower, but higher accuracy potential |

**Summary**: Your "Tree equivalent" is the `self.head` in your model code. But unlike a standalone tree, your head helps retrain the entire vision system.
