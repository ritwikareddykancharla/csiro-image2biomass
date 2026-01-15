# SigLIP: Sigmoid Loss for Language Image Pre-Training

## What is SigLIP?
**SigLIP** (Sigmoid Loss for Language Image Pre-Training) is a state-of-the-art vision-language model introduced by Google DeepMind (2023). It is an architectural improvement over the popular **CLIP** (Contrastive Language-Image Pre-Training) from OpenAI.

## The Key Innovation: Sigmoid vs. Softmax
The fundamental difference lies in how the model calculates the loss between image and text pairs.

### 1. CLIP (The Old Way) uses Softmax
CLIP uses a **Contrastive Loss** with Softmax normalization. For every image, it looks at the correct text and *all* incorrect texts in the batch.
- It forces the model to say: "This image matches Text A **more than** Text B, C, D..."
- **Drawback**: This requires a massive global view of the batch (all negatives must be available). It scales poorly (requires huge memory) and is computationally expensive for large batches.

### 2. SigLIP (The New Way) uses Sigmoid
SigLIP replaces the global Softmax with a simple **Sigmoid Loss** computed on every image-text pair independently.
- It forces the model to say: "Does this image match Text A? Yes/No." (Binary Classification).
- **Advantage**: It decouples the batch elements. You don't need to normalize across the whole batch. 
- **Result**: You can train with **much larger batch sizes** on the same hardware, leading to better performance.

## Why use SigLIP in the CSIRO Competition?
While CSIRO is not a text-to-image task, pre-trained Vision-Language models like SigLIP are excellent feature extractors (backbones) because they have "seen it all."

1.  **Stronger Semantic Understanding**: SigLIP models (like `google/siglip-so400m-patch14-384`) often outperform standard CLIP and ViT models on zero-shot classification and retrieval. They "understand" objects and scenes better.
2.  **Alternative to DINOv2**: Your current strategy relies on DINOv2. SigLIP provides a **different** perspective (language-aligned features vs. self-supervised features).
    - *Ensemble Strategy*: Combine DINOv2 (structure/geometry focus) with SigLIP (semantic/concept focus).
3.  **High Resolution**: Many SigLIP checkpoints are fine-tuned at higher resolutions (384x384 or higher), which is great for the texture-heavy biomass task.

## Implementation Code (timm / transformers)

SigLIP is available in the `timm` library and Hugging Face `transformers`.

```python
import torch
import open_clip

# Loading a pretrained SigLIP model
# "ViT-SO400M" is a Shape-Optimized 400M parameter model (very strong)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-SO400M-14-SigLIP-384', 
    pretrained='webli'
)

# Extracting Features
image = preprocess(your_image).unsqueeze(0)
with torch.no_grad():
    image_features = model.encode_image(image)
    # image_features shape: [1, 1152] - ready for your regression head
```

## Summary Recommendation
Add a **SigLIP** model to your ensemble as a diversity booster. It captures semantic nuances that pure vision models (like ConvNeXt) might miss.
