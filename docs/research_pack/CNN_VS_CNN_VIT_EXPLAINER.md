# CNN vs CNN-ViT Hybrid Explainer

## Overview

This document explains in accessible but academically grounded language what each
model family learns, how they differ architecturally, why the hybrid design is
appropriate for this task, and what the preserved results suggest — and do not prove.

---

## What the CNN Baseline Learns

A Convolutional Neural Network (CNN) applies learned filters across an input image
using local receptive fields. Each filter slides across the spatial dimensions of
the image, computing dot products between the filter weights and local image patches.

For satellite imagery, this local filtering is well-suited to detecting:

- **Texture patterns**: Row-pattern regularity characteristic of planted crop fields
- **Edge structure**: Sharp boundaries between cultivated and uncultivated terrain
- **Color/intensity gradients**: Vegetation index proxies visible in RGB bands
- **Local repetition**: Periodic spatial structures of irrigation channels or
  orchard grids

Deeper convolutional layers compose these local detections into progressively more
abstract representations. A final classifier head (fully connected layer or global
pooling + linear) maps the learned feature representation to the binary output
(`class_0_non_agri` or `class_1_agri`).

The key characteristic of the CNN baseline is its **local spatial inductive bias**:
filters are defined over local receptive fields, and the network has no explicit
mechanism for relating spatially distant regions of the image to each other.

---

## Why CNNs Are Appropriate for Satellite Imagery

CNNs are well-matched to the binary agricultural classification task for several
reasons:

1. **Translation equivariance**: Agricultural patterns (crop rows, field edges)
   appear at varying positions within a scene. Convolutional weight sharing provides
   useful translation invariance without requiring explicit data augmentation.

2. **Local structure dominance**: At 64×64 pixel resolution, most discriminative
   features are locally structured. Global scene context is less critical at this
   scale than at higher resolutions where field boundaries span larger spatial extents.

3. **Data efficiency**: CNNs can be trained effectively on moderately sized datasets.
   Pure patch-based Vision Transformers typically require substantially larger
   training sets or pre-training on large image corpora to converge well from scratch.

4. **Proven remote-sensing record**: Convolutional architectures have a long and
   well-validated track record in remote-sensing image classification tasks.

---

## What the CNN-ViT Hybrid Design Adds

The CNN-ViT hybrid models in this repository add a **transformer encoder stage**
on top of the CNN backbone's output. The pipeline is:

```
Input image (64×64×3)
      │
      ▼
CNN backbone
(convolutional layers producing a spatial feature map)
      │
      ▼
Token construction
(CNN feature map reshaped into a sequence of spatial tokens)
      │
      ▼
Transformer encoder
(multi-head self-attention over the token sequence)
      │
      ▼
Classifier head
(binary output: agri vs non-agri)
```

The transformer encoder applies **multi-head self-attention** across the token
sequence. Unlike convolution, self-attention computes pairwise relationships between
all tokens simultaneously, regardless of their spatial distance. This gives the
hybrid model access to **global context**: a token representing the upper-left
region of the image can directly influence the representation of a token from the
lower-right region.

---

## What Transformer Tokens Represent in This Project

Tokens in this CNN-ViT hybrid are **not** raw image patches (as in a pure ViT).
They are spatial feature vectors produced by the CNN backbone. Each token encodes
the CNN-extracted local features at a specific spatial position in the feature map.

The transformer then models relationships **between these CNN feature positions**,
allowing the model to capture:

- Co-occurrence of feature patterns across different spatial regions
- Contextual consistency (e.g., whether local texture patterns in one region are
  consistent with agricultural patterns detected in adjacent regions)
- Global scene composition signals that may not be accessible to purely local
  convolutional filters

---

## Why the Hybrid Is Not the Same as a Pure ViT

A pure Vision Transformer (Dosovitskiy et al., "An Image Is Worth 16×16 Words")
divides the input image into fixed-size patches and linearly projects each patch
into a token. The token sequence is then processed entirely by the transformer
stack, with no convolutional components.

The CNN-ViT hybrid in this repository differs in two critical ways:

1. **Tokens are CNN features, not raw patches.** The CNN backbone's local spatial
   inductive bias shapes the token content before the transformer ever sees it.
   This reduces the amount of global context modeling the transformer must perform
   and makes the model more data-efficient.

2. **The architecture has convolutional components.** It is not a pure attention
   model. The properties of the convolutional backbone — weight sharing, local
   receptive fields, spatial hierarchy — are integral to the model's behavior.

Calling this architecture "a ViT" without qualification would be technically
inaccurate. The correct description is "CNN-backed transformer hybrid" or
"CNN-ViT hybrid model."

---

## Why Comparing Keras and PyTorch Implementations Is Useful

Both Keras (TensorFlow backend) and PyTorch implementations of the CNN baseline and
CNN-ViT hybrid are provided in this repository. This parallel implementation serves
several purposes:

1. **Framework consistency check**: If both implementations produce similar preserved
   metrics on identical data, it increases confidence that the results reflect genuine
   model behavior rather than framework-specific artifacts (numerical precision,
   default initializations, optimizer behavior).

2. **Reproducibility surface**: Researchers who prefer one framework over the other
   have access to a working reference implementation.

3. **Cross-framework learning**: Examining the structural differences between the
   Keras and PyTorch implementations of the same architecture is pedagogically
   valuable for understanding how framework abstractions influence model construction.

---

## What the Preserved Results Suggest

The preserved metrics (from `results/metrics.json` and `results/model_comparison.csv`)
show:

- All four model variants achieve high preserved accuracy on the binary
  `images_dataSAT` task (0.9925–0.9990).
- PyTorch variants consistently preserve slightly higher scores than their Keras
  counterparts in both model families.
- The PyTorch CNN-ViT hybrid achieves the highest preserved accuracy (0.9990) among
  the four models.
- ROC-AUC of 1.0000 for three of four models suggests strong class separability
  in the preserved evaluation context.

These results are consistent with the hypothesis that the binary agricultural
classification task on `images_dataSAT` is learnable at high accuracy by both CNN
and CNN-ViT architectures, and that the PyTorch implementations converge to slightly
stronger solutions in the preserved records.

---

## What the Preserved Results Do Not Prove

- **Generalization beyond `images_dataSAT`**: The preserved scores apply to this
  specific dataset. Performance on satellite imagery from different sensors,
  resolutions, seasons, or geographic regions is unknown.

- **Superiority of CNN-ViT over CNN in general**: The margin between CNN and CNN-ViT
  preserved scores is small (0.9988 vs 0.9990 for PyTorch), and the evaluation
  context (externally referenced checkpoints, specific splits) limits the strength
  of this claim.

- **Benefit of the transformer stage in isolation**: Because the CNN backbone is
  shared between the baseline and hybrid architectures, it is not possible to
  attribute performance differences solely to the transformer encoder without a
  controlled ablation.

- **Production or operational suitability**: These are research experiment results.
  They do not imply production readiness or deployment suitability.

- **Benchmark comparison**: No comparison with other published methods on this
  dataset has been performed.
