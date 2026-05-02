# Explainability Protocol

## Status

**No explainability maps are generated in this pass.**

No Grad-CAM overlays, attention visualizations, saliency maps, token importance
maps, or any other interpretability artifacts have been generated for any model in
this evidence-pack pass.

This document defines the protocol that must be followed before any explainability
output is generated and added to this repository.

---

## Why Explainability Matters for Remote-Sensing Classification

Explainability tools help answer a critical question in remote-sensing research:
is the model attending to the correct land-surface features (crop patterns, field
boundaries, vegetation density) or is it exploiting spurious signals (image
compression artifacts, frame borders, lighting conditions)?

Without interpretability analysis, even high preserved accuracy values leave open
the possibility that the model is using unintended cues. Explainability artifacts
would:

- Increase reviewer confidence that the model generalizes for the right reasons
- Enable detection of dataset bias or shortcut learning
- Provide visually communicable evidence of model behavior for research presentations
- Support responsible use disclosures for downstream applications

---

## Grad-CAM for CNN Baselines — Future Direction

Gradient-weighted Class Activation Mapping (Grad-CAM) computes a weighted sum of
CNN feature map gradients with respect to the predicted class, producing a coarse
spatial heatmap that highlights the image regions most influential for the prediction.

### Required Inputs Before Generating Grad-CAM

| Input | Description | Current availability |
|---|---|---|
| Trained CNN checkpoint | `models/keras_cnn_best.keras` or `models/pytorch_cnn_best.pth` | Not bundled |
| Sample images | Representative images from both `class_0_non_agri` and `class_1_agri` | Dataset not bundled |
| Target layer name | The final convolutional layer name in the specific model architecture | Must be verified per checkpoint |
| Inference environment | Python environment with TensorFlow/Keras or PyTorch installed | Available after `pip install -r requirements.txt` |

### Required Commands

```bash
# After dataset and checkpoints are locally available:
# Keras CNN Grad-CAM
python scripts/run_predict.py \
    --model-type keras_cnn \
    --model-path models/keras_cnn_best.keras \
    --image-path path/to/sample_image.jpg

# PyTorch CNN Grad-CAM
python scripts/run_predict.py \
    --model-type pytorch_cnn \
    --model-path models/pytorch_cnn_best.pth \
    --image-path path/to/sample_image.jpg
```

Note: Grad-CAM generation will require extending the prediction scripts or adding
a dedicated explainability script. The current `run_predict.py` outputs class
labels and probabilities only.

### Provenance Requirements for Grad-CAM Outputs

Any Grad-CAM image added to this repository must be accompanied by:

1. The exact checkpoint file name and SHA-256 hash
2. The exact sample image path and its class label
3. The target convolutional layer name used
4. The predicted class and probability
5. The framework and version used to generate the map
6. The command used to generate the map

---

## Attention and Token Visualization for CNN-ViT Hybrids — Future Direction

For the CNN-ViT hybrid models, the transformer encoder's self-attention weights
provide a natural source for token-level visualization. An attention rollout or
raw attention weight map can indicate which spatial token positions the model
emphasizes when classifying an image.

### Key Distinction

Because tokens in these hybrid models are CNN feature positions (not raw image
patches), the attention visualization maps back to CNN feature map positions, not
pixel-level image locations. Interpreting these visualizations requires awareness
of the CNN backbone's receptive field and spatial resolution reduction.

### Required Inputs Before Generating Attention Visualizations

| Input | Description | Current availability |
|---|---|---|
| Trained CNN-ViT checkpoint | `models/keras_cnn_vit_best.keras` or `models/pytorch_cnn_vit_best.pth` | Not bundled |
| Sample images | Representative images from both classes | Dataset not bundled |
| Attention head selection | Which head(s) to visualize (single head, mean, rollout) | Must be decided per experiment |
| Receptive field mapping | CNN backbone receptive field to map tokens back to image regions | Must be computed from architecture |

### Provenance Requirements for Attention Visualizations

Any attention visualization added to this repository must be accompanied by:

1. The exact checkpoint file name and SHA-256 hash
2. The exact sample image path and class label
3. The attention head selection method used (single head index, mean across heads,
   or attention rollout)
4. The transformer layer depth from which attention weights were extracted
5. The predicted class and probability
6. The command used to generate the visualization

---

## Why Visualizations Must Not Be Fabricated

Adding explainability maps that were not generated from actual model inference on
real data would constitute research misconduct. Specifically:

- A fabricated Grad-CAM or attention map that highlights plausible-looking regions
  without being derived from real model gradients or attention weights misrepresents
  the model's actual behavior.
- It would invalidate any research claim that cites the visualization as evidence
  of what the model attends to.
- It would expose the repository and its authors to credibility damage if the
  fabrication were discovered during review or replication.

**No explainability visualization may be added to this repository unless it was
generated from a real checkpoint, real input image, and a documented, reproducible
command.**

---

## How to Label Explainability Outputs

When explainability artifacts are eventually generated and added to this repository,
each file must include:

- A filename that encodes the model type, image class, and generation method,
  e.g.: `results/explainability/gradcam_pytorch_cnn_class1_agri_sample01.png`
- A companion provenance record (JSON or Markdown) stored alongside the image
  documenting checkpoint hash, source image, target layer, command, and framework
  version
- A reference entry in `METRIC_PROVENANCE_MATRIX.md` or a dedicated
  `EXPLAINABILITY_PROVENANCE.md` file

---

## Blocked Until

Explainability artifacts are blocked from generation until **all** of the following
are satisfied:

- [ ] Local dataset is prepared: `data/images_dataSAT/` exists with both class folders
- [ ] At least one model checkpoint is available locally in `models/`
- [ ] The checkpoint's training provenance is documented (source notebook or script,
      epoch count, optimizer, seed)
- [ ] A representative sample image from each class is selected and documented
- [ ] The target layer or attention head selection is decided and recorded
- [ ] A reproducible generation command is written, tested, and documented
