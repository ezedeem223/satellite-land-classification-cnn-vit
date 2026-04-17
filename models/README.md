# Models

This directory is intentionally empty by default except for `.gitkeep`.

Expected checkpoint outputs from the refactored scripts:

- `models/keras_cnn_best.keras`
- `models/pytorch_cnn_best.pth`
- `models/keras_cnn_vit_best.keras`
- `models/pytorch_cnn_vit_best.pth`

Important context from the notebooks:

- Module 2 comparison notebook uses downloaded pretrained CNN checkpoints rather than the short local training runs alone.
- Module 3 and Module 4 hybrid notebooks also reference downloaded pretrained checkpoints:
  - Keras CNN checkpoint URL from the course notebook:
    `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/U-uPeyCyOQYh0GrZPGsqoQ/ai-capstone-keras-best-model-model.keras`
  - PyTorch CNN state dict URL from the course notebook:
    `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/8J2QEyQqD8x9zjrlnv6N7g/ai-capstone-pytorch-best-model-20250713.pth`
  - Keras CNN-ViT hybrid URL from the course notebook:
    `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7uNMQhNyTA8qSSDGn5Cc7A/keras-cnn-vit-ai-capstone.keras`
  - PyTorch CNN-ViT hybrid URL from the course notebook:
    `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rFBrDlu1NNcAzir5Uww8eg/pytorch-cnn-vit-ai-capstone-model-state-dict.pth`

The repository does not ship those binary weights. Train the models locally with the provided scripts, or place externally downloaded checkpoints here before running evaluation or prediction.

