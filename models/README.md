# Models

This directory is intentionally empty by default except for `.gitkeep`.

Expected checkpoint outputs from the refactored scripts:

- `models/keras_cnn_best.keras`
- `models/pytorch_cnn_best.pth`
- `models/keras_cnn_vit_best.keras`
- `models/pytorch_cnn_vit_best.pth`

Important context from the preserved evaluation record:

- The preserved framework-comparison records rely on external pretrained checkpoints in addition to the short local training runs.
- Those binaries are not bundled here.
- To reproduce the preserved evaluation outputs, either:
  - train compatible checkpoints locally with the scripts in this repository, or
  - place compatible external checkpoints into `models/` and point the configs to them

For prediction and evaluation, this directory must contain the model file referenced by the command or config you run.
