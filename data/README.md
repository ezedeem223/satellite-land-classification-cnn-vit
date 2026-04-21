# Data

This project uses the public `images-dataSAT.tar` archive referenced throughout the preserved experiment records:

- Archive URL: `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar`
- Extracted directory name: `images_dataSAT`
- Practical task implemented in the preserved experiments: binary satellite image classification between agricultural and non-agricultural land

Expected local layout:

```text
data/
`-- images_dataSAT/
    |-- class_0_non_agri/
    |   |-- image_0001.jpg
    |   `-- ...
    `-- class_1_agri/
        |-- image_0001.jpg
        `-- ...
```

Notes:

- Earlier experiment records extracted the archive beside the notebook files; this repository standardizes that location.
- The refactored project standardizes that location to `data/images_dataSAT/`.
- `class_0_non_agri` and `class_1_agri` are preserved exactly because they are the actual class folder names used in the preserved experiments.
- The images are not tracked in git. Use `python scripts/run_prepare_data.py --config configs/data.yaml` to download and extract them locally.
- The preserved experiment records cover multiple loading strategies:
  - direct bulk or path-based inspection
  - Keras generator-based loading and augmentation
  - PyTorch `Dataset` / `ImageFolder` / `DataLoader` pipelines
