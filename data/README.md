# Data

This repository is built around the course dataset used throughout the notebooks:

- Archive URL: `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar`
- Extracted directory name: `images_dataSAT`
- Practical task implemented in the notebooks: binary satellite image classification between agricultural and non-agricultural land

Expected local layout:

```text
data/
└── images_dataSAT/
    ├── class_0_non_agri/
    │   ├── image_0001.jpg
    │   └── ...
    └── class_1_agri/
        ├── image_0001.jpg
        └── ...
```

Notes:

- The original course notebooks download the archive into the working directory and extract `images_dataSAT/` beside the notebooks.
- The refactored project standardizes that location to `data/images_dataSAT/`.
- `class_0_non_agri` and `class_1_agri` are preserved exactly because they are the actual class folder names used in the notebooks.
- The images are not tracked in git. Use `python scripts/run_prepare_data.py --config configs/data.yaml` to download and extract them locally.
- Module 1 demonstrates multiple loading strategies:
  - bulk or path-based inspection in `Compare_Memory-Based_Versus_Generator-Based_Data_Loading.ipynb`
  - Keras generator-based loading and augmentation
  - PyTorch `Dataset` / `ImageFolder` / `DataLoader` pipelines

