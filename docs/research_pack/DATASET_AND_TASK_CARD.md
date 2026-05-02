# Dataset and Task Card

---

## Task Name

Binary Satellite Image Land Classification: Agricultural vs. Non-Agricultural

---

## Task Type

Binary image classification (remote-sensing)

---

## Label Space

| Label index | Class folder | Human-readable label |
|---|---|---|
| 0 | `class_0_non_agri` | non-agricultural |
| 1 | `class_1_agri` | agricultural |

---

## Class Folders

The class folder names are preserved exactly as they appeared in the original
experiment notebooks:

- `class_0_non_agri`
- `class_1_agri`

These names are used throughout the codebase, configurations, and tests. They must
not be renamed without updating all dependent paths and configs.

---

## Expected Local Dataset Layout

After running the data preparation script, the expected layout is:

```
data/
└── images_dataSAT/
    ├── class_0_non_agri/
    │   ├── image_0001.jpg
    │   └── ...
    └── class_1_agri/
        ├── image_0001.jpg
        └── ...
```

---

## Dataset Not Bundled

The `images_dataSAT` archive is **not** committed to this repository. Images are
not tracked in git.

**Source:** Public IBM Cloud Object Storage archive
**URL:** `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar`
**Archive name:** `images-dataSAT.tar`
**Extracted directory:** `data/images_dataSAT/`

---

## Expected Preparation Path

1. Ensure Python environment is set up:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. Download and extract the dataset:
   ```bash
   python scripts/run_prepare_data.py --config configs/data.yaml
   ```

3. Verify the layout exists:
   ```
   data/images_dataSAT/class_0_non_agri/   ← non-agricultural images
   data/images_dataSAT/class_1_agri/       ← agricultural images
   ```

---

## Image Properties

| Property | Value |
|---|---|
| Image size (model input) | 64 × 64 pixels |
| Channels | 3 (RGB) |
| Format | JPEG (`.jpg`) |
| Validation split | 20% |
| Random seed | 7331 |

---

## Evaluation Split (Preserved Records)

The preserved classification reports show 3,000 images per class in the evaluation
set (6,000 total), balanced across both classes.

---

## Binary Task Boundaries

This is a binary classification task. The scope boundaries are:

- The model classifies image patches as either agricultural or non-agricultural.
- It does not distinguish between types of agriculture (e.g., cropland vs. orchard
  vs. pasture).
- It does not classify urban, forest, water, or other land-cover types as distinct
  categories.
- It does not produce confidence maps, segmentation masks, or pixel-level labels.
- It does not incorporate spatial coordinates or acquisition metadata.

---

## Not a General Land-Cover Taxonomy

The `images_dataSAT` task is a binary remote-sensing classification problem. It is
**not** a general land-cover taxonomy benchmark such as:

- LandCover.ai (multi-class semantic segmentation)
- BigEarthNet (multi-label Sentinel-2 classification)
- EuroSAT (multi-class land-use classification)
- UC Merced Land Use Dataset (21-class aerial imagery)

Results and model designs from this repository should not be compared to those
benchmarks without substantial modifications.

---

## Not Geospatial Metadata-Aware

The current data pipeline does not incorporate:

- Geographic coordinates (latitude, longitude, bounding box)
- Acquisition timestamps or seasonal information
- Sensor type or spectral band metadata
- Spatial resolution metadata

The image patches are processed as standard RGB images without geospatial context.

---

## Ethical and Research Limitations

- **No demographic or sensitive personal data**: The dataset consists of satellite
  image patches of land surface. No personally identifiable information is present.
- **Dataset provenance**: The dataset is publicly provided via an IBM Cloud Object
  Storage URL. Independent verification of dataset composition (total image count,
  geographic origin, acquisition dates) has not been performed in this repository.
- **Label quality**: Label quality for `class_0_non_agri` and `class_1_agri` is
  assumed from the original dataset provider. No independent label audit has been
  performed.
- **Geographic bias**: The geographic distribution of the images in `images_dataSAT`
  is unknown from this repository's vantage point. Results may not generalize across
  geographic regions.
- **No fairness analysis**: No analysis of differential performance across geographic
  regions, seasons, or land-cover sub-types has been performed.

---

## Reproducibility Notes

| Step | Requires dataset? | Requires checkpoints? | Current status |
|---|---|---|---|
| `pip install -e .` | No | No | Fully reproducible |
| `pytest tests/` | No | No | All 7 tests pass |
| `python tools/evidence/validate_research_pack.py` | No | No | Fully reproducible |
| `python scripts/run_prepare_data.py` | Downloads it | No | Requires internet access |
| Training scripts | Yes | No | Requires dataset |
| Evaluation scripts | Yes | Yes | Requires dataset + checkpoints |
| Prediction scripts | No (single image) | Yes | Requires a checkpoint + image |
