# LUNA16 Dataset — Download & Setup Instructions

## Overview

The **Lung Nodule Analysis 2016 (LUNA16)** dataset is derived from the
[LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
database. It contains 888 CT scans with annotated pulmonary nodules ≥3mm
where at least 3 of 4 radiologists agreed on the nodule's existence.

---

## Step 1: Register for Access

1. Navigate to the **LUNA16 Grand Challenge** portal:
   [https://luna16.grand-challenge.org/](https://luna16.grand-challenge.org/)

2. Create an account and accept the **data use agreement**.

3. Navigate to **"Download"** → **"Data"**.

---

## Step 2: Download the Data

LUNA16 is distributed as 10 subsets. Download all subsets:

```
subset0.zip   (~1.8 GB)
subset1.zip   (~1.8 GB)
...
subset9.zip   (~1.8 GB)
```

Also download the annotation files:
```
annotations.csv          # Ground truth: nodule center + diameter
candidates_V2.csv        # All candidates (TP + FP)
```

### Using the Download Script

After obtaining your credentials, you can use the provided script:

```bash
bash scripts/download_luna16.sh \
    --output_dir /data/LUNA16 \
    --username YOUR_USERNAME \
    --password YOUR_PASSWORD
```

---

## Step 3: Verify the Download

After extraction, your directory should look like:

```
/data/LUNA16/
├── subset0/
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.raw
│   └── ...  (more .mhd/.raw pairs)
├── subset1/
│   └── ...
...
├── subset9/
│   └── ...
├── annotations.csv
└── candidates_V2.csv
```

Verify the file count:

```bash
# Should print 888
find /data/LUNA16 -name "*.mhd" | wc -l
```

---

## Step 4: Annotations File Format

`annotations.csv` columns:
| Column | Description |
|---|---|
| `seriesuid` | CT scan series identifier |
| `coordX` | Nodule center X coordinate (mm) |
| `coordY` | Nodule center Y coordinate (mm) |
| `coordZ` | Nodule center Z coordinate (mm) |
| `diameter_mm` | Nodule diameter in mm |

Example:
```
seriesuid,coordX,coordY,coordZ,diameter_mm
1.3.6.1...860,-128.699421,-175.319272,-298.387506,5.65154
```

---

## Step 5: Preprocess the Data

Run the preprocessing pipeline to:
- Resample all volumes to **isotropic 1mm³ spacing**
- Apply **HU windowing** (-1000 to 400 HU)
- Normalize to **[0, 1]**
- Generate **binary nodule masks** from annotation coordinates

```bash
python preprocessing/ct_preprocessing.py \
    --data_dir /data/LUNA16 \
    --output_dir /data/LUNA16/preprocessed \
    --annotations /data/LUNA16/annotations.csv \
    --target_spacing 1.0 1.0 1.0 \
    --n_workers 8
```

**Expected output:**
```
/data/LUNA16/preprocessed/
├── subset0/
│   ├── {seriesuid}_image.npy   # Shape: (D, H, W), float32 in [0, 1]
│   └── {seriesuid}_mask.npy    # Shape: (D, H, W), uint8 {0, 1}
├── subset1/
│   └── ...
...
└── metadata.csv               # Preprocessed volume sizes + spacing info
```

---

## Data Statistics (Post-Preprocessing)

| Metric | Value |
|---|---|
| Total scans | 888 |
| Avg volume shape | ~(280, 512, 512) after resampling |
| Avg scan size on disk | ~140 MB (float32 .npy) |
| Total disk usage | ~120 GB |
| Positive (nodule) voxel fraction | ~0.08% |
| Nodule diameter range | 3 – 30 mm |

---

## Alternative: LIDC-IDRI via TCIA

For broader access, the full LIDC-IDRI dataset (which LUNA16 is derived from)
is available via the **The Cancer Imaging Archive (TCIA)**:

```bash
# Install TCIA downloader
pip install tcia_utils

# Download LIDC-IDRI
python -c "
from tcia_utils import nbia
nbia.downloadSeries('LIDC-IDRI', input_data=None, path='/data/LIDC-IDRI')
"
```

---

## Troubleshooting

**"Cannot read .mhd file"**
```bash
pip install SimpleITK==2.3.0
python -c "import SimpleITK as sitk; img = sitk.ReadImage('/data/LUNA16/subset0/xxx.mhd'); print(img.GetSize())"
```

**"Disk space issues"**
Process one subset at a time:
```bash
python preprocessing/ct_preprocessing.py --subsets 0 1 2 --output_dir /data/LUNA16/preprocessed
```

---

## License & Data Use Agreement

The LUNA16 dataset is released under the **Creative Commons Attribution 3.0
Unported License**. By downloading, you agree to:
- Use the data for research purposes only
- Not redistribute the data
- Cite the LUNA16 paper in publications

> Setio, A.A.A., et al. "Validation, comparison, and combination of algorithms
> for automatic detection of pulmonary nodules in computed tomography images:
> the LUNA16 challenge." *Medical Image Analysis* 42 (2017): 1-13.
