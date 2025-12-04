# WedgeFieldDoseCal

Multichannel dose calibration based on scans of EDW Wedge fields.

## Environment

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

The API stitches together TIFF film scans and RTDOSE DICOM planes, performs rigid mutual-information registration, crops to the overlap, and fits a multichannel calibration model with dose-dependent covariance estimates.

```python
from wedge_field_dose_cal.pipeline import calibrate_from_files, dose_from_tiff

# Provide matching TIFF scans and RTDOSE files (e.g., 500, 1000, 2000, 4000 MU deliveries)
tiff_paths = ["edw_0500.tiff", "edw_1000.tiff", "edw_2000.tiff", "edw_4000.tiff"]
dicom_paths = ["edw_0500.dcm", "edw_1000.dcm", "edw_2000.dcm", "edw_4000.dcm"]

calibrator = calibrate_from_files(tiff_paths, dicom_paths)

# Convert a new film scan to dose
predicted_dose = dose_from_tiff("unknown_field.tiff", calibrator)
```

### Key steps

1. **Load** TIFF and DICOM data with physical pixel spacing.
2. **Resample** the DICOM dose grid to the film DPI.
3. **Register** dose to film via 2D rigid mutual information and crop the valid overlap.
4. **Calibrate** using log optical density across all three channels with dose-binned 3×3 covariance maps.
5. **Predict** dose for future films by weighting dose bins with Mahalanobis distances (covariance handling is internal).
