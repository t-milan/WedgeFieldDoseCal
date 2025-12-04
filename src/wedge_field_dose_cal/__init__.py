"""Tools for film dose registration and multichannel calibration."""
from .io import load_tiff_rgb, load_dicom_dose, estimate_tiff_spacing_mm
from .resample import resample_dose_to_film_spacing
from .registration import register_dose_to_film
from .calibration import MultichannelDoseCalibrator
from .pipeline import calibrate_from_files, dose_from_tiff

__all__ = [
    "load_tiff_rgb",
    "load_dicom_dose",
    "estimate_tiff_spacing_mm",
    "resample_dose_to_film_spacing",
    "register_dose_to_film",
    "MultichannelDoseCalibrator",
    "calibrate_from_files",
    "dose_from_tiff",
]
