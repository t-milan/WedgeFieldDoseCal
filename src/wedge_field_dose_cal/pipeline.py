from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .calibration import MultichannelDoseCalibrator
from .io import load_dicom_dose, load_tiff_rgb
from .registration import register_dose_to_film
from .resample import resample_dose_to_film_spacing


def calibrate_from_files(
    tiff_paths: Sequence[str | Path],
    dicom_paths: Sequence[str | Path],
    num_covariance_bins: int = 24,
    margin_mm: float = 5.0,
) -> MultichannelDoseCalibrator:
    """
    Build a multichannel calibration from paired TIFF scans and DICOM dose planes.

    Parameters
    ----------
    tiff_paths : sequence of paths
        Paths to scanned film TIFF files
    dicom_paths : sequence of paths
        Paths to corresponding DICOM dose planes
    num_covariance_bins : int
        Number of bins for dose-dependent covariance estimation
    margin_mm : float
        Margin to crop from field edges after registration (mm)
    """
    if len(tiff_paths) != len(dicom_paths):
        raise ValueError("Number of TIFF and DICOM inputs must match")

    calibrator = MultichannelDoseCalibrator(num_covariance_bins=num_covariance_bins)
    for tiff_path, dicom_path in zip(tiff_paths, dicom_paths):
        film_rgb, film_spacing = load_tiff_rgb(tiff_path)
        dose_plane, dose_spacing = load_dicom_dose(dicom_path)
        dose_resampled = resample_dose_to_film_spacing(dose_plane, dose_spacing, film_spacing)
        registration = register_dose_to_film(film_rgb, dose_resampled, film_spacing, margin_mm=margin_mm)
        calibrator.add_pair(registration.cropped_film, registration.cropped_dose)

    calibrator.fit()
    return calibrator


def dose_from_tiff(tiff_path: str | Path, calibrator: MultichannelDoseCalibrator) -> np.ndarray:
    film_rgb, _ = load_tiff_rgb(tiff_path)
    return calibrator.predict_dose(film_rgb)
