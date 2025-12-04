"""Tools for film dose registration and multichannel calibration."""
from .io import load_tiff_rgb, load_dicom_dose, estimate_tiff_spacing_mm
from .resample import resample_dose_to_film_spacing
from .registration import register_dose_to_film, RegistrationResult
from .calibration import MultichannelDoseCalibrator, CovarianceModel
from .pipeline import calibrate_from_files, dose_from_tiff
from .qa import (
    plot_registration_overlay,
    plot_calibration_curves,
    plot_dose_comparison,
    plot_gamma_analysis,
    compute_gamma_index,
    print_calibration_summary,
)

__all__ = [
    "load_tiff_rgb",
    "load_dicom_dose",
    "estimate_tiff_spacing_mm",
    "resample_dose_to_film_spacing",
    "register_dose_to_film",
    "RegistrationResult",
    "MultichannelDoseCalibrator",
    "CovarianceModel",
    "calibrate_from_files",
    "dose_from_tiff",
    # QA tools
    "plot_registration_overlay",
    "plot_calibration_curves",
    "plot_dose_comparison",
    "plot_gamma_analysis",
    "compute_gamma_index",
    "print_calibration_summary",
]
