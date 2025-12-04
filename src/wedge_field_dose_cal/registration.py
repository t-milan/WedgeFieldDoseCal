from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import SimpleITK as sitk


@dataclass
class RegistrationResult:
    aligned_dose: np.ndarray
    cropped_film: np.ndarray
    cropped_dose: np.ndarray
    transform: sitk.Transform
    crop_slice: Tuple[slice, slice]


def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.tensordot(rgb, weights, axes=([2], [0]))


def register_dose_to_film(
    film_rgb: np.ndarray,
    dose_resampled: np.ndarray,
    pixel_spacing_mm: tuple[float, float],
    mutual_information_bins: int = 50,
    sampling_percentage: float = 0.2,
) -> RegistrationResult:
    """
    Register a resampled DICOM dose plane to the film using mutual information.
    The transform is rigid (translation + rotation) in 2D.
    """
    film_gray = _to_grayscale(film_rgb).astype(np.float32)
    dose_float = dose_resampled.astype(np.float32)

    # SimpleITK expects spacing as (x, y) = (col, row)
    spacing = (float(pixel_spacing_mm[1]), float(pixel_spacing_mm[0]))

    film_img = sitk.GetImageFromArray(film_gray)
    dose_img = sitk.GetImageFromArray(dose_float)
    film_img.SetSpacing(spacing)
    dose_img.SetSpacing(spacing)

    initial_transform = sitk.CenteredTransformInitializer(
        film_img,
        dose_img,
        sitk.VersorRigid2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(mutual_information_bins)
    registration_method.SetMetricSamplingPercentage(sampling_percentage, sitk.sitkWallClock)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        relaxationFactor=0.5,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(film_img, dose_img)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(film_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)
    resampler.SetDefaultPixelValue(np.nan)

    aligned_dose = sitk.GetArrayFromImage(resampler.Execute(dose_img))

    valid_mask = np.isfinite(aligned_dose)
    if not np.any(valid_mask):
        raise RuntimeError("Registration produced no overlapping region between dose and film")

    row_any = valid_mask.any(axis=1)
    col_any = valid_mask.any(axis=0)
    row_inds = np.where(row_any)[0]
    col_inds = np.where(col_any)[0]

    row_slice = slice(row_inds.min(), row_inds.max() + 1)
    col_slice = slice(col_inds.min(), col_inds.max() + 1)

    cropped_dose = aligned_dose[row_slice, col_slice]
    cropped_film = film_rgb[row_slice, col_slice]

    return RegistrationResult(
        aligned_dose=aligned_dose,
        cropped_film=cropped_film,
        cropped_dose=cropped_dose,
        transform=final_transform,
        crop_slice=(row_slice, col_slice),
    )
