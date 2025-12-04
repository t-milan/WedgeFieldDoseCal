from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def resample_dose_to_film_spacing(
    dose: np.ndarray,
    dose_spacing_mm: tuple[float, float],
    film_spacing_mm: tuple[float, float],
) -> np.ndarray:
    """
    Resample the DICOM dose plane to match the film pixel spacing using linear interpolation.
    """
    row_scale = dose_spacing_mm[0] / film_spacing_mm[0]
    col_scale = dose_spacing_mm[1] / film_spacing_mm[1]
    factors = (row_scale, col_scale)
    return zoom(dose, factors, order=1)
