from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pydicom
import tifffile


def estimate_tiff_spacing_mm(path: str | Path) -> Tuple[float, float]:
    """
    Estimate the pixel spacing (row, col) in millimeters from TIFF resolution metadata.

    The TIFF is expected to contain XResolution/YResolution tags and a ResolutionUnit of inch or cm.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = page.tags
        x_res_tag = tags.get("XResolution")
        y_res_tag = tags.get("YResolution")
        unit_tag = tags.get("ResolutionUnit")

        if x_res_tag is None or y_res_tag is None or unit_tag is None:
            raise ValueError(f"TIFF {path} lacks resolution metadata needed for spacing calculation")

        x_res = x_res_tag.value  # typically (num, den)
        y_res = y_res_tag.value
        unit_value = unit_tag.value

        def rational_to_float(val: tuple[int, int] | float) -> float:
            if isinstance(val, tuple):
                num, den = val
                return float(num) / float(den)
            return float(val)

        x_dpi = rational_to_float(x_res)
        y_dpi = rational_to_float(y_res)

        if unit_value == 1:
            raise ValueError(f"TIFF {path} ResolutionUnit=1 (no-unit) cannot be converted to spacing")
        elif unit_value == 2:  # inch
            mm_per_unit = 25.4
        elif unit_value == 3:  # centimeter
            mm_per_unit = 10.0
        else:
            raise ValueError(f"Unknown ResolutionUnit {unit_value} in TIFF {path}")

        row_spacing_mm = mm_per_unit / y_dpi
        col_spacing_mm = mm_per_unit / x_dpi
        return float(row_spacing_mm), float(col_spacing_mm)


def load_tiff_rgb(path: str | Path) -> tuple[np.ndarray, tuple[float, float]]:
    """Load an RGB TIFF and return the array and (row_spacing_mm, col_spacing_mm)."""
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        image = tif.asarray()
    spacing = estimate_tiff_spacing_mm(path)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"TIFF {path} is expected to be RGB with shape (H, W, 3)")
    return image.astype(np.float32), spacing


def load_dicom_dose(path: str | Path) -> tuple[np.ndarray, tuple[float, float]]:
    """Load an RTDOSE DICOM file and return the dose array and pixel spacing in mm."""
    ds = pydicom.dcmread(path)
    dose = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "DoseGridScaling"):
        dose = dose * float(ds.DoseGridScaling)

    if not hasattr(ds, "PixelSpacing"):
        raise ValueError(f"DICOM dose file {path} missing PixelSpacing")
    pixel_spacing = tuple(map(float, ds.PixelSpacing))
    return dose, (pixel_spacing[0], pixel_spacing[1])
