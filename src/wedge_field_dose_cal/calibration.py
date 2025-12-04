from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Callable

import numpy as np
from scipy.interpolate import UnivariateSpline


@dataclass
class ChannelSpline:
    """Smoothing spline for a single channel: dose -> pixel."""

    spline: Callable[[np.ndarray], np.ndarray]
    dose_min: float
    dose_max: float

    def predict_pixel(self, dose: np.ndarray) -> np.ndarray:
        """Predict pixel value from dose."""
        dose = np.clip(np.asarray(dose), self.dose_min, self.dose_max)
        return self.spline(dose)


@dataclass
class CovarianceModel:
    """Dose-dependent covariance model using binned residuals."""

    dose_centers: np.ndarray
    covariances: List[np.ndarray]
    covariance_inverses: List[np.ndarray]

    def get_covariance(self, dose: float) -> np.ndarray:
        """Interpolate covariance matrix for a given dose."""
        if dose <= self.dose_centers[0]:
            return self.covariances[0]
        if dose >= self.dose_centers[-1]:
            return self.covariances[-1]

        upper_idx = int(np.searchsorted(self.dose_centers, dose))
        lower_idx = upper_idx - 1
        frac = (dose - self.dose_centers[lower_idx]) / (
            self.dose_centers[upper_idx] - self.dose_centers[lower_idx]
        )
        return (1 - frac) * self.covariances[lower_idx] + frac * self.covariances[upper_idx]


@dataclass
class MultichannelDoseCalibrator:
    """
    Film dose calibrator using smoothed linear interpolation.

    Bins calibration data by dose, computes mean pixel values per bin,
    then fits a smoothing spline through the bin means for each RGB channel.
    Prediction inverts the spline via lookup table interpolation.
    """

    num_bins: int = 50
    smoothing_factor: float | None = None  # None = automatic, higher = smoother

    # Stored calibration data
    pixels: List[np.ndarray] = field(default_factory=list)
    doses: List[np.ndarray] = field(default_factory=list)

    # Fitted models
    channel_splines: List[ChannelSpline] | None = None
    covariance_model: CovarianceModel | None = None
    dose_range: Tuple[float, float] | None = None
    bin_centers: np.ndarray | None = None
    bin_means: np.ndarray | None = None  # shape: (num_bins, 3)

    def add_pair(self, film_rgb: np.ndarray, dose: np.ndarray) -> None:
        """Add a film-dose pair for calibration."""
        if film_rgb.shape[:2] != dose.shape:
            raise ValueError("Film and dose shapes do not match for calibration")
        self.pixels.append(film_rgb.reshape(-1, 3).astype(np.float64))
        self.doses.append(dose.reshape(-1).astype(np.float64))

    def fit(self) -> None:
        """Fit the calibration models to all added data."""
        if not self.pixels:
            raise RuntimeError("No calibration pairs added")

        # Concatenate all data
        all_pixels = np.concatenate(self.pixels, axis=0)
        all_doses = np.concatenate(self.doses, axis=0)

        # Filter out invalid data
        valid_mask = np.isfinite(all_doses) & np.all(np.isfinite(all_pixels), axis=1)
        pixels = all_pixels[valid_mask]
        doses = all_doses[valid_mask]

        self.dose_range = (float(np.min(doses)), float(np.max(doses)))

        # Bin data and compute means
        self._compute_bin_statistics(pixels, doses)

        # Fit smoothing spline for each channel
        self.channel_splines = []
        for ch in range(3):
            spline = self._fit_channel_spline(ch)
            self.channel_splines.append(spline)

        # Compute dose-dependent covariance from residuals
        self.covariance_model = self._fit_covariance_model(pixels, doses)

    def _compute_bin_statistics(self, pixels: np.ndarray, doses: np.ndarray) -> None:
        """Compute binned mean pixel values."""
        dose_min, dose_max = self.dose_range
        bins = np.linspace(dose_min, dose_max, self.num_bins + 1)
        bin_indices = np.clip(np.digitize(doses, bins) - 1, 0, self.num_bins - 1)

        self.bin_centers = 0.5 * (bins[:-1] + bins[1:])
        self.bin_means = np.zeros((self.num_bins, 3))
        bin_counts = np.zeros(self.num_bins)

        for i in range(self.num_bins):
            mask = bin_indices == i
            count = np.sum(mask)
            bin_counts[i] = count
            if count > 0:
                self.bin_means[i] = np.mean(pixels[mask], axis=0)

        # Fill empty bins by interpolation
        for ch in range(3):
            valid = bin_counts > 0
            if not np.all(valid) and np.any(valid):
                self.bin_means[~valid, ch] = np.interp(
                    self.bin_centers[~valid],
                    self.bin_centers[valid],
                    self.bin_means[valid, ch]
                )

    def _fit_channel_spline(self, channel: int) -> ChannelSpline:
        """Fit a smoothing spline for one channel."""
        # Use smoothing spline through bin means
        # s=None lets scipy choose automatically based on data
        s = self.smoothing_factor
        if s is None:
            # Auto smoothing: use a fraction of the sum of squared differences
            s = len(self.bin_centers) * 0.1  # mild smoothing

        spline = UnivariateSpline(
            self.bin_centers,
            self.bin_means[:, channel],
            s=s,
            k=3,  # cubic spline
        )

        return ChannelSpline(
            spline=spline,
            dose_min=self.dose_range[0],
            dose_max=self.dose_range[1],
        )

    def _fit_covariance_model(self, pixels: np.ndarray, doses: np.ndarray) -> CovarianceModel:
        """Fit dose-dependent covariance model from residuals."""
        # Compute residuals (measured - predicted from spline)
        predicted = self.get_expected_pixel(doses)
        residuals = pixels - predicted

        # Bin by dose and compute covariance in each bin
        dose_min, dose_max = self.dose_range
        num_cov_bins = min(24, self.num_bins)
        bins = np.linspace(dose_min, dose_max, num_cov_bins + 1)
        bin_indices = np.clip(np.digitize(doses, bins) - 1, 0, num_cov_bins - 1)
        centers = 0.5 * (bins[:-1] + bins[1:])

        covariances: List[np.ndarray] = []
        cov_inverses: List[np.ndarray] = []

        # Compute global fallback covariance
        global_cov = np.cov(residuals.T)
        if global_cov.ndim == 0:
            global_cov = np.eye(3) * global_cov

        for i in range(num_cov_bins):
            mask = bin_indices == i
            if np.sum(mask) < 10:
                covariances.append(global_cov)
            else:
                bin_residuals = residuals[mask]
                cov = np.cov(bin_residuals.T)
                if cov.ndim == 0:
                    cov = np.eye(3) * cov
                covariances.append(cov)

            # Regularize and invert
            cov_reg = covariances[-1] + np.eye(3) * 1e-6
            cov_inverses.append(np.linalg.inv(cov_reg))

        return CovarianceModel(
            dose_centers=centers,
            covariances=covariances,
            covariance_inverses=cov_inverses,
        )

    def predict_dose(self, film_rgb: np.ndarray) -> np.ndarray:
        """Predict dose from film RGB values using lookup table interpolation."""
        if self.channel_splines is None or self.dose_range is None:
            raise RuntimeError("Calibrator must be fitted before predicting")

        flat_pixels = film_rgb.reshape(-1, 3).astype(np.float64)

        # Create lookup table (dose -> pixel for each channel)
        dose_min, dose_max = self.dose_range
        dose_lut = np.linspace(dose_min, dose_max, 1000)

        # Get per-channel dose estimates via interpolation
        channel_doses = np.empty((flat_pixels.shape[0], 3), dtype=np.float64)

        for ch, spline in enumerate(self.channel_splines):
            pixel_lut = spline.predict_pixel(dose_lut)
            # Reverse arrays since pixel decreases with dose
            channel_doses[:, ch] = np.interp(
                flat_pixels[:, ch],
                pixel_lut[::-1],
                dose_lut[::-1]
            )

        # Combine channel estimates using weighted average
        weights = np.array([1.0, 0.8, 0.5])
        predicted = np.average(channel_doses, axis=1, weights=weights)

        return predicted.reshape(film_rgb.shape[:2])

    def get_expected_pixel(self, dose: float | np.ndarray) -> np.ndarray:
        """Get expected RGB pixel values for a given dose."""
        if self.channel_splines is None:
            raise RuntimeError("Calibrator must be fitted first")
        dose = np.asarray(dose)
        return np.column_stack([s.predict_pixel(dose) for s in self.channel_splines])
