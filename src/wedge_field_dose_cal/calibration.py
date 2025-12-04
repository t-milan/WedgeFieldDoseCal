from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


def compute_log_optical_density(film_rgb: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Compute log10 optical density per channel."""
    if reference is None:
        reference = np.percentile(film_rgb, 99, axis=(0, 1))
    reference = np.maximum(reference, 1.0)
    return np.log10(reference / np.clip(film_rgb, 1.0, None))


@dataclass
class BinStatistics:
    """Dose-binned multichannel statistics used for Mahalanobis scoring."""

    dose_centers: np.ndarray
    means: List[np.ndarray]
    covariances: List[np.ndarray]
    covariance_inverses: List[np.ndarray]


@dataclass
class MultichannelDoseCalibrator:
    num_bins: int = 24
    reference_intensity: np.ndarray | None = None
    features: List[np.ndarray] = field(default_factory=list)
    doses: List[np.ndarray] = field(default_factory=list)
    bin_stats: BinStatistics | None = None

    def add_pair(self, film_rgb: np.ndarray, dose: np.ndarray) -> None:
        if film_rgb.shape[:2] != dose.shape:
            raise ValueError("Film and dose shapes do not match for calibration")
        if self.reference_intensity is None:
            self.reference_intensity = np.percentile(film_rgb, 99, axis=(0, 1))
        log_od = compute_log_optical_density(film_rgb, self.reference_intensity)
        self.features.append(log_od.reshape(-1, 3))
        self.doses.append(dose.reshape(-1))

    def fit(self) -> None:
        if not self.features:
            raise RuntimeError("No calibration pairs added")
        X = np.concatenate(self.features, axis=0)
        y = np.concatenate(self.doses, axis=0)
        self.bin_stats = self._build_bin_statistics(X, y)

    def _build_bin_statistics(self, X: np.ndarray, y: np.ndarray) -> BinStatistics:
        dose_min, dose_max = float(np.min(y)), float(np.max(y))
        bins = np.linspace(dose_min, dose_max, self.num_bins + 1)
        bin_indices = np.digitize(y, bins) - 1
        centers = 0.5 * (bins[:-1] + bins[1:])
        covariances: List[np.ndarray] = []
        means: List[np.ndarray] = []
        cov_inverses: List[np.ndarray] = []
        fallback_mean = np.mean(X, axis=0)
        for i in range(self.num_bins):
            mask = bin_indices == i
            if np.sum(mask) < 4:
                means.append(fallback_mean)
                covariances.append(np.eye(3) * 1e-3)
            else:
                subset = X[mask]
                means.append(np.mean(subset, axis=0))
                covariances.append(np.cov(subset.T))
            cov_inverses.append(np.linalg.pinv(covariances[-1]))
        return BinStatistics(
            dose_centers=centers,
            means=means,
            covariances=covariances,
            covariance_inverses=cov_inverses,
        )

    def predict_dose(self, film_rgb: np.ndarray) -> np.ndarray:
        if self.reference_intensity is None or self.bin_stats is None:
            raise RuntimeError("Calibrator must be fitted before predicting")
        log_od = compute_log_optical_density(film_rgb, self.reference_intensity)
        flat_features = log_od.reshape(-1, 3)
        predicted = self._dose_from_mahalanobis(flat_features)
        return predicted.reshape(film_rgb.shape[:2])

    def _dose_from_mahalanobis(self, flat_features: np.ndarray) -> np.ndarray:
        assert self.bin_stats is not None
        centers = self.bin_stats.dose_centers
        means = np.stack(self.bin_stats.means, axis=0)
        inv_cov = np.stack(self.bin_stats.covariance_inverses, axis=0)

        # Compute squared Mahalanobis distance to every bin for each pixel
        diff = flat_features[:, None, :] - means[None, :, :]
        mahal_sq = np.einsum("nbc,bcd,nbd->nb", diff, inv_cov, diff)
        weights = np.exp(-0.5 * mahal_sq)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums = np.clip(weight_sums, 1e-12, None)
        weighted_dose = weights @ centers
        return (weighted_dose / weight_sums).ravel()

    def dose_uncertainty_map(self, predicted_dose: np.ndarray) -> np.ndarray:
        if self.bin_stats is None:
            raise RuntimeError("Calibrator must be fitted before computing uncertainty")
        covariances = self._interpolate_covariances(predicted_dose.ravel())
        variances = np.array([np.trace(cov) for cov in covariances])
        return variances.reshape(predicted_dose.shape)

    def _interpolate_covariances(self, dose_values: np.ndarray) -> List[np.ndarray]:
        assert self.bin_stats is not None
        dose_values = np.asarray(dose_values, dtype=float)
        centers = self.bin_stats.dose_centers
        covariances = self.bin_stats.covariances
        interpolated: List[np.ndarray] = []
        for dose in dose_values:
            if dose <= centers[0]:
                interpolated.append(covariances[0])
                continue
            if dose >= centers[-1]:
                interpolated.append(covariances[-1])
                continue
            upper_idx = np.searchsorted(centers, dose)
            lower_idx = upper_idx - 1
            frac = (dose - centers[lower_idx]) / (centers[upper_idx] - centers[lower_idx])
            cov_low = covariances[lower_idx]
            cov_high = covariances[upper_idx]
            interpolated.append((1 - frac) * cov_low + frac * cov_high)
        return interpolated
