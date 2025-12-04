"""QA and visualization tools for film dose calibration."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .calibration import MultichannelDoseCalibrator
    from .registration import RegistrationResult


def plot_registration_overlay(
    registration: RegistrationResult,
    figsize: tuple[float, float] = (14, 5),
) -> Figure:
    """
    Visualize the registration result with film, dose, and overlay.

    Shows three panels:
    1. Cropped film (grayscale)
    2. Aligned dose
    3. Overlay with dose contours on film
    """
    film_gray = np.mean(registration.cropped_film, axis=2)
    dose = registration.cropped_dose

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Film
    axes[0].imshow(film_gray, cmap="gray")
    axes[0].set_title("Film (grayscale)")
    axes[0].axis("off")

    # Dose
    valid_dose = np.where(np.isfinite(dose), dose, 0)
    im = axes[1].imshow(valid_dose, cmap="jet")
    axes[1].set_title("Aligned Dose")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="Dose (Gy)")

    # Overlay
    axes[2].imshow(film_gray, cmap="gray")
    # Create contour levels at 10%, 25%, 50%, 75%, 90% of max dose
    max_dose = np.nanmax(dose)
    if max_dose > 0:
        levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        contour_levels = [l * max_dose for l in levels]
        cs = axes[2].contour(valid_dose, levels=contour_levels, colors="cyan", linewidths=0.8)
        axes[2].clabel(cs, inline=True, fontsize=8, fmt="%.1f")
    axes[2].set_title("Overlay (dose contours on film)")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def plot_calibration_curves(
    calibrator: MultichannelDoseCalibrator,
    max_points: int = 5000,
    figsize: tuple[float, float] = (12, 4),
) -> Figure:
    """
    Plot pixel value vs dose for each RGB channel.

    Shows the calibration data as scatter plots with bi-exponential fits overlaid.
    """
    if not calibrator.pixels or not calibrator.doses:
        raise RuntimeError("Calibrator has no data to plot")

    X = np.concatenate(calibrator.pixels, axis=0)
    y = np.concatenate(calibrator.doses, axis=0)

    # Filter NaN values
    valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    # Subsample for plotting if too many points
    if len(y) > max_points:
        idx = np.random.choice(len(y), max_points, replace=False)
        X_plot = X[idx]
        y_plot = y[idx]
    else:
        X_plot = X
        y_plot = y

    channel_names = ["Red", "Green", "Blue"]
    channel_colors = ["red", "green", "blue"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i, (ax, name, color) in enumerate(zip(axes, channel_names, channel_colors)):
        ax.scatter(y_plot, X_plot[:, i], alpha=0.1, s=1, c=color, label="Data")

        # Overlay fitted bi-exponential curve if available
        if calibrator.channel_models is not None and calibrator.dose_range is not None:
            dose_curve = np.linspace(calibrator.dose_range[0], calibrator.dose_range[1], 200)
            pixel_curve = calibrator.channel_models[i].predict_pixel(dose_curve)
            ax.plot(dose_curve, pixel_curve, "k-", linewidth=2, label="Bi-exp fit")

        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Pixel Value")
        ax.set_title(f"{name} Channel")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_dose_comparison(
    predicted_dose: np.ndarray,
    known_dose: np.ndarray,
    figsize: tuple[float, float] = (16, 4),
) -> Figure:
    """
    Compare predicted dose to known dose with difference map.

    Shows four panels:
    1. Known dose
    2. Predicted dose
    3. Difference (predicted - known)
    4. Scatter plot of predicted vs known
    """
    # Handle NaN values
    valid_mask = np.isfinite(known_dose) & np.isfinite(predicted_dose)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Known dose
    vmax = np.nanmax(known_dose)
    im0 = axes[0].imshow(np.where(np.isfinite(known_dose), known_dose, 0),
                          cmap="jet", vmin=0, vmax=vmax)
    axes[0].set_title("Known Dose")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], label="Gy")

    # Predicted dose
    im1 = axes[1].imshow(np.where(np.isfinite(predicted_dose), predicted_dose, 0),
                          cmap="jet", vmin=0, vmax=vmax)
    axes[1].set_title("Predicted Dose")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], label="Gy")

    # Difference
    diff = predicted_dose - known_dose
    diff_valid = np.where(valid_mask, diff, 0)
    max_diff = np.nanpercentile(np.abs(diff), 99)
    im2 = axes[2].imshow(diff_valid, cmap="RdBu_r", vmin=-max_diff, vmax=max_diff)
    axes[2].set_title("Difference (Pred - Known)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], label="Gy")

    # Scatter plot
    known_flat = known_dose[valid_mask].flatten()
    pred_flat = predicted_dose[valid_mask].flatten()

    # Subsample for scatter if needed
    if len(known_flat) > 10000:
        idx = np.random.choice(len(known_flat), 10000, replace=False)
        known_flat = known_flat[idx]
        pred_flat = pred_flat[idx]

    axes[3].scatter(known_flat, pred_flat, alpha=0.1, s=1)
    max_val = max(np.max(known_flat), np.max(pred_flat))
    axes[3].plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Unity")
    axes[3].set_xlabel("Known Dose (Gy)")
    axes[3].set_ylabel("Predicted Dose (Gy)")
    axes[3].set_title("Predicted vs Known")
    axes[3].set_aspect("equal")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compute_gamma_index(
    predicted_dose: np.ndarray,
    known_dose: np.ndarray,
    pixel_spacing_mm: tuple[float, float],
    dose_threshold_pct: float = 3.0,
    distance_threshold_mm: float = 3.0,
    dose_cutoff_pct: float = 10.0,
    search_radius_mm: float = 5.0,
) -> np.ndarray:
    """
    Compute the gamma index for dose comparison.

    Parameters
    ----------
    predicted_dose : array
        The predicted/evaluated dose distribution
    known_dose : array
        The reference/known dose distribution
    pixel_spacing_mm : tuple
        (row_spacing, col_spacing) in mm
    dose_threshold_pct : float
        Dose difference threshold as percentage of max dose
    distance_threshold_mm : float
        Distance-to-agreement threshold in mm
    dose_cutoff_pct : float
        Ignore pixels below this percentage of max dose
    search_radius_mm : float
        Search radius for finding minimum gamma

    Returns
    -------
    gamma : array
        Gamma index at each pixel (NaN where below cutoff)
    """
    max_dose = np.nanmax(known_dose)
    dose_threshold = dose_threshold_pct / 100.0 * max_dose
    dose_cutoff = dose_cutoff_pct / 100.0 * max_dose

    # Create search grid
    row_spacing, col_spacing = pixel_spacing_mm
    search_rows = int(np.ceil(search_radius_mm / row_spacing))
    search_cols = int(np.ceil(search_radius_mm / col_spacing))

    gamma = np.full_like(predicted_dose, np.nan)

    rows, cols = predicted_dose.shape

    for r in range(rows):
        for c in range(cols):
            ref_dose = known_dose[r, c]

            if not np.isfinite(ref_dose) or ref_dose < dose_cutoff:
                continue

            min_gamma_sq = np.inf

            # Search in neighborhood
            for dr in range(-search_rows, search_rows + 1):
                for dc in range(-search_cols, search_cols + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        eval_dose = predicted_dose[nr, nc]
                        if not np.isfinite(eval_dose):
                            continue

                        # Distance in mm
                        dist_mm = np.sqrt((dr * row_spacing)**2 + (dc * col_spacing)**2)

                        # Dose difference
                        dose_diff = eval_dose - ref_dose

                        # Gamma squared
                        gamma_sq = (dist_mm / distance_threshold_mm)**2 + (dose_diff / dose_threshold)**2
                        min_gamma_sq = min(min_gamma_sq, gamma_sq)

            if np.isfinite(min_gamma_sq):
                gamma[r, c] = np.sqrt(min_gamma_sq)

    return gamma


def plot_gamma_analysis(
    predicted_dose: np.ndarray,
    known_dose: np.ndarray,
    pixel_spacing_mm: tuple[float, float],
    dose_threshold_pct: float = 3.0,
    distance_threshold_mm: float = 3.0,
    figsize: tuple[float, float] = (12, 4),
) -> Figure:
    """
    Compute and plot gamma index analysis.
    """
    gamma = compute_gamma_index(
        predicted_dose, known_dose, pixel_spacing_mm,
        dose_threshold_pct, distance_threshold_mm
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Gamma map
    im = axes[0].imshow(gamma, cmap="RdYlGn_r", vmin=0, vmax=2)
    axes[0].set_title(f"Gamma ({dose_threshold_pct}%/{distance_threshold_mm}mm)")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0])

    # Pass/fail map
    valid_gamma = gamma[np.isfinite(gamma)]
    pass_map = np.where(np.isfinite(gamma), gamma <= 1.0, np.nan)
    axes[1].imshow(pass_map, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_title("Pass (green) / Fail (red)")
    axes[1].axis("off")

    # Histogram
    if len(valid_gamma) > 0:
        pass_rate = np.sum(valid_gamma <= 1.0) / len(valid_gamma) * 100
        axes[2].hist(valid_gamma, bins=50, edgecolor="black", alpha=0.7)
        axes[2].axvline(1.0, color="red", linestyle="--", linewidth=2, label="γ=1")
        axes[2].set_xlabel("Gamma Index")
        axes[2].set_ylabel("Count")
        axes[2].set_title(f"Pass Rate: {pass_rate:.1f}%")
        axes[2].legend()

    plt.tight_layout()
    return fig


def print_calibration_summary(calibrator: MultichannelDoseCalibrator) -> None:
    """Print a summary of calibration statistics."""
    if calibrator.channel_models is None:
        print("Calibrator not fitted yet")
        return

    X = np.concatenate(calibrator.pixels, axis=0)
    y = np.concatenate(calibrator.doses, axis=0)
    valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)

    print("=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Total pixels: {len(y):,}")
    print(f"Valid pixels: {np.sum(valid_mask):,} ({100*np.sum(valid_mask)/len(y):.1f}%)")
    print(f"Dose range: [{calibrator.dose_range[0]:.3f}, {calibrator.dose_range[1]:.3f}] Gy")
    print(f"Covariance bins: {calibrator.num_covariance_bins}")
    print()

    print("Bi-Exponential Model: pixel = a·exp(-dose·b) + c·exp(-dose·d) + e")
    print("-" * 60)
    channel_names = ["Red", "Green", "Blue"]
    for name, model in zip(channel_names, calibrator.channel_models):
        print(f"  {name:5s}: a={model.a:8.1f}, b={model.b:.4f}, "
              f"c={model.c:8.1f}, d={model.d:.4f}, e={model.e:8.1f}")

    # Compute and show fit quality (RMSE)
    print()
    print("Fit Quality (RMSE):")
    print("-" * 60)
    pixels_valid = X[valid_mask]
    doses_valid = y[valid_mask]
    for i, (name, model) in enumerate(zip(channel_names, calibrator.channel_models)):
        predicted = model.predict_pixel(doses_valid)
        rmse = np.sqrt(np.mean((pixels_valid[:, i] - predicted) ** 2))
        print(f"  {name:5s}: RMSE = {rmse:.1f} pixel values")
