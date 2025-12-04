"""Example demonstrating QA visualization tools."""
import matplotlib.pyplot as plt
from wedge_field_dose_cal import (
    load_tiff_rgb,
    load_dicom_dose,
    resample_dose_to_film_spacing,
    register_dose_to_film,
    MultichannelDoseCalibrator,
    plot_registration_overlay,
    plot_calibration_curves,
    plot_dose_comparison,
    print_calibration_summary,
)

# Paths to calibration data
CALIB_TIFF = r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\Wedge Fields ZZZ_FILMCaAL\16h\1000MU_DQA_SBRT_20251204_05_CROPPED.tif"
CALIB_DICOM = r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\Wedge Fields ZZZ_FILMCaAL\RD.1000MU_ZZZ_FILMCAL.PlaneDose_60EDW_6x.dcm"

# Test patient data
TEST_TIFF = r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\EVA E6408934\16h\DQA_SBRT_20251204_10_CROPPED.tif"

print("Loading calibration data...")
film_rgb, film_spacing = load_tiff_rgb(CALIB_TIFF)
dose_plane, dose_spacing = load_dicom_dose(CALIB_DICOM)

print("Resampling dose to film spacing...")
dose_resampled = resample_dose_to_film_spacing(dose_plane, dose_spacing, film_spacing)

print("Registering dose to film...")
registration = register_dose_to_film(film_rgb, dose_resampled, film_spacing)

# 1. Visualize registration
print("\n=== Registration Visualization ===")
fig1 = plot_registration_overlay(registration)
fig1.savefig("qa_registration.png", dpi=150, bbox_inches="tight")
print("Saved: qa_registration.png")

# 2. Build calibrator and visualize curves
print("\n=== Calibration Curves ===")
calibrator = MultichannelDoseCalibrator(num_bins=24)
calibrator.add_pair(registration.cropped_film, registration.cropped_dose)
calibrator.fit()

print_calibration_summary(calibrator)

fig2 = plot_calibration_curves(calibrator)
fig2.savefig("qa_calibration_curves.png", dpi=150, bbox_inches="tight")
print("Saved: qa_calibration_curves.png")

# 3. Self-validation: predict dose on calibration film
print("\n=== Self-Validation ===")
predicted_dose = calibrator.predict_dose(registration.cropped_film)
fig3 = plot_dose_comparison(predicted_dose, registration.cropped_dose)
fig3.savefig("qa_self_validation.png", dpi=150, bbox_inches="tight")
print("Saved: qa_self_validation.png")

# 4. Predict on test patient
print("\n=== Test Patient Prediction ===")
test_film, test_spacing = load_tiff_rgb(TEST_TIFF)
test_pred = calibrator.predict_dose(test_film)
print(f"Test prediction range: [{test_pred.min():.3f}, {test_pred.max():.3f}] Gy")

plt.figure(figsize=(8, 6))
plt.imshow(test_pred, cmap="jet")
plt.colorbar(label="Dose (Gy)")
plt.title("Test Patient Predicted Dose")
plt.axis("off")
plt.savefig("qa_test_prediction.png", dpi=150, bbox_inches="tight")
print("Saved: qa_test_prediction.png")

print("\nDone! Check the generated PNG files for QA visualization.")
plt.show()
