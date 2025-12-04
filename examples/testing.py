from wedge_field_dose_cal import calibrate_from_files, load_tiff_rgb, print_calibration_summary, plot_calibration_curves

calibrator = calibrate_from_files(
    [r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\Wedge Fields ZZZ_FILMCaAL\16h\500MU_DQA_SBRT_20251204_02_CROPPED.tif",
     r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\Wedge Fields ZZZ_FILMCaAL\16h\1000MU_DQA_SBRT_20251204_05_CROPPED.tif",
     r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\Wedge Fields ZZZ_FILMCaAL\16h\2000MU_DQA_SBRT_20251204_06_CROPPED.tif",
     r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\Wedge Fields ZZZ_FILMCaAL\16h\4000MU_DQA_SBRT_20251204_08_CROPPED.tif"],
    [r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\Wedge Fields ZZZ_FILMCaAL\RD.500MU_ZZZ_FILMCAL.PlaneDose_60EDW_6x.dcm",
     r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\Wedge Fields ZZZ_FILMCaAL\RD.1000MU_ZZZ_FILMCAL.PlaneDose_60EDW_6x.dcm",
     r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\Wedge Fields ZZZ_FILMCaAL\RD.2000MU_ZZZ_FILMCAL.PlaneDose_60EDW_6x.dcm",
     r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\Wedge Fields ZZZ_FILMCaAL\RD.4000MU_ZZZ_FILMCAL.PlaneDose_60EDW_6x.dcm"],
     margin_mm=1.0#num_bins=36
)

print_calibration_summary(calibrator)
import matplotlib.pyplot as plt
fig2 = plot_calibration_curves(calibrator)
plt.show()

test_patient_path = r"W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\EVA E6408934\16h\DQA_SBRT_20251204_09_CROPPED.tif"
test_patient = load_tiff_rgb(test_patient_path)

pred_dose = calibrator.predict_dose(test_patient[0])

import tifffile
import numpy as np
tifffile.imwrite(r'W:\Radiation Oncology\SCG\RadoncPhysics\Dosimetry & Equipment\Equipment and Calibrations\Film+Scanner\Film Calibration\Calibration 2025-12-01 Batch 12032405\Validation\EVA E6408934\16h\DQA_SBRT_20251204_09_CROPPED_pred2.tif', pred_dose.astype(np.float32))
