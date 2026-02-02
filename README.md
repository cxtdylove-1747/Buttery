# Buttery Model Validation

This repository contains the dataset `T4_clean.csv` and a model diagram `model.png`. The script below shows how to validate a 2nd order RC battery model from the phone dataset.

## Usage

```bash
python validate_rc_model.py --csv T4_clean.csv --output validation_results.png
```

The script will:

1. Select the longest discharge segment (`Battery_charge_type=0` and `Battery_status=3`).
2. Estimate battery capacity from the SOC drop and current integration.
3. Fit a 2nd order RC model (R0, R1, C1, R2, C2).
4. Build a SOC-OCV curve from low-current points.
5. Estimate SOC using an EKF and compare with ampere-hour integration.

The output figure (`validation_results.png`) includes voltage and SOC comparison plots.
