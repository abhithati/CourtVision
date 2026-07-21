"""Model training, calibration, and persistence.

Split by season (not randomly) to avoid leaking future games into training.
Calibrate probabilities — a "70%" prediction should be right ~70% of the time.
"""
