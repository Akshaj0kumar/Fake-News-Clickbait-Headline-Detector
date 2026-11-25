# Datasets Information

### Included in Repository:
- `clickbait_data.csv`: 1,600 balanced clickbait and legitimate news headlines.
- `sample_fake.csv` & `sample_true.csv`: Sample subsets from the ISOT dataset for quick testing.

### Full Datasets for Retraining:
For full training (which was used to train the included `models/misinfo.joblib` with 97% accuracy):
- Download `fake.csv` and `true.csv` from Kaggle (ISOT Fake News Dataset).
- Place them in this `data/` directory.
- Run `python train.py` to retrain both classification heads.
