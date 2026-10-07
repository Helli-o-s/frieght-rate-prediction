# Freight Rate Prediction

LightGBM model (log-target, Huber loss) with distance/geo sanity features, daily market context, and forward-chaining time validation.

## Run
```bash
python -m pip install -r requirements.txt
python train.py   # prints time-fold MAPE; writes validation_predictions.csv and december_predictions.csv
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```
Input CSVs are expected in this folder (train-test.csv, validation.csv, december-chart-inputs.csv).

## Validation
Forward-chaining folds (train on earlier months, test on the next two), mirroring the strictly-later Nov-Dec validation set:
- train < Jul 2025 -> Jul-Aug: MAPE 5.77%, MAE $130
- train < Sep 2025 -> Sep-Oct: MAPE 4.86%, MAE $111
