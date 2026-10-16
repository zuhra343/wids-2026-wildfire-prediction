# 🔥 Wildfire Threat Probability Modeling (WiDS Global Datathon 2026)

Predicting the probability that a wildfire reaches within 5 km of an
evacuation-zone centroid within **12, 24, 48 and 72 hours**, using features
from the first 5 hours of fire perimeter observations.

## The problem
Evacuation decisions need calibrated probabilities, not a single
"time to hit" guess. The data is **right-censored**: some fires stopped
being observed before reaching the threat zone, so treating them as "safe"
at every horizon would bias the model.

## Approach
- **Censor-aware targets per horizon:** for 12/24/48h, a fire is used only if
  its status by that horizon is known; fires censored earlier are excluded.
  Per the dataset definition (event = 0 means no hit within the 72h window),
  non-events are treated as known negatives at the 72h boundary.
- **One probability model per horizon**, instead of converting a predicted
  time into probabilities with an arbitrary curve.
- **Domain features:** log distance to the zone and closing-speed / distance ratio,
  on top of 34 growth, distance, kinematics and directionality features.
- **Models:** equal-weight XGBoost + LightGBM ensemble with median imputation.
- **Monotonic probabilities:** a row-wise cumulative max ensures
  P(hit by 12h) ≤ P(24h) ≤ P(48h) ≤ P(72h).
- **Submission checks:** ID alignment, no duplicates, values in [0, 1], monotonicity.

## Results (20% hold-out validation)
| Horizon | Training rows | Validation rows | Brier ↓ | ROC AUC ↑ |
|---|---|---|---|---|
| 12h | 215 | 45 | 0.073 | 0.967 |
| 24h | 196 | 39 | 0.039 | 0.970 |
| 48h | 166 | 34 | 0.015 | 1.000 |
| 72h | 221 | 45 | 0.001 | 1.000 |

## Limitations
- **Small data:** 221 training fires and 34–45 validation rows per horizon.
  Perfect AUCs at 48h/72h are likely small-sample effects, not proof of a perfect model.
- Random split only; grouped validation by incident or region would better test generalization.
- Local Brier scores don't exactly match the competition's censor-aware metric.
- Ensemble weights are a baseline and were not tuned.

## Next steps
Cross-validation with repeated splits, calibration curves, and a survival-model
baseline (e.g. Cox or random survival forest) for comparison.

## Run it
pip install -r requirements.txt
Download the competition data from Kaggle into this folder, then open
`WiDS_Wildfire_Risk_Modeling.ipynb`.

**Tools:** Python · pandas · scikit-learn · XGBoost · LightGBM
