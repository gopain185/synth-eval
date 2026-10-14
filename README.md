# synth-eval: evaluating synthetic tabular data

A small, transparent harness that scores synthetic tabular data on four families of metrics and shows
**which metric catches which failure**. Kaggle notebook:
https://www.kaggle.com/code/locdunhat/evaluating-synthetic-tabular-data

| family | question | ideal |
| --- | --- | --- |
| fidelity | do columns, and pairs of columns, look real? | marginal TVD → 0, association error → 0 |
| detectability | can a classifier tell synthetic from real? | AUC → 0.5 |
| utility | does a model trained on synthetic data work on real data (TSTR)? | close to train-real-test-real |
| privacy | are synthetic rows copies of training rows? | close_rate ≈ 0.05, exact copies → 0 |

## Results

Data: [Airline Satisfaction](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k) (CC0),
60k train / 30k test / rest hold-out, 30k synthetic rows per generator. `holdout_real` is the reference row.

| generator | marginal_tvd | association_error | detectability_auc | tstr_auc | dcr_ratio | close_rate | exact_copy_rate |
|---|---|---|---|---|---|---|---|
| holdout_real | 0.007 | 0.007 | 0.503 | 0.994 | 1.000 | 0.048 | 0.000 |
| independent | 0.008 | 0.133 | 0.998 | 0.473 | 2.021 | 0.000 | 0.000 |
| gaussian_copula | 0.009 | 0.028 | 0.989 | 0.920 | 1.625 | 0.000 | 0.000 |
| noisy_copy_15 | 0.008 | 0.038 | 0.823 | 0.987 | 1.226 | 0.153 | 0.051 |
| noisy_copy_50 | 0.007 | 0.100 | 0.987 | 0.969 | 1.865 | 0.001 | 0.000 |

- Independent marginals match every column (TVD equal to real data) yet are useless for modelling (TSTR 0.47).
- The 15% noisy copy looks best on fidelity and utility, but 5% of its rows are exact training rows.
  The median distance-to-closest-record ratio (1.23) misses this; the tail metric `close_rate` (0.153 vs 0.05) catches it.

## Usage

```bash
pip install numpy pandas scipy scikit-learn pytest
python run.py path/to/data.csv          # writes results.csv
pytest                                   # metric sanity tests
python build_notebook.py                 # regenerates the Kaggle notebook in kernel/
```
