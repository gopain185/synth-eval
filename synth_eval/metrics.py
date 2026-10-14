"""Four families of metrics for synthetic tabular data, each returning plain floats.

fidelity        does each column, and each pair of columns, look like the real data?
detectability   can a classifier tell synthetic rows from real ones? (0.5 = indistinguishable)
utility         does a model trained on synthetic data work on real data? (train-synthetic-test-real)
privacy         are synthetic rows suspiciously close to, or exact copies of, real training rows?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors


def _is_cat(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s)


def _binned(real: pd.Series, other: pd.Series, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if _is_cat(real) or real.nunique() <= bins:
        cats = pd.Index(real.dropna().unique()).union(pd.Index(other.dropna().unique()))
        p = real.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        q = other.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        return p, q
    edges = np.unique(np.nanquantile(real, np.linspace(0, 1, bins + 1)))
    p = np.histogram(real.dropna(), edges)[0] / real.notna().sum()
    q = np.histogram(other.dropna().clip(edges[0], edges[-1]), edges)[0] / max(other.notna().sum(), 1)
    return p, q


def marginal_tvd(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Mean total-variation distance between per-column distributions (0 = identical)."""
    return float(np.mean([0.5 * np.abs(np.subtract(*_binned(real[c], synth[c]))).sum() for c in real.columns]))


def _codes(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in ref.columns:
        if _is_cat(ref[c]):
            out[c] = pd.Categorical(df[c], categories=ref[c].value_counts().index).codes.astype(float)
        else:
            out[c] = df[c].astype(float).fillna(ref[c].median())
    return pd.DataFrame(out)


def association_error(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Mean absolute difference between pairwise Spearman correlation matrices (0 = same dependency structure)."""
    a = _codes(real, real).corr(method="spearman").to_numpy()
    b = _codes(synth, real).corr(method="spearman").to_numpy()
    iu = np.triu_indices_from(a, k=1)
    return float(np.nanmean(np.abs(a[iu] - b[iu])))


def detectability_auc(real: pd.DataFrame, synth: pd.DataFrame, seed: int = 0) -> float:
    """Cross-validated AUC of a real-vs-synthetic classifier. 0.5 is ideal; 1.0 means trivially detectable."""
    n = min(len(real), len(synth))
    both = pd.concat([real.sample(n, random_state=seed), synth.sample(n, random_state=seed)], ignore_index=True)
    X = _codes(both, real)
    label = np.r_[np.zeros(n), np.ones(n)]
    cat_mask = [_is_cat(real[c]) for c in real.columns]
    clf = HistGradientBoostingClassifier(max_iter=200, categorical_features=cat_mask, random_state=seed)
    p = cross_val_predict(clf, X, label, cv=3, method="predict_proba")[:, 1]
    return float(roc_auc_score(label, p))


def tstr_auc(synth: pd.DataFrame, real_test: pd.DataFrame, target: str, seed: int = 0) -> float:
    """Train on synthetic, test on real: ROC AUC of a model fitted only on synthetic rows."""
    feats = [c for c in real_test.columns if c != target]
    ref = real_test[feats]
    cat_mask = [_is_cat(ref[c]) for c in feats]
    clf = HistGradientBoostingClassifier(max_iter=300, categorical_features=cat_mask, random_state=seed)
    clf.fit(_codes(synth[feats], ref), synth[target].astype(int))
    return float(roc_auc_score(real_test[target].astype(int), clf.predict_proba(_codes(ref, ref))[:, 1]))


def privacy(synth: pd.DataFrame, real_train: pd.DataFrame, real_holdout: pd.DataFrame, n: int = 5000, seed: int = 0) -> dict:
    """Distance to closest record (DCR) of synthetic rows vs. genuinely unseen real rows.

    dcr_ratio = median DCR(synthetic → train) / median DCR(holdout → train). A ratio well below 1 means the
    generator is reproducing training rows more closely than new real data would, i.e. memorisation.
    """
    ref = real_train
    scale = _codes(ref, ref).std().replace(0, 1)
    nn = NearestNeighbors(n_neighbors=1).fit(_codes(ref, ref) / scale)
    s = synth.sample(min(n, len(synth)), random_state=seed)
    h = real_holdout.sample(min(n, len(real_holdout)), random_state=seed)
    d_s = nn.kneighbors(_codes(s, ref) / scale)[0][:, 0]
    d_h = nn.kneighbors(_codes(h, ref) / scale)[0][:, 0]
    # The median hides memorisation when only a minority of rows are copied, so also look at the tail:
    # how many synthetic rows sit closer to a training row than the closest 5% of genuinely new real rows?
    tail = np.quantile(d_h, 0.05)
    return {
        "dcr_median": float(np.median(d_s)),
        "dcr_ratio": float(np.median(d_s) / max(np.median(d_h), 1e-9)),
        "close_rate": float((d_s <= tail).mean()),  # 0.05 expected for a generator that does not memorise
        "exact_copy_rate": float((d_s < 1e-9).mean()),
    }


def evaluate(synth: pd.DataFrame, real_train: pd.DataFrame, real_test: pd.DataFrame, target: str) -> dict:
    return {
        "marginal_tvd": marginal_tvd(real_test, synth),
        "association_error": association_error(real_test, synth),
        "detectability_auc": detectability_auc(real_test, synth),
        "tstr_auc": tstr_auc(synth, real_test, target),
        **privacy(synth, real_train, real_test),
    }
