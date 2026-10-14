"""Simple tabular data generators, each fitted on a real training frame.

They are deliberately simple and transparent so the evaluation harness can show *what* each metric catches:
- IndependentMarginals preserves every column distribution but destroys all dependencies.
- GaussianCopula preserves marginals and (rank) correlations, but not higher-order structure.
- NoisyCopy resamples real rows and re-draws a fraction of cells from the marginals. It looks excellent on
  fidelity and utility and fails on privacy. This is roughly how many "synthetic" competition datasets behave.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class Generator:
    name = "base"

    def fit(self, real: pd.DataFrame) -> "Generator":
        self.columns = list(real.columns)
        self.real = real.reset_index(drop=True)
        return self

    def sample(self, n: int, seed: int = 0) -> pd.DataFrame:
        raise NotImplementedError


class IndependentMarginals(Generator):
    name = "independent"

    def sample(self, n: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({c: self.real[c].to_numpy()[rng.integers(0, len(self.real), n)] for c in self.columns})


class GaussianCopula(Generator):
    """Map each column to normal scores through its empirical CDF, fit a correlation matrix, sample, invert.

    Categorical columns are integer-coded by frequency first, which makes the copula treat them as ordinal.
    This is a known weakness that the dependency metrics should expose.
    """

    name = "gaussian_copula"

    def fit(self, real: pd.DataFrame) -> "GaussianCopula":
        super().fit(real)
        self.categories = {}
        codes = {}
        for c in self.columns:
            col = self.real[c]
            if not pd.api.types.is_numeric_dtype(col):
                order = col.value_counts().index
                self.categories[c] = order
                codes[c] = pd.Categorical(col, categories=order).codes.astype(float)
            else:
                codes[c] = col.astype(float).fillna(col.median()).to_numpy()
        self.encoded = pd.DataFrame(codes)
        n = len(self.encoded)
        z = np.column_stack([stats.norm.ppf((stats.rankdata(self.encoded[c]) - 0.5) / n) for c in self.columns])
        self.corr = np.corrcoef(z, rowvar=False)
        self.sorted = {c: np.sort(self.encoded[c].to_numpy()) for c in self.columns}
        return self

    def sample(self, n: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        z = rng.multivariate_normal(np.zeros(len(self.columns)), self.corr, size=n)
        u = stats.norm.cdf(z)
        out = {}
        for j, c in enumerate(self.columns):
            ref = self.sorted[c]
            vals = ref[np.clip((u[:, j] * len(ref)).astype(int), 0, len(ref) - 1)]
            if c in self.categories:
                out[c] = self.categories[c][vals.astype(int)]
            else:
                out[c] = vals.astype(self.real[c].dtype) if self.real[c].notna().all() else vals
        return pd.DataFrame(out)


class NoisyCopy(Generator):
    name = "noisy_copy"

    def __init__(self, noise: float = 0.15):
        self.noise = noise
        self.name = f"noisy_copy_{int(noise * 100)}"

    def sample(self, n: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        base = self.real.iloc[rng.integers(0, len(self.real), n)].reset_index(drop=True).copy()
        for c in self.columns:
            mask = rng.random(n) < self.noise
            base.loc[mask, c] = self.real[c].to_numpy()[rng.integers(0, len(self.real), mask.sum())]
        return base


class HoldoutReal(Generator):
    """Reference point: genuinely new real rows. Every metric should look 'ideal' here."""

    name = "holdout_real"

    def __init__(self, holdout: pd.DataFrame):
        self.holdout = holdout.reset_index(drop=True)

    def sample(self, n: int, seed: int = 0) -> pd.DataFrame:
        return self.holdout.sample(n=min(n, len(self.holdout)), random_state=seed).reset_index(drop=True)
