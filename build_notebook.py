"""Builds kernel/synth-eval.ipynb: a self-contained Kaggle notebook (CPU only) from the synth_eval package."""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _module(path: str) -> str:
    src = (HERE / path).read_text(encoding="utf-8")
    src = src.replace("from __future__ import annotations\n", "")
    return re.sub(r"^from synth_eval\..*\n", "", src, flags=re.M)


cells = [
    ("markdown", """# How do you know synthetic tabular data is any good?

A small, transparent **evaluation harness for synthetic tabular data**, applied to the public
[Airline Satisfaction](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k) survey data (CC0).

I fit four deliberately simple generators on 60k real rows and score each one with four families of metrics.
A real hold-out set serves as the reference for what "ideal" looks like on every metric.

| family | question | ideal |
| --- | --- | --- |
| fidelity | do columns (and pairs of columns) look real? | TVD → 0, association error → 0 |
| detectability | can a classifier tell synthetic from real? | AUC → 0.5 |
| utility | does a model trained on synthetic work on real data (TSTR)? | close to train-real-test-real |
| privacy | are synthetic rows copies of training rows? | close_rate ≈ 0.05, exact copies → 0 |

The point is not the generators. It is showing **which metric catches which failure**, and that every
single metric can be fooled on its own.
"""),
    ("code", "import numpy as np, pandas as pd, glob\nfrom scipy import stats\n"),
    ("markdown", "## Generators"),
    ("code", _module("synth_eval/generators.py")),
    ("markdown", "## Metrics"),
    ("code", _module("synth_eval/metrics.py")),
    ("markdown", "## Run"),
    ("code", """TARGET = "satisfaction"
real = pd.read_csv(glob.glob("/kaggle/input/**/data.csv", recursive=True)[0]).sample(frac=1, random_state=0).reset_index(drop=True)
real[TARGET] = real[TARGET].astype(int)
train, test, holdout = real.iloc[:60_000], real.iloc[60_000:90_000], real.iloc[90_000:]

rows = []
for g in [HoldoutReal(holdout), IndependentMarginals(), GaussianCopula(), NoisyCopy(0.15), NoisyCopy(0.5)]:
    synth = (g if isinstance(g, HoldoutReal) else g.fit(train)).sample(30_000, seed=0)
    rows.append({"generator": g.name, **evaluate(synth, train, test, TARGET)})
results = pd.DataFrame(rows).set_index("generator")
results.round(4)
"""),
    ("markdown", """## Reading the table

- **Independent marginals** gets near-perfect per-column fidelity (TVD ≈ 0.008, same as real data), yet a
  model trained on it is useless on real data (TSTR AUC ≈ 0.47) and a classifier spots it almost perfectly.
  *Per-column fidelity alone says nothing about usefulness.*
- **Gaussian copula** fixes most pairwise structure (association error drops roughly 5×) and recovers a lot of
  utility, but is still trivially detectable: rank correlations are not the whole joint distribution, and
  integer-coding categoricals forces a fake ordering on them.
- **Noisy copy (15%)** looks best on fidelity and utility (TSTR ≈ 0.987 vs 0.994 for real data) and is the
  hardest to detect. It is also a **privacy failure**: about 5% of its rows are exact copies of training rows,
  and about 15% sit closer to a training row than the closest 5% of genuinely new passengers.
  The *median* distance-to-closest-record ratio (≈ 1.2) does **not** reveal this. Only the tail metric does.
- **Noisy copy (50%)** trades that leak for much worse dependency structure and detectability.

**Takeaway for anyone building or judging synthetic data:** report all four families, include a real hold-out
reference row, and look at the tail of the privacy distribution, not just its median.
"""),
    ("code", """import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
r = results.drop(index="holdout_real")
ref = results.loc["holdout_real"]
for ax, (col, title) in zip(axes, [("detectability_auc", "Detectability AUC (0.5 = ideal)"),
                                    ("tstr_auc", "Train-synthetic-test-real AUC"),
                                    ("close_rate", "Privacy: close_rate (0.05 = ideal)")]):
    r[col].plot.barh(ax=ax, color="#4c78a8")
    ax.axvline(ref[col], color="black", ls="--", lw=1, label="real hold-out")
    ax.set_title(title); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
plt.tight_layout(); plt.show()
"""),
]


def build() -> None:
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, s in cells
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = HERE / "kernel"
    out.mkdir(exist_ok=True)
    (out / "synth-eval.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": "locdunhat/evaluating-synthetic-tabular-data",
        "title": "Evaluating Synthetic Tabular Data",
        "code_file": "synth-eval.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": ["arseniyshutko/binary-aviation-satisfaction-129k"],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    build()
