"""Fit every generator on a real training split and score it with the full metric suite."""

import argparse
from pathlib import Path

import pandas as pd

from synth_eval.generators import GaussianCopula, HoldoutReal, IndependentMarginals, NoisyCopy
from synth_eval.metrics import evaluate

TARGET = "satisfaction"


def main(csv: Path, n: int, seed: int) -> pd.DataFrame:
    real = pd.read_csv(csv).sample(frac=1, random_state=seed).reset_index(drop=True)
    real[TARGET] = real[TARGET].astype(int)
    train, test, holdout = real.iloc[:60_000], real.iloc[60_000:90_000], real.iloc[90_000:]
    generators = [HoldoutReal(holdout), IndependentMarginals(), GaussianCopula(), NoisyCopy(0.15), NoisyCopy(0.5)]
    rows = []
    for g in generators:
        synth = (g if isinstance(g, HoldoutReal) else g.fit(train)).sample(n, seed=seed)
        scores = evaluate(synth, train, test, TARGET)
        rows.append({"generator": g.name, **scores})
        print(g.name, {k: round(v, 4) for k, v in scores.items()}, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--n", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.csv, args.n, args.seed).to_csv("results.csv", index=False)
