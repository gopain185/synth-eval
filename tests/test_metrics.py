import numpy as np
import pandas as pd
import pytest

from synth_eval.generators import IndependentMarginals, NoisyCopy
from synth_eval.metrics import association_error, detectability_auc, marginal_tvd, privacy, tstr_auc


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(0)
    n = 6000
    a = rng.normal(size=n)
    df = pd.DataFrame({
        "a": a,
        "b": a * 0.8 + rng.normal(scale=0.6, size=n),
        "c": rng.choice(["x", "y", "z"], size=n),
        "t": (a + rng.normal(scale=0.5, size=n) > 0).astype(int),
    })
    return df.iloc[:3000].reset_index(drop=True), df.iloc[3000:].reset_index(drop=True)


def test_identical_data_scores_ideal(frames):
    train, test = frames
    assert marginal_tvd(test, test) == pytest.approx(0)
    assert association_error(test, test) == pytest.approx(0)


def test_independent_marginals_breaks_dependencies_not_marginals(frames):
    train, test = frames
    synth = IndependentMarginals().fit(train).sample(3000)
    assert marginal_tvd(test, synth) < 0.05
    assert association_error(test, synth) > 0.2
    assert tstr_auc(synth, test, "t") < 0.6
    assert detectability_auc(test, synth) > 0.7


def test_copying_is_flagged_by_privacy_tail(frames):
    train, test = frames
    copies = train.sample(3000, replace=True, random_state=1)
    p = privacy(copies, train, test, n=2000)
    assert p["exact_copy_rate"] > 0.99
    assert p["close_rate"] > 0.9


def test_light_noisy_copy_leaks_some_exact_rows(frames):
    train, test = frames
    synth = NoisyCopy(0.1).fit(train).sample(3000)
    assert privacy(synth, train, test, n=2000)["exact_copy_rate"] > 0.3  # 4 columns: 0.9^4 ≈ 66% untouched
