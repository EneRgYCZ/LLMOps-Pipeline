"""Tests for the regime-split and metric helpers in rq4_experiment.py.

All cases use synthetic DataFrames, not results/rq4/data/rq4_vram_raw.csv, so
they do not depend on hardware measurements and stay stable across reruns of
the real experiment.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rq4_experiment import (
    ModelSpec,
    build_summary,
    compute_overall_metrics,
    compute_regime_split,
)

CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 114688]


def _raw_df(measured_by_n: dict[int, float], model_spec: ModelSpec) -> pd.DataFrame:
    """Builds a raw-CSV-shaped DataFrame with 3 repeats per context length."""
    rows = []
    for n, measured in measured_by_n.items():
        for _ in range(3):
            rows.append(
                {
                    "context_length": n,
                    "measured_vram_gb": measured,
                    "predicted_vram_gb": model_spec.predicted_vram_gb(n),
                }
            )
    return pd.DataFrame(rows)


def test_predicted_vram_gb_matches_hand_calculation():
    # weights=4.48, overhead=1.19, kv=512*2*34*(4096/4)*2*1e-9=0.0713 -> ~5.7413
    spec = ModelSpec()
    assert spec.predicted_vram_gb(512) == pytest.approx(5.7413, abs=1e-3)


def test_perfect_fit_gives_zero_error_full_r_squared_and_zero_regime_error():
    spec = ModelSpec()
    predicted = {n: spec.predicted_vram_gb(n) for n in CONTEXT_LENGTHS}
    df = _raw_df(predicted, spec)
    summary = build_summary(df)
    metrics = compute_overall_metrics(summary)

    assert metrics["mae_gb"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["mape_pct"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["r_squared"] == pytest.approx(1.0, abs=1e-9)

    regime = compute_regime_split(summary)
    assert regime["overestimate"]["mae_gb"] == pytest.approx(0.0, abs=1e-9)
    assert regime["overestimate"]["mape_pct"] == pytest.approx(0.0, abs=1e-9)
    assert regime["accurate"]["mae_gb"] == pytest.approx(0.0, abs=1e-9)
    assert regime["accurate"]["mape_pct"] == pytest.approx(0.0, abs=1e-9)


def test_constant_offset_gives_mae_equal_to_offset():
    """measured = predicted + 0.5 GB for every N.

    MAE is exactly the offset regardless of N, since abs_error_gb is
    constant. R^2 is NOT 1.0 here: compute_overall_metrics measures fit
    against the fixed formula (ss_res = sum(offset^2)), not against a
    regression refit to the shifted data, so ss_tot is the variance of the
    *predicted* values around their own mean (since a constant shift doesn't
    change variance) and a nonzero constant offset necessarily costs some
    R^2. With this sweep's predicted values (5.74-21.64 GB) and a 0.5 GB
    offset, ss_res=9*0.5^2=2.25 against ss_tot~=236.55, giving R^2~=0.9905 -
    close to 1 because the offset is small relative to the spread, but not
    exactly 1. This is expected behavior of the formula, not a bug.
    """
    spec = ModelSpec()
    offset = 0.5
    measured = {n: spec.predicted_vram_gb(n) + offset for n in CONTEXT_LENGTHS}
    df = _raw_df(measured, spec)
    summary = build_summary(df)
    metrics = compute_overall_metrics(summary)

    assert metrics["mae_gb"] == pytest.approx(offset, abs=1e-9)
    assert metrics["r_squared"] == pytest.approx(0.9904883769692756, abs=1e-9)
    assert metrics["r_squared"] < 1.0


def test_regime_split_breakpoint_matches_observed_thesis_data():
    """Real pct_error pattern from results/rq4/csvs/rq4_summary.csv.

    Values (16.0, 14.5, 11.8, 10.9, 9.4, 7.1, 4.0, 0.6, -1.7) cross the 2%
    "accurate" threshold between N=32768 (4.0%, still above) and N=65536
    (0.6%, below) -- that is where the thesis places the regime change.
    """
    spec = ModelSpec()
    pct_errors = [16.0, 14.5, 11.8, 10.9, 9.4, 7.1, 4.0, 0.6, -1.7]
    predicted = [spec.predicted_vram_gb(n) for n in CONTEXT_LENGTHS]
    measured = [p * (1 + pct / 100.0) for p, pct in zip(predicted, pct_errors)]
    df = _raw_df(dict(zip(CONTEXT_LENGTHS, measured)), spec)
    summary = build_summary(df)

    regime = compute_regime_split(summary)

    assert 32768 < regime["breakpoint_n"] < 65536
    assert regime["overestimate"]["n_points"] == 7
    assert regime["accurate"]["n_points"] == 2
