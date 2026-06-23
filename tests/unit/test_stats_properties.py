"""Unit tests for the statistical data properties computation and helpers."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.stats.base import (
    DataProperties,
    compute_data_properties,
    compute_data_properties_for_columns,
    compute_expected_cell_counts,
    compute_missing_summary,
    compute_outliers,
    compute_sphericity,
    compute_variance_homogeneity,
    guess_outcome_type,
)


@pytest.fixture
def normal_groups_df() -> pd.DataFrame:
    """Return a DataFrame with two normally distributed groups."""
    np.random.seed(42)
    g1 = np.random.normal(loc=10.0, scale=1.0, size=20)
    g2 = np.random.normal(loc=11.0, scale=1.0, size=20)
    df = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "value": np.concatenate([g1, g2]),
        }
    )
    return df


def test_compute_data_properties_valid(normal_groups_df: pd.DataFrame) -> None:
    """Test compute_data_properties with valid data."""
    props = compute_data_properties(normal_groups_df, "value", "group")

    assert props.outcome_type_guess == "continuous"
    assert props.n_groups == 2
    assert props.group_sizes == {"A": 20, "B": 20}
    # For normally distributed data, shapiro p should be > 0.05
    assert props.normality["A"].p_value is not None
    assert props.normality["A"].p_value > 0.05
    assert props.normality["B"].p_value is not None
    assert props.normality["B"].p_value > 0.05
    assert props.normality["A"].is_normal is True
    assert props.normality["B"].is_normal is True
    # Variance homogeneity should be > 0.05
    assert props.variance_homogeneity is not None
    assert props.variance_homogeneity.p_value > 0.05
    assert props.variance_homogeneity.equal_variances is True
    assert props.all_groups_normal is True
    assert props.sampled is False


def test_compute_data_properties_invalid_inputs(normal_groups_df: pd.DataFrame) -> None:
    """Test compute_data_properties with invalid columns or empty dataset."""
    with pytest.raises(ValueError, match="Group column"):
        compute_data_properties(normal_groups_df, "value", "missing_col")

    with pytest.raises(ValueError, match="Value column"):
        compute_data_properties(normal_groups_df, "missing_col", "group")

    # Empty DataFrame should raise ValueError
    df_empty = pd.DataFrame(columns=["group", "value"])
    with pytest.raises(ValueError, match="DataFrame is empty"):
        compute_data_properties(df_empty, "value", "group")


def test_compute_data_properties_small_groups() -> None:
    """Test compute_data_properties behavior on small groups (< 3 samples)."""
    # Construct a dataset with > 10 unique values total, but group A has < 3 samples
    df = pd.DataFrame(
        {
            "group": ["A", "A"] + ["B"] * 11,
            "value": [1.0, 2.0] + [float(i) for i in range(11)],
        }
    )
    props = compute_data_properties(df, "value", "group")
    # Group A has n < 3, so its normality test is skipped and is_normal is False
    assert props.normality["A"].is_normal is False
    assert props.normality["A"].p_value is None
    assert props.normality["A"].n == 2


def test_guess_outcome_type() -> None:
    """Test the outcome type guessing logic."""
    # Continuous numeric
    s_cont = pd.Series(range(20))
    assert guess_outcome_type(s_cont) == "continuous"

    # Categorical nominal (<= 10 values, non-integer)
    s_nom_str = pd.Series(["A", "B", "C"] * 3)
    assert guess_outcome_type(s_nom_str) == "categorical_nominal"

    # Categorical ordinal unclear (Likert scale 1-5)
    s_likert = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1])
    assert guess_outcome_type(s_likert) == "categorical_ordinal_unclear"

    # Categorical nominal due to gaps (not contiguous)
    s_not_contig = pd.Series([1, 3, 5, 1, 3, 5])
    assert guess_outcome_type(s_not_contig) == "categorical_nominal"


def test_expected_cell_counts() -> None:
    """Test expected cell counts calculation for categorical outcomes."""
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "B"],
            "outcome": ["X", "Y", "X", "Y", "Y"],
        }
    )
    expected_list, min_expected = compute_expected_cell_counts(df, "outcome", "group")
    assert expected_list is not None
    assert min_expected is not None
    # Row totals: A=2, B=3. Col totals: X=2, Y=3. Total N=5.
    # Expected A, X = 2 * 2 / 5 = 0.8
    assert min_expected == pytest.approx(0.8)


def test_compute_sphericity() -> None:
    """Test Mauchly's sphericity test."""
    # 1. Not repeated measures
    res_none = compute_sphericity(pd.DataFrame(), "val", "group", repeated_measures=False, n_conditions=None)
    assert res_none is None

    # 2. Repeated measures but < 3 conditions
    df_two_conds = pd.DataFrame({"group": ["T1", "T1", "T2", "T2"], "val": [10.0, 11.0, 12.0, 13.0]})
    res_two = compute_sphericity(df_two_conds, "val", "group", repeated_measures=True, n_conditions=2)
    assert res_two is None

    # 3. Repeated measures with 3 conditions, sphericity met
    np.random.seed(42)
    s1 = np.random.normal(loc=10.0, scale=1.0, size=15)
    s2 = np.random.normal(loc=10.0, scale=1.0, size=15)
    s3 = np.random.normal(loc=10.0, scale=1.0, size=15)
    df_spher = pd.DataFrame({"group": ["T1"] * 15 + ["T2"] * 15 + ["T3"] * 15, "val": np.concatenate([s1, s2, s3])})
    res = compute_sphericity(df_spher, "val", "group", repeated_measures=True, n_conditions=3)
    assert res is not None
    assert res.statistic > 0.0
    assert res.p_value > 0.05
    assert res.sphericity_assumed is True


def test_compute_missing_summary() -> None:
    """Test missing data summary and association check."""
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", None],
            "outcome": [1.0, None, 3.0, 4.0, 5.0],
        }
    )
    summary = compute_missing_summary(df, "outcome", "group")
    assert summary.outcome_missing.count == 1
    assert summary.outcome_missing.percentage == pytest.approx(20.0)
    assert summary.group_missing.count == 1
    assert summary.group_missing.percentage == pytest.approx(20.0)


def test_compute_outliers() -> None:
    """Test outlier detection using IQR method."""
    df = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "outcome": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 10.0],
        }
    )
    outliers = compute_outliers(df, "outcome", "group")
    assert outliers["A"].count == 0
    assert outliers["B"].count == 1
    assert outliers["B"].indices == [11]


def test_data_properties_sampling() -> None:
    """Test that data properties module samples large datasets (> 50,000 rows)."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "group": ["A"] * 25000 + ["B"] * 25001,
            "value": np.random.normal(loc=10.0, scale=1.0, size=50001),
        }
    )
    props = compute_data_properties(df, "value", "group")
    assert props.sampled is True
    total_n_normality = sum(res.n for res in props.normality.values())
    assert total_n_normality == 50000


def test_stats_edge_cases_and_fallbacks() -> None:
    """Test all statistical utility edge cases and fallback paths for coverage."""
    # 1. compute_variance_homogeneity with group size < 2
    df_small_var = pd.DataFrame({"group": ["A", "B"], "value": [1.0, 2.0]})
    res_var = compute_variance_homogeneity(df_small_var, "value", "group")
    assert res_var is not None
    assert res_var.equal_variances is False

    # 2. compute_expected_cell_counts with empty data
    empty_df = pd.DataFrame(columns=["outcome", "group"])
    expected, min_expected = compute_expected_cell_counts(empty_df, "outcome", "group")
    assert expected is None
    assert min_expected is None

    # 3. compute_expected_cell_counts fallback manual calculations
    zero_df_con = pd.DataFrame({"group": ["A", "A"], "outcome": ["X", "X"]})
    expected, min_expected = compute_expected_cell_counts(zero_df_con, "outcome", "group")
    assert expected is not None

    # 4. compute_sphericity insufficient subjects (n <= p)
    df_spher_insufficient = pd.DataFrame({"group": ["T1", "T2", "T3"], "value": [1.0, 2.0, 3.0]})
    res_spher = compute_sphericity(
        df_spher_insufficient,
        "value",
        "group",
        repeated_measures=True,
        n_conditions=3,
    )
    assert res_spher is not None
    assert "Insufficient subjects" in res_spher.note

    # 5. compute_missing_summary with empty df
    empty_df2 = pd.DataFrame({"group": [], "value": []})
    summary = compute_missing_summary(empty_df2, "value", "group")
    assert summary.outcome_missing.count == 0
    assert summary.group_missing.count == 0
    assert summary.association.significant is None

    # 6. compute_outliers with group having < 4 elements
    df_outliers_small = pd.DataFrame({"group": ["A"] * 3, "value": [1.0, 2.0, 3.0]})
    outliers = compute_outliers(df_outliers_small, "value", "group")
    assert outliers["A"].count == 0

    # 7. compute_data_properties with a categorical outcome
    df_categorical = pd.DataFrame(
        {"group": ["A"] * 5 + ["B"] * 5, "value": ["X"] * 3 + ["Y"] * 2 + ["X"] * 2 + ["Y"] * 3}
    )
    props_cat = compute_data_properties(df_categorical, "value", "group")
    assert props_cat.outcome_type_guess == "categorical_nominal"
    assert props_cat.normality == {}
    assert props_cat.variance_homogeneity is None
    assert props_cat.expected_cell_counts is not None
    assert props_cat.min_expected_cell_count is not None

    # 8. compute_data_properties with small sample size warning
    df_small_sample = pd.DataFrame({"group": ["A"] * 4 + ["B"] * 10, "value": [float(i) for i in range(14)]})
    props_warn = compute_data_properties(df_small_sample, "value", "group")
    assert props_warn.sample_size_warning is not None
    assert "small sample sizes" in props_warn.sample_size_warning

    # 9. compute_data_properties_for_columns
    props_map = compute_data_properties_for_columns(df_small_sample, "group", ["value"])
    assert "value" in props_map

    # 10. DataProperties old format converter fallback default value for missing
    props_old = DataProperties.model_validate(
        {
            "n_groups": 2,
            "group_sizes": {"A": 10, "B": 10},
            "normality": {"A": 0.5, "B": 0.6},
            "variance_homogeneity": 0.8,
        }
    )
    assert props_old.outcome_type_guess == "continuous"
    assert props_old.all_groups_normal is True
    assert props_old.missing.outcome_missing.count == 0


def test_stats_properties_extra_edge_cases() -> None:
    """Test further statistical edge cases to maximize code coverage."""
    # 1. convert_old_format with non-dict input
    assert DataProperties.convert_old_format([1, 2, 3]) == [1, 2, 3]

    # 2. _convert_old_normality with non-dict normality
    from app.stats.base import _convert_old_normality

    data = {"normality": "not-a-dict"}
    _convert_old_normality(data)
    assert data["normality"] == "not-a-dict"

    # 3. _get_all_groups_normal fallback cases
    from app.stats.base import _get_all_groups_normal

    assert _get_all_groups_normal("not-a-dict") is False
    assert _get_all_groups_normal({"A": {"is_normal": False}}) is False

    # Custom class for testing hasattr(v, "is_normal")
    class DummyNormalObj:
        def __init__(self, is_normal: bool):
            self.is_normal = is_normal

    assert _get_all_groups_normal({"A": DummyNormalObj(True)}) is True
    assert _get_all_groups_normal({"A": DummyNormalObj(False)}) is False
    assert _get_all_groups_normal({"A": "invalid-value"}) is False

    # 4. guess_outcome_type empty series and exceptions
    empty_series = pd.Series([], dtype=float)
    assert guess_outcome_type(empty_series) == "categorical_nominal"

    # Exception in is_integer guess_outcome_type
    with patch("pandas.api.types.is_numeric_dtype", return_value=True):
        bad_series = pd.Series(["abc", "def"])
        assert guess_outcome_type(bad_series) == "categorical_nominal"

    # 5. compute_normality Shapiro-Wilk and D'Agostino-Pearson failures
    with patch("scipy.stats.shapiro", side_effect=ValueError("Mock shapiro error")):
        df_shapiro = pd.DataFrame({"group": ["A"] * 5, "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        from app.stats.base import compute_normality

        res_sh = compute_normality(df_shapiro, "value", "group")
        assert "Shapiro-Wilk failed" in (res_sh["A"].note or "")
        assert res_sh["A"].p_value is None

    with patch("scipy.stats.normaltest", side_effect=ValueError("Mock normaltest error")):
        df_dp = pd.DataFrame({"group": ["A"] * 5001, "value": [1.0] * 5001})
        res_dp = compute_normality(df_dp, "value", "group")
        assert "D'Agostino-Pearson failed" in (res_dp["A"].note or "")
        assert res_dp["A"].p_value is None

    # 6. stats.levene returning NaN p-value
    with patch("scipy.stats.levene", return_value=(0.0, np.nan)):
        df_levene_nan = pd.DataFrame({"group": ["A"] * 5 + ["B"] * 5, "value": [1.0] * 10})
        res_lev_nan = compute_variance_homogeneity(df_levene_nan, "value", "group")
        assert res_lev_nan is not None
        assert res_lev_nan.equal_variances is False

    # stats.levene raising exception
    with patch("scipy.stats.levene", side_effect=RuntimeError("Mock Levene Error")):
        df_levene_err = pd.DataFrame({"group": ["A"] * 5 + ["B"] * 5, "value": [1.0] * 10})
        res_lev_err = compute_variance_homogeneity(df_levene_err, "value", "group")
        assert res_lev_err is not None
        assert res_lev_err.equal_variances is False

    # 7. chi2_contingency raising exception in compute_expected_cell_counts
    with patch("scipy.stats.chi2_contingency", side_effect=ValueError("Mock chi2 error")):
        df_chi = pd.DataFrame({"group": ["A"] * 5 + ["B"] * 5, "outcome": ["X"] * 10})
        expected, min_val = compute_expected_cell_counts(df_chi, "outcome", "group")
        assert expected is not None

    # 8. sphericity linalg exceptions and trace <= 0
    with patch("numpy.linalg.qr", side_effect=Exception("Mock QR Error")):
        df_spher_err = pd.DataFrame({"group": ["T1"] * 5 + ["T2"] * 5 + ["T3"] * 5, "value": [1.0] * 15})
        res_spher_err = compute_sphericity(df_spher_err, "value", "group", repeated_measures=True, n_conditions=3)
        assert res_spher_err is not None
        assert "Error constructing contrast matrix" in res_spher_err.note

    with patch("numpy.linalg.det", side_effect=Exception("Mock Det Error")):
        df_spher_err2 = pd.DataFrame({"group": ["T1"] * 5 + ["T2"] * 5 + ["T3"] * 5, "value": [1.0] * 15})
        res_spher_err2 = compute_sphericity(df_spher_err2, "value", "group", repeated_measures=True, n_conditions=3)
        assert res_spher_err2 is not None
        assert "Error computing transformed covariance matrix" in res_spher_err2.note

    with patch("numpy.trace", return_value=0.0):
        df_spher_trace = pd.DataFrame({"group": ["T1"] * 5 + ["T2"] * 5 + ["T3"] * 5, "value": [1.0] * 15})
        res_spher_trace = compute_sphericity(df_spher_trace, "value", "group", repeated_measures=True, n_conditions=3)
        assert res_spher_trace is not None
        assert "Zero trace" in res_spher_trace.note

    # 9. compute_missing_summary chi2_contingency failure
    with patch("scipy.stats.chi2_contingency", side_effect=ValueError("Mock contingency error")):
        df_miss_err = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "outcome": [1.0, None, 3.0, None],
            }
        )
        summary_err = compute_missing_summary(df_miss_err, "outcome", "group")
        assert "Association test failed" in (summary_err.association.note or "")
