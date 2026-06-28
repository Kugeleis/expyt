import numpy as np
import pandas as pd
import scipy.stats as stats

from app.stats.models import (
    NormalityResult,
    SphericityResult,
    VarianceHomogeneityResult,
)


def guess_outcome_type(series: pd.Series) -> str:
    """Guess the outcome column's statistical variable type."""
    clean_series = series.dropna()
    n_unique = clean_series.nunique()

    if n_unique == 0:
        return "categorical_nominal"

    is_numeric = pd.api.types.is_numeric_dtype(series)

    if is_numeric and n_unique > 10:
        return "continuous"

    if is_numeric:
        try:
            is_integer = bool(np.all(clean_series == clean_series.astype(int)))
        except (ValueError, TypeError, OverflowError):
            is_integer = False

        if is_integer:
            val_min = int(clean_series.min())
            val_max = int(clean_series.max())
            if val_max - val_min + 1 == n_unique:
                return "categorical_ordinal_unclear"

    return "categorical_nominal"


def compute_normality(df: pd.DataFrame, outcome_col: str, group_col: str) -> dict[str, NormalityResult]:
    """Compute Shapiro-Wilk or D'Agostino-Pearson normality tests per group."""
    results = {}
    grouped = df.groupby(group_col)[outcome_col]
    for name, group_series in grouped:
        name_str = str(name)
        vals = group_series.dropna().values
        n = len(vals)
        if n < 3:
            results[name_str] = NormalityResult(
                test_used="None",
                p_value=None,
                n=n,
                is_normal=False,
                note="Insufficient data (n < 3)",
            )
            continue

        if n <= 5000:
            test_used = "Shapiro-Wilk"
            try:
                _, p_val = stats.shapiro(vals)
                p_val_float = float(p_val)
                is_normal = p_val_float > 0.05
                note = None
            except Exception as e:
                p_val_float = None
                is_normal = False
                note = f"Shapiro-Wilk failed: {e}"
        else:
            test_used = "D'Agostino-Pearson"
            try:
                _, p_val = stats.normaltest(vals)
                p_val_float = float(p_val)
                is_normal = p_val_float > 0.05
                note = None
            except Exception as e:
                p_val_float = None
                is_normal = False
                note = f"D'Agostino-Pearson failed: {e}"

        results[name_str] = NormalityResult(
            test_used=test_used,
            p_value=p_val_float,
            n=n,
            is_normal=is_normal,
            note=note,
        )
    return results


def compute_variance_homogeneity(
    df: pd.DataFrame, outcome_col: str, group_col: str
) -> VarianceHomogeneityResult | None:
    """Compute Levene's homogeneity of variance test robustly across all groups."""
    grouped = df.groupby(group_col)[outcome_col]
    group_arrays = [group_series.dropna().values for _, group_series in grouped]
    group_arrays = [arr for arr in group_arrays if len(arr) > 0]

    if len(group_arrays) < 2:
        return None

    if any(len(arr) < 2 for arr in group_arrays):
        return VarianceHomogeneityResult(
            test_used="Levene",
            statistic=0.0,
            p_value=0.0,
            equal_variances=False,
        )

    try:
        stat, p_val = stats.levene(*group_arrays, center="median")
        p_val_float = float(p_val)
        if np.isnan(p_val_float):
            return VarianceHomogeneityResult(
                test_used="Levene",
                statistic=0.0,
                p_value=0.0,
                equal_variances=False,
            )
        return VarianceHomogeneityResult(
            test_used="Levene",
            statistic=float(stat),
            p_value=p_val_float,
            equal_variances=p_val_float > 0.05,
        )
    except Exception:
        return VarianceHomogeneityResult(
            test_used="Levene",
            statistic=0.0,
            p_value=0.0,
            equal_variances=False,
        )


def compute_expected_cell_counts(
    df: pd.DataFrame, outcome_col: str, group_col: str
) -> tuple[list[list[float]] | None, float | None]:
    """Compute expected cell counts contingent matrix and minimum expected count."""
    clean_df = df[[group_col, outcome_col]].dropna()
    if clean_df.empty:
        return None, None

    observed = pd.crosstab(clean_df[group_col], clean_df[outcome_col]).values
    if observed.size == 0 or observed.sum() == 0:
        return None, None

    try:
        _, _, _, expected = stats.chi2_contingency(observed)
        expected_list = expected.tolist()
        min_val = float(np.min(expected))
        return expected_list, min_val
    except Exception:
        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        n = observed.sum()
        if n == 0:
            return None, None
        expected = (row_totals @ col_totals) / n
        expected_list = expected.tolist()
        min_val = float(np.min(expected))
        return expected_list, min_val


def compute_sphericity(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    repeated_measures: bool,
    n_conditions: int | None,
) -> SphericityResult | None:
    """Compute Mauchly's sphericity test for repeated measures."""
    if not repeated_measures:
        return None
    if n_conditions is None or n_conditions < 3:
        return None

    clean_df = df[[group_col, outcome_col]].dropna()
    grouped = clean_df.groupby(group_col)[outcome_col]
    unique_groups = list(grouped.groups.keys())
    p = len(unique_groups)

    if p < 3:
        return SphericityResult(
            statistic=1.0,
            p_value=1.0,
            sphericity_assumed=True,
            note="Sphericity mathematically guaranteed for < 3 conditions.",
        )

    groups_data = {str(name): group_series.values for name, group_series in grouped}
    lengths = [len(arr) for arr in groups_data.values()]
    n = min(lengths) if lengths else 0

    if n <= p:
        return SphericityResult(
            statistic=0.0,
            p_value=0.0,
            sphericity_assumed=False,
            note=f"Insufficient subjects (n={n}) relative to conditions (p={p}) to perform Mauchly's test.",
        )

    X = np.column_stack([np.asarray(arr[:n], dtype=float) for arr in groups_data.values()])
    sigma = np.cov(X, rowvar=False)

    d = p - 1
    A = np.zeros((p, p))
    A[:, 0] = 1.0
    A[:, 1:] = np.eye(p)[:, :-1]
    try:
        _, R = np.linalg.qr(A)
        Q, _ = np.linalg.qr(A)
        M = Q[:, 1:]
    except Exception as e:
        return SphericityResult(
            statistic=0.0,
            p_value=0.0,
            sphericity_assumed=False,
            note=f"Error constructing contrast matrix: {e}",
        )

    try:
        S = M.T @ sigma @ M
        det_S = np.linalg.det(S)
        tr_S = np.trace(S)
    except Exception as e:
        return SphericityResult(
            statistic=0.0,
            p_value=0.0,
            sphericity_assumed=False,
            note=f"Error computing transformed covariance matrix: {e}",
        )

    if tr_S <= 0:
        return SphericityResult(
            statistic=0.0,
            p_value=0.0,
            sphericity_assumed=False,
            note="Zero trace of transformed covariance matrix.",
        )

    W = det_S / ((tr_S / d) ** d)
    W = float(np.clip(W, 1e-15, 1.0))

    df_chi2 = int(p * (p - 1) / 2 - 1)
    if df_chi2 <= 0:
        df_chi2 = 1

    correction = 1.0 - (2.0 * (d**2) + d + 2.0) / (6.0 * d * (n - 1))
    chi2_val = -(n - 1) * correction * np.log(W)
    chi2_val = float(max(0.0, chi2_val))

    p_val = float(stats.chi2.sf(chi2_val, df_chi2))

    return SphericityResult(
        statistic=W,
        p_value=p_val,
        sphericity_assumed=p_val > 0.05,
        note=None,
    )
