"""Exploratory statistics for field-level comparisons.

Every result is descriptive: n = 10 fields per grower and 5 field-years, so
p-values are reported next to effect sizes, never as a decision rule.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261120
N_PERM = 10_000
N_BOOT = 2_000


def kruskal_epsilon(df: pd.DataFrame, value: str, group: str) -> dict:
    """Kruskal-Wallis H with epsilon-squared effect size (descriptive)."""
    clean = df[[value, group]].dropna()
    groups = [g[value].to_numpy() for _, g in clean.groupby(group, sort=True)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {"test": "kruskal_wallis", "n": int(len(clean)), "error": "fewer than two groups"}
    h, p = stats.kruskal(*groups)
    n = int(len(clean))
    k = len(groups)
    eps2 = float(max(0.0, (h - k + 1) / (n - k)))
    return {
        "test": "kruskal_wallis",
        "value": value,
        "group": group,
        "groups": {str(name): int(len(g)) for name, g in clean.groupby(group, sort=True)},
        "n": n,
        "H": float(h),
        "p": float(p),
        "epsilon_squared": eps2,
        "group_medians": {
            str(name): float(g[value].median())
            for name, g in clean.groupby(group, sort=True)
        },
    }


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def pairwise_mannwhitney(df: pd.DataFrame, value: str, group: str) -> list[dict]:
    clean = df[[value, group]].dropna()
    names = sorted(clean[group].astype(str).unique())
    rows: list[dict] = []
    for a, b in combinations(names, 2):
        xa = clean.loc[clean[group].astype(str) == a, value].to_numpy()
        xb = clean.loc[clean[group].astype(str) == b, value].to_numpy()
        if not len(xa) or not len(xb):
            continue
        u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        # rank-biserial correlation, sign = direction (a vs b)
        rb = float(1.0 - 2.0 * u / (len(xa) * len(xb)))
        rows.append(
            {
                "value": value,
                "group_a": a,
                "group_b": b,
                "n_a": int(len(xa)),
                "n_b": int(len(xb)),
                "median_a": float(np.median(xa)),
                "median_b": float(np.median(xb)),
                "U": float(u),
                "p": float(p),
                "rank_biserial": rb,
            }
        )
    if rows:
        adj = holm([r["p"] for r in rows])
        for row, p_adj in zip(rows, adj):
            row["p_holm"] = float(p_adj)
    return rows


def _spearman_once(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def spearman_perm(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    n_perm: int = N_PERM,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Spearman rho with permutation p-value and bootstrap 95 % CI."""
    clean = df[[x, y]].dropna()
    xs = clean[x].to_numpy(dtype=float)
    ys = clean[y].to_numpy(dtype=float)
    n = len(xs)
    if n < 5:
        return {"test": "spearman", "x": x, "y": y, "n": n, "error": "n < 5"}
    rho = _spearman_once(xs, ys)
    rng = np.random.default_rng(seed)
    if np.isnan(rho):
        return {"test": "spearman", "x": x, "y": y, "n": n, "error": "constant input"}
    extreme = 0
    for _ in range(n_perm):
        if abs(_spearman_once(xs, rng.permutation(ys))) >= abs(rho) - 1e-12:
            extreme += 1
    p_perm = (extreme + 1) / (n_perm + 1)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        val = _spearman_once(xs[idx], ys[idx])
        if not np.isnan(val):
            boots.append(val)
    lo, hi = (
        (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        if boots
        else (np.nan, np.nan)
    )
    return {
        "test": "spearman",
        "x": x,
        "y": y,
        "n": n,
        "rho": rho,
        "p_perm": float(p_perm),
        "ci95": [lo, hi],
        "n_perm": n_perm,
        "n_boot": n_boot,
        "seed": seed,
    }


def _r2(y: np.ndarray, X: np.ndarray | None) -> float:
    if X is None or X.shape[1] == 0:
        return 0.0
    # Intercept + reference-dropped dummies: without the intercept the dropped
    # level is forced to zero and R^2 goes wildly negative.
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _dummies(values: pd.Series) -> np.ndarray:
    cats = sorted(values.astype(str).unique())
    return np.column_stack([(values.astype(str) == c).to_numpy(dtype=float) for c in cats[1:]])


def variance_decomposition(df: pd.DataFrame, value: str) -> dict:
    """Sequential (Type I) R^2 shares for a nested design.

    Order: year -> grower -> field within grower -> residual. Year goes first
    because interannual weather is the driver we want to size up; field is
    nested in grower, so the third block is the field-within-grower contrast.
    """
    clean = df[[value, "year", "grower_slug", "field_id"]].dropna()
    y = clean[value].to_numpy(dtype=float)
    n = len(y)
    if n < 10:
        return {"test": "variance_decomposition", "value": value, "n": n, "error": "n < 10"}

    X_year = _dummies(clean["year"])
    X_grower = _dummies(clean["grower_slug"])
    X_field = _dummies(clean["field_id"])

    r2_year = _r2(y, X_year)
    r2_grower = _r2(y, np.column_stack([X_year, X_grower]))
    r2_field = _r2(y, np.column_stack([X_year, X_grower, X_field]))
    shares = {
        "year": max(0.0, r2_year),
        "grower": max(0.0, r2_grower - r2_year),
        "field_within_grower": max(0.0, r2_field - r2_grower),
        "residual": max(0.0, 1.0 - r2_field),
    }
    return {
        "test": "variance_decomposition",
        "value": value,
        "n": n,
        "order": ["year", "grower", "field_within_grower", "residual"],
        "shares": shares,
        "total_mean": float(y.mean()),
        "total_sd": float(y.std(ddof=1)),
        "cv_pct": float(100.0 * y.std(ddof=1) / y.mean()) if y.mean() else np.nan,
    }


def describe(df: pd.DataFrame, value: str, group: str) -> list[dict]:
    rows = []
    for name, g in df.groupby(group, sort=True):
        v = g[value].dropna().to_numpy(dtype=float)
        if not len(v):
            continue
        rows.append(
            {
                "group": str(name),
                "n": int(len(v)),
                "mean": float(v.mean()),
                "median": float(np.median(v)),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                "min": float(v.min()),
                "max": float(v.max()),
                "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)),
            }
        )
    return rows
