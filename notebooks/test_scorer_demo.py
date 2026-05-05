"""Test/demo: sklearn scorers for ordinal RPS used with subject-level CV."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score, GridSearchCV

from ord_dts import OrdinalDiscreteTimeSurvival, AdaBoostBinaryRegressor
from ord_dts_scoring import (
    ranked_probability_score,
    macro_ranked_probability_score,
    per_class_ranked_probability_score,
    make_rps_scorer,
    make_macro_rps_scorer,
    rps_scorer,
    macro_rps_scorer,
)


# ---------------- data ----------------

def simulate(n_subjects=300, n_visits=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_subjects):
        is_fast = rng.random() < 0.3
        slope = 1.2 if is_fast else 0.15
        b = rng.normal(0.0, 0.4)
        x1_base = rng.normal()
        for j in range(n_visits):
            x1 = x1_base + rng.normal(0.0, 0.2)
            x2 = rng.normal()
            eta = (
                0.6 * x1 + 0.3 * x2 + 0.5 * x1 * x2 + 0.4 * x1 ** 2
                + slope * j + b - 1.5 + rng.normal(0.0, 0.4)
            )
            rows.append((sid, float(j), x1, x2, eta))
    df = pd.DataFrame(rows, columns=["subj_id", "time", "x1", "x2", "eta"])
    df["Y"] = np.searchsorted(np.array([-0.5, 0.7, 2.0]), df["eta"])
    return df.drop(columns=["eta"])


# ---------------- unit-style sanity checks ----------------

def test_rps_known_values():
    # Perfect prediction: RPS = 0
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert ranked_probability_score(y_true, y_proba) == 0.0
    print("perfect prediction: RPS = 0 ✓")

    # Worst-case: predict opposite end of ordinal scale
    y_true = np.array([0])
    y_proba = np.array([[0, 0, 1]], dtype=float)  # truth=0, predict mass on 2
    # sum_m (cum_pred - cum_obs)^2: m=0: (0-1)^2=1, m=1: (0-1)^2=1, m=2: (1-1)^2=0 → 2
    expected = 2.0
    got = ranked_probability_score(y_true, y_proba, classes=[0, 1, 2])
    assert abs(got - expected) < 1e-12, f"{got} vs {expected}"
    print(f"worst-case ordinal RPS: {got:.3f} (expected 2.0) ✓")

    # Adjacent miss is much less penalized than far miss (the ordinal point)
    y_true = np.array([0])
    near = ranked_probability_score(
        y_true, np.array([[0, 1, 0]], dtype=float), classes=[0, 1, 2]
    )
    far = ranked_probability_score(
        y_true, np.array([[0, 0, 1]], dtype=float), classes=[0, 1, 2]
    )
    assert near < far, f"near={near}, far={far}"
    print(f"adjacent miss RPS={near}, far miss RPS={far} (near < far ✓)")


def test_macro_vs_plain():
    """Macro and plain should agree on perfectly balanced y_true; differ otherwise."""
    rng = np.random.default_rng(0)

    # Balanced
    y = np.repeat([0, 1, 2, 3], 25)
    P = rng.dirichlet(np.ones(4), size=len(y))
    plain = ranked_probability_score(y, P, classes=[0, 1, 2, 3])
    macro = macro_ranked_probability_score(y, P, classes=[0, 1, 2, 3])
    print(f"balanced y: plain={plain:.3f}, macro={macro:.3f} (close to equal expected)")

    # Imbalanced
    y_imb = np.concatenate([[0] * 90, [1] * 5, [2] * 3, [3] * 2])
    P_imb = rng.dirichlet(np.ones(4), size=len(y_imb))
    plain_imb = ranked_probability_score(y_imb, P_imb, classes=[0, 1, 2, 3])
    macro_imb = macro_ranked_probability_score(y_imb, P_imb, classes=[0, 1, 2, 3])
    print(f"imbalanced y: plain={plain_imb:.3f}, macro={macro_imb:.3f} (should differ)")
    assert abs(plain_imb - macro_imb) > 0.01, "macro should differ from plain on imbalanced data"


def test_class_validation():
    # Non-numeric without classes should raise
    try:
        ranked_probability_score(
            np.array(["a", "b"]),
            np.array([[0.5, 0.5], [0.3, 0.7]]),
        )
    except ValueError as e:
        print(f"non-numeric without classes: raised ✓ ({e})")
    else:
        raise AssertionError("should have raised")

    # y_true value not in classes
    try:
        ranked_probability_score(
            np.array([0, 1, 99]),
            np.array([[0.5, 0.5], [0.3, 0.7], [0.2, 0.8]]),
            classes=[0, 1],
        )
    except ValueError as e:
        print(f"unknown y_true value: raised ✓")
    else:
        raise AssertionError("should have raised")

    # Mismatched proba columns
    try:
        ranked_probability_score(
            np.array([0, 1]),
            np.array([[0.3, 0.3, 0.4], [0.2, 0.5, 0.3]]),
            classes=[0, 1],
        )
    except ValueError:
        print("mismatched proba columns: raised ✓")
    else:
        raise AssertionError("should have raised")


# ---------------- end-to-end: cross_val_score with GroupKFold ----------------

def test_cross_val_score():
    df = simulate(n_subjects=300, n_visits=6, seed=42)
    X, y = df.drop(columns="Y"), df["Y"].to_numpy()
    groups = df["subj_id"].to_numpy()

    model = OrdinalDiscreteTimeSurvival(
        order=[0, 1, 2, 3],
        base_estimator=LogisticRegression(max_iter=1000),
        class_weight="balanced",
        calibration_method="platt",
        calibration_random_state=0,
    )

    # Subject-level CV is essential — use GroupKFold with subj_id as groups
    cv = GroupKFold(n_splits=5)

    scorer_plain = make_rps_scorer(classes=[0, 1, 2, 3])
    scorer_macro = make_macro_rps_scorer(classes=[0, 1, 2, 3])

    plain_scores = cross_val_score(
        model, X, y, scoring=scorer_plain, cv=cv, groups=groups
    )
    macro_scores = cross_val_score(
        model, X, y, scoring=scorer_macro, cv=cv, groups=groups
    )
    # greater_is_better=False ⇒ sklearn negates internally; flip back for reporting
    print(f"\n5-fold GroupKFold cross_val_score (subject-level):")
    print(f"  plain RPS : {-plain_scores.mean():.3f} ± {plain_scores.std():.3f}")
    print(f"  macro RPS : {-macro_scores.mean():.3f} ± {macro_scores.std():.3f}")

    # Pre-built default scorers also work (since y is numeric and all classes present)
    default_plain = cross_val_score(
        model, X, y, scoring=rps_scorer, cv=cv, groups=groups
    )
    print(f"  default rps_scorer matches: {np.allclose(default_plain, plain_scores)}")


# ---------------- end-to-end: GridSearchCV ----------------

def test_grid_search():
    df = simulate(n_subjects=200, n_visits=5, seed=7)
    X, y = df.drop(columns="Y"), df["Y"].to_numpy()
    groups = df["subj_id"].to_numpy()

    model = OrdinalDiscreteTimeSurvival(
        order=[0, 1, 2, 3],
        base_estimator=AdaBoostBinaryRegressor(random_state=0),
        class_weight="balanced",
        calibration_method="platt",
        calibration_random_state=0,
    )

    # Search over calibration-fraction and AdaBoost depth via base_estimator.
    # We can't grid the base_estimator's internal params easily through clone,
    # so just grid the OrdinalDiscreteTimeSurvival params here.
    grid = {
        "calibration_fraction": [0.2, 0.3],
        "calibration_method": ["platt", "prior"],
    }
    gs = GridSearchCV(
        model,
        grid,
        scoring=make_rps_scorer(classes=[0, 1, 2, 3]),
        cv=GroupKFold(n_splits=3),
        n_jobs=1,
        refit=True,
    )
    gs.fit(X, y, groups=groups)
    print(f"\nGridSearchCV best params: {gs.best_params_}")
    print(f"GridSearchCV best (negated) plain RPS: {-gs.best_score_:.3f}")


# ---------------- per-class breakdown demo ----------------

def test_per_class_diagnostic():
    df = simulate(n_subjects=300, n_visits=6, seed=42)
    rng = np.random.default_rng(0)
    train_mask = np.isin(df["subj_id"], rng.choice(df["subj_id"].unique(), 200, replace=False))
    tr, te = df[train_mask], df[~train_mask]
    X_tr, y_tr = tr.drop(columns="Y"), tr["Y"].to_numpy()
    X_te, y_te = te.drop(columns="Y"), te["Y"].to_numpy()

    model = OrdinalDiscreteTimeSurvival(
        order=[0, 1, 2, 3],
        base_estimator=AdaBoostBinaryRegressor(random_state=0),
        class_weight="balanced",
        calibration_method="platt",
        calibration_random_state=0,
    ).fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)

    plain = ranked_probability_score(y_te, proba, classes=[0, 1, 2, 3])
    macro = macro_ranked_probability_score(y_te, proba, classes=[0, 1, 2, 3])
    per_class = per_class_ranked_probability_score(y_te, proba, classes=[0, 1, 2, 3])
    counts = pd.Series(y_te).value_counts().sort_index().to_dict()

    print(f"\nPer-class RPS breakdown on held-out test:")
    print(f"  plain RPS : {plain:.3f}")
    print(f"  macro RPS : {macro:.3f}")
    for c in [0, 1, 2, 3]:
        print(f"     class {c}: RPS={per_class[c]:.3f}  (n={counts.get(c, 0)})")


if __name__ == "__main__":
    print("=" * 60); print("metric sanity checks"); print("=" * 60)
    test_rps_known_values()
    test_macro_vs_plain()
    test_class_validation()

    print("\n" + "=" * 60); print("cross_val_score with GroupKFold"); print("=" * 60)
    test_cross_val_score()

    print("\n" + "=" * 60); print("GridSearchCV"); print("=" * 60)
    test_grid_search()

    print("\n" + "=" * 60); print("per-class diagnostic"); print("=" * 60)
    test_per_class_diagnostic()

    print("\nAll checks passed.")