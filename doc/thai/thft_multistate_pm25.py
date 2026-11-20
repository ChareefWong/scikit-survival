"""Multistate (illness-death) model of PM2.5 exposure with scikit-survival.

States: 0 = healthy, 1 = disease, 2 = death
Transitions: 1 = 0->1, 2 = 0->2, 3 = 1->2 (clock-reset / semi-Markov)

Run with your own data:   python thft_multistate_pm25.py thft.csv
Run with simulated data:  python thft_multistate_pm25.py
See README_multistate_pm25_th.md for the Thai explanation of each step.
"""

import sys

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

COVARIATES = ["pm25_10", "age", "male", "smoker"]
SEED = 42


# ---------------------------------------------------------------- 1. data
def simulate_thft(n=3000, seed=SEED):
    """Simulated data with the same columns the guide expects."""
    rng = np.random.default_rng(seed)
    pm25 = rng.normal(30, 10, n).clip(5, 80)  # µg/m³
    age = rng.normal(50, 10, n).clip(20, 85)
    male = rng.integers(0, 2, n)
    smoker = rng.binomial(1, 0.25, n)
    lp = 0.25 * (pm25 / 10) + 0.03 * (age - 50) + 0.2 * male + 0.4 * smoker
    t_ill_lat = rng.exponential(1 / (0.02 * np.exp(lp)))
    t_dth_lat = rng.exponential(1 / (0.005 * np.exp(lp)))
    t_cens = rng.uniform(5, 15, n)

    ill = (t_ill_lat < t_dth_lat) & (t_ill_lat < t_cens)
    lp3 = 0.15 * (pm25 / 10) + 0.04 * (age - 50)
    t_after = rng.exponential(1 / (0.05 * np.exp(lp3)))
    t_death_total = np.where(ill, t_ill_lat + t_after, t_dth_lat)
    death = t_death_total < t_cens
    return pd.DataFrame({
        "id": np.arange(n),
        "pm25": pm25.round(1),
        "age": age.round(0),
        "male": male,
        "smoker": smoker,
        "ill": ill.astype(int),
        "t_ill": np.where(ill, t_ill_lat, np.minimum(t_death_total, t_cens)).round(3),
        "death": death.astype(int),
        "t_death": np.minimum(t_death_total, t_cens).round(3),
    })


def prepare(df):
    df = df.copy()
    df["pm25_10"] = df["pm25"] / 10  # HR per 10 µg/m³
    # illness and death on the same day: shift illness slightly earlier
    same = (df["ill"] == 1) & (df["t_ill"] >= df["t_death"])
    df.loc[same, "t_ill"] = df.loc[same, "t_death"] - 0.5 / 365.25
    return df


# ------------------------------------------------- 2. transition datasets
def make_transitions(df):
    """Clock-reset data sets, one per transition."""
    t1 = df[COVARIATES].copy()
    t1["time"] = df["t_ill"]
    t1["event"] = df["ill"] == 1  # 0 -> 1

    t2 = df[COVARIATES].copy()
    t2["time"] = df["t_ill"]
    t2["event"] = (df["ill"] == 0) & (df["death"] == 1)  # 0 -> 2

    sick = df[df["ill"] == 1]
    t3 = sick[COVARIATES].copy()
    t3["age"] = sick["age"] + sick["t_ill"]  # age at disease onset
    t3["time"] = sick["t_death"] - sick["t_ill"]  # time since onset
    t3["event"] = sick["death"] == 1  # 1 -> 2
    return {"0->1 (healthy->disease)": t1,
            "0->2 (healthy->death)": t2,
            "1->2 (disease->death)": t3}


# ------------------------------------------------------- 3. fit models
def fit_models(trans):
    models = {}
    for name, d in trans.items():
        y = Surv.from_arrays(d["event"], d["time"])
        m = CoxPHSurvivalAnalysis(ties="efron").fit(d[COVARIATES], y)
        models[name] = m
    return models


def bootstrap_hr(trans, n_boot=200, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for name, d in trans.items():
        est = []
        for _ in range(n_boot):
            b = d.iloc[rng.integers(0, len(d), len(d))]
            y = Surv.from_arrays(b["event"], b["time"])
            m = CoxPHSurvivalAnalysis(ties="efron").fit(b[COVARIATES], y)
            est.append(m.coef_)
        est = np.exp(np.array(est))
        for j, c in enumerate(COVARIATES):
            lo, hi = np.percentile(est[:, j], [2.5, 97.5])
            rows.append({"transition": name, "covariate": c, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# -------------------------------------------- 4. state probabilities
def _draw_times(model, X, rng):
    """Draw one event time per row from the predicted survival curve."""
    out = np.full(len(X), np.inf)
    for i, fn in enumerate(model.predict_survival_function(X)):
        u = rng.uniform()
        hit = np.nonzero(fn.y <= u)[0]
        if hit.size:
            out[i] = fn.x[hit[0]]
    return out


def state_probabilities(models, profile, times, n_sim=2000, seed=SEED):
    """Monte-Carlo P(state at t) for one covariate profile."""
    rng = np.random.default_rng(seed)
    m1, m2, m3 = models.values()
    X = pd.DataFrame([profile] * n_sim)[COVARIATES]
    T1 = _draw_times(m1, X, rng)
    T2 = _draw_times(m2, X, rng)
    to_ill = T1 < T2
    X3 = X.copy()
    X3["age"] = X["age"] + np.where(np.isfinite(T1), T1, 0)
    T3 = _draw_times(m3, X3, rng)
    t_death = np.where(to_ill, T1 + T3, T2)
    res = []
    for t in times:
        dead = t_death <= t
        sick = to_ill & (T1 <= t) & ~dead
        res.append({"time": t, "P(healthy)": 1 - (dead | sick).mean(),
                    "P(disease)": sick.mean(), "P(death)": dead.mean()})
    return pd.DataFrame(res)


# ------------------------------------------------------------- main
def main():
    df = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else simulate_thft()
    df = prepare(df)
    trans = make_transitions(df)

    print("Events per transition:")
    for name, d in trans.items():
        print(f"  {name}: n={len(d)}, events={int(d['event'].sum())}")

    models = fit_models(trans)
    print("\nHazard ratios (pm25_10 = per 10 µg/m³):")
    hr = pd.DataFrame({n: np.exp(m.coef_) for n, m in models.items()}, index=COVARIATES)
    print(hr.round(3))

    print("\nBootstrap 95% CI (200 resamples):")
    print(bootstrap_hr(trans).round(3).to_string(index=False))

    print("\nHarrell's C-index per transition:")
    for name, d in trans.items():
        c = concordance_index_censored(d["event"], d["time"], models[name].predict(d[COVARIATES]))[0]
        print(f"  {name}: {c:.3f}")

    base = {"age": 50, "male": 1, "smoker": 0}
    for pm in (15, 45):
        print(f"\nState probabilities, PM2.5 = {pm} µg/m³, age 50, male, non-smoker:")
        p = state_probabilities(models, {**base, "pm25_10": pm / 10}, times=[1, 3, 5, 10])
        print(p.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
