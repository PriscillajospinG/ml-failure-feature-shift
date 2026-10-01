"""Dataset loaders and natural domain splits.

Every loader returns a dict with
    X        : pd.DataFrame of numeric features (one-hot encoded where needed)
    y        : np.ndarray of 0/1 labels
    domains  : dict  attribute_name -> pd.Series of domain ids (same index as X)
    drop     : dict  attribute_name -> list of X columns to remove when splitting
                     on that attribute (so the domain id itself is not a feature)
    kind     : dict  attribute_name -> "group" (all ordered pairs) or "time"
                     (only past -> future pairs)

All raw files are public mirrors fetched by scripts/download_data.sh.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _path(name: str) -> str:
    return os.path.join(RAW, name)


def _onehot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, dtype=float)


def load_adult():
    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital",
            "occupation", "relationship", "race", "sex", "capital_gain",
            "capital_loss", "hours", "country", "income"]
    df = pd.read_csv(_path("adult.csv"), header=None, names=cols, skipinitialspace=True)
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)
    y = (df.pop("income").str.strip().str.startswith(">50K")).astype(int).values
    df = df.drop(columns=["fnlwgt", "education"])  # education duplicated by education_num
    age_grp = pd.cut(df["age"], [0, 30, 45, 200], labels=["age<30", "age30-45", "age>45"]).astype(str)
    edu_grp = np.where(df["education_num"] >= 13, "degree", "no-degree")
    domains = {
        "sex": df["sex"].copy(),
        "race": df["race"].where(df["race"].isin(["White", "Black", "Asian-Pac-Islander"]), "Other"),
        "age": age_grp,
        "education": pd.Series(edu_grp),
        "workclass": df["workclass"].where(df["workclass"].isin(["Private", "Self-emp-not-inc", "Local-gov"]), "Other"),
    }
    cat = ["workclass", "marital", "occupation", "relationship", "race", "sex", "country"]
    df["country"] = np.where(df["country"] == "United-States", "US", "non-US")
    X = _onehot(df, cat)
    drop = {
        "sex": [c for c in X.columns if c.startswith("sex_")],
        "race": [c for c in X.columns if c.startswith("race_")],
        "age": ["age"],
        "education": ["education_num"],
        "workclass": [c for c in X.columns if c.startswith("workclass_")],
    }
    kind = {k: "group" for k in domains}
    return dict(X=X, y=y, domains=domains, drop=drop, kind=kind)


def load_compas():
    df = pd.read_csv(_path("compas.csv"))
    df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30)
            & (df.is_recid != -1) & (df.c_charge_degree != "O")].reset_index(drop=True)
    y = df["two_year_recid"].astype(int).values
    feats = df[["age", "sex", "race", "juv_fel_count", "juv_misd_count", "juv_other_count",
                "priors_count", "c_charge_degree"]].copy()
    feats["race"] = feats["race"].where(feats["race"].isin(["African-American", "Caucasian", "Hispanic"]), "Other")
    domains = {
        "race": feats["race"].copy(),
        "sex": feats["sex"].copy(),
        "age": df["age_cat"].copy(),
    }
    X = _onehot(feats, ["sex", "race", "c_charge_degree"])
    drop = {
        "race": [c for c in X.columns if c.startswith("race_")],
        "sex": [c for c in X.columns if c.startswith("sex_")],
        "age": ["age"],
    }
    return dict(X=X, y=y, domains=domains, drop=drop, kind={k: "group" for k in domains})


def load_housing():
    df = pd.read_csv(_path("housing.csv")).dropna().reset_index(drop=True)
    df = df[df.ocean_proximity != "ISLAND"].reset_index(drop=True)
    y = (df.pop("median_house_value") > 179700).astype(int).values  # global median
    df["rooms_per_hh"] = df.total_rooms / df.households
    df["beds_per_room"] = df.total_bedrooms / df.total_rooms
    df["pop_per_hh"] = df.population / df.households
    region = np.where(df.latitude >= 36.0, "north", "south")
    domains = {"proximity": df["ocean_proximity"].copy(), "region": pd.Series(region)}
    X = _onehot(df, ["ocean_proximity"])
    drop = {
        "proximity": [c for c in X.columns if c.startswith("ocean_proximity_")],
        "region": ["latitude", "longitude"],
    }
    return dict(X=X, y=y, domains=domains, drop=drop, kind={"proximity": "group", "region": "group"})


def load_bank():
    df = pd.read_csv(_path("bank.csv"), sep=";")
    y = (df.pop("y") == "yes").astype(int).values
    df = df.drop(columns=["duration"])  # known target leak
    # rows are chronological (May 2008 -> Nov 2010); split into 6 equal time blocks
    block = pd.Series(np.repeat(np.arange(6), int(np.ceil(len(df) / 6)))[: len(df)])
    cat = ["job", "marital", "education", "default", "housing", "loan", "contact",
           "month", "day_of_week", "poutcome"]
    X = _onehot(df, cat)
    # calendar month is a deterministic function of the time block; drop it for the time split
    drop = {"time": [c for c in X.columns if c.startswith("month_")]}
    return dict(X=X, y=y, domains={"time": block}, drop=drop, kind={"time": "time"})


def load_elec():
    df = pd.read_csv(_path("elec.csv"))
    y = df.pop("class").astype(int).values
    block = pd.Series(np.repeat(np.arange(8), int(np.ceil(len(df) / 8)))[: len(df)])
    return dict(X=df, y=y, domains={"time": block}, drop={"time": []}, kind={"time": "time"})


def load_airlines(top_k: int = 8):
    df = pd.read_csv(_path("airlines.csv"))
    y = df.pop("Delay").astype(int).values
    df = df.drop(columns=["Flight"])
    top = df["Airline"].value_counts().index[:top_k]
    keep = df["Airline"].isin(top).values
    df, y = df[keep].reset_index(drop=True), y[keep]
    dom = "carrier_" + df["Airline"].astype(str)
    X = df.drop(columns=["Airline"])
    return dict(X=X, y=y, domains={"carrier": dom}, drop={"carrier": []}, kind={"carrier": "group"})


def load_covtype():
    df = pd.read_csv(_path("covtype.csv"))
    y = (df.pop("Cover_Type") == 2).astype(int).values  # lodgepole pine vs rest
    wcols = [f"Wilderness_Area{i}" for i in range(1, 5)]
    area = pd.Series("area" + (df[wcols].values.argmax(1) + 1).astype(str))
    return dict(X=df, y=y, domains={"wilderness": area}, drop={"wilderness": wcols},
                kind={"wilderness": "group"})


LOADERS = {
    "adult": load_adult,
    "compas": load_compas,
    "housing": load_housing,
    "bank": load_bank,
    "elec": load_elec,
    "airlines": load_airlines,
    "covtype": load_covtype,
}


def domain_pairs(ds: dict, min_size: int = 400):
    """Yield (attribute, source_value, target_value) for every admissible pair."""
    for attr, dom in ds["domains"].items():
        dom = pd.Series(np.asarray(dom))
        counts = dom.value_counts()
        vals = [v for v in counts.index if counts[v] >= min_size]
        if ds["kind"][attr] == "time":
            vals = sorted(vals)
            for i, s in enumerate(vals):
                for t in vals[i + 1:]:
                    yield attr, s, t
        else:
            for s in vals:
                for t in vals:
                    if s != t:
                        yield attr, s, t
