"""
Week 5 utilities: Titanic analysis + Plotly figs.
Functions return pandas objects or plotly.graph_objects.Figure.
PEP-8 compliant and defensive on data loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px


# -----------------------------
# Data loading
# -----------------------------
def load_titanic(candidates: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Load the Titanic dataset (Kaggle-style columns).
    Tries common relative paths; as a last resort, falls back to seaborn's
    titanic and adapts columns (note: seaborn lacks the Name field).

    Returns
    -------
    pd.DataFrame
        Columns expected (at least): Survived, Pclass, Sex, Age, SibSp,
        Parch, Fare, [Name]
    """
    if candidates is None:
        candidates = (
            "data/titanic.csv", "data/train.csv",
            "./titanic.csv", "./train.csv",
            "../data/titanic.csv", "../data/train.csv",
        )

    for p in candidates:
        path = Path(p)
        if path.is_file():
            df = pd.read_csv(path)
            return df

    # Fallback: seaborn dataset (column names differ; no Name)
    try:
        import seaborn as sns  # optional dependency in lab envs
        df = sns.load_dataset("titanic")
        # Map to Kaggle-like columns
        # seaborn has: survived, pclass, sex, age, sibsp, parch, fare
        rename_map = {
            "survived": "Survived",
            "pclass": "Pclass",
            "sex": "Sex",
            "age": "Age",
            "sibsp": "SibSp",
            "parch": "Parch",
            "fare": "Fare",
        }
        df = df.rename(columns=rename_map)
        if "Name" not in df.columns:
            df["Name"] = pd.NA  # keep API stable for last_names()
        # Ensure ints where appropriate (when possible)
        for col in ["Survived", "Pclass", "SibSp", "Parch"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        return df
    except Exception:
        raise FileNotFoundError(
            "Titanic CSV not found. Place Kaggle 'train.csv' (or titanic.csv) "
            "in ./data/ or project root."
        )


# -----------------------------
# Exercise 1 — Demographics
# -----------------------------
def survival_demographics() -> pd.DataFrame:
    """
    Build a table of survival metrics by passenger class, sex, and age group.

    Age groups (ordered, categorical):
      - Child (<=12)
      - Teen (13–19)
      - Adult (20–59)
      - Senior (>=60)

    Returns
    -------
    pd.DataFrame with columns:
      Pclass, Sex, age_group, n_passengers, n_survivors, survival_rate
      (sorted by Pclass asc, Sex asc, then age_group order)
    """
    df = load_titanic()

    # Age bins → categorical labels
    bins = [-float("inf"), 12, 19, 59, float("inf")]
    labels = pd.CategoricalDtype(
        categories=["Child", "Teen", "Adult", "Senior"], ordered=True
    )
    age_group = pd.cut(df["Age"], bins=bins, labels=labels.categories)
    df = df.assign(age_group=age_group.astype(labels))

    # Clean essential columns
    use = df.dropna(subset=["Pclass", "Sex", "Survived", "age_group"])

    grouped = (
        use.groupby(["Pclass", "Sex", "age_group"], observed=True)
        .agg(
            n_passengers=("Survived", "size"),
            n_survivors=("Survived", "sum"),
            survival_rate=("Survived", "mean"),
        )
        .reset_index()
        .sort_values(["Pclass", "Sex", "age_group"])
        .reset_index(drop=True)
    )
    # Make survival_rate nice
    grouped["survival_rate"] = grouped["survival_rate"].astype(float)
    return grouped


def visualize_demographic(table: pd.DataFrame) -> "plotly.graph_objs._figure.Figure":
    """
    Create a Plotly figure that answers a demographic survival question.
    Default: clustered bars of survival rate by age group, faceted by Pclass,
    colored by Sex (common “women and children first?” exploration).

    Parameters
    ----------
    table : pd.DataFrame
        Output of survival_demographics().

    Returns
    -------
    plotly Figure
    """
    fig = px.bar(
        table,
        x="age_group",
        y="survival_rate",
        color="Sex",
        barmode="group",
        facet_col="Pclass",
        category_orders={"age_group": ["Child", "Teen", "Adult", "Senior"]},
        labels={"age_group": "Age Group", "survival_rate": "Survival Rate"},
        title="Survival rate by age group, sex, and passenger class",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(legend_title_text="Sex")
    return fig


# -----------------------------
# Exercise 2 — Families & Wealth
# -----------------------------
def family_groups() -> pd.DataFrame:
    """
    Explore relationship between family size, class, and fare.

    Returns
    -------
    pd.DataFrame with:
      Pclass, family_size, n_passengers, avg_fare, min_fare, max_fare
      (sorted by Pclass, then family_size)
    """
    df = load_titanic()
    df = df.assign(family_size=(df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1))

    use = df.dropna(subset=["Pclass", "Fare", "family_size"])
    table = (
        use.groupby(["family_size", "Pclass"], observed=True)
        .agg(
            n_passengers=("Fare", "size"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max"),
        )
        .reset_index()
        .sort_values(["Pclass", "family_size"])
        .reset_index(drop=True)
    )
    return table


def last_names() -> pd.Series:
    """
    Extract last names from 'Name' and return counts (desc).
    If Name is missing (e.g., seaborn fallback), returns an empty Series.
    """
    df = load_titanic()
    if "Name" not in df.columns or df["Name"].isna().all():
        return pd.Series(dtype="int64")

    # Titanic names like "Braund, Mr. Owen Harris" → take part before first comma
    last = df["Name"].astype(str).str.extract(r"^\s*([^,]+)\s*,", expand=False)
    counts = last.dropna().str.strip().value_counts()
    return counts


def visualize_families(table: pd.DataFrame) -> "plotly.graph_objs._figure.Figure":
    """
    Plot avg fare vs family_size, line per Pclass (shows wealth patterns).
    """
    fig = px.line(
        table,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        markers=True,
        labels={"avg_fare": "Average Fare"},
        title="Average fare by family size across passenger classes",
    )
    return fig


# -----------------------------
# Bonus — Age division vs class
# -----------------------------
def determine_age_division() -> pd.DataFrame:
    """
    Add a boolean column 'older_passenger' indicating whether a passenger's
    Age is above the median Age of *their* Pclass.
    """
    df = load_titanic()
    # compute within-class medians
    med = df.groupby("Pclass", observed=True)["Age"].median()
    df = df.assign(
        class_median=df["Pclass"].map(med),
        older_passenger=(df["Age"] > df["class_median"]),
    )
    return df.drop(columns=["class_median"])


def visualize_age_division(df: pd.DataFrame) -> "plotly.graph_objs._figure.Figure":
    """
    Bar: survival rate by older_passenger (True/False) faceted by Pclass.
    """
    use = df.dropna(subset=["older_passenger", "Survived", "Pclass"])
    agg = (
        use.groupby(["Pclass", "older_passenger"], observed=True)["Survived"]
        .mean()
        .reset_index(name="survival_rate")
    )
    fig = px.bar(
        agg,
        x="older_passenger",
        y="survival_rate",
        facet_col="Pclass",
        text="survival_rate",
        labels={"older_passenger": "Older than class median age"},
        title="Survival rate by older/younger within class",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
    return fig