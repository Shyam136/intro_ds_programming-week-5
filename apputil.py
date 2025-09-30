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
    Prefer a local Kaggle-style CSV with a Name column; only fall back to seaborn if nothing found.
    """
    if candidates is None:
        candidates = (
            # common grader paths
            "data/train.csv", "./data/train.csv", "../data/train.csv",
            "train.csv", "./train.csv", "../train.csv",
            "data/titanic.csv", "./data/titanic.csv", "../data/titanic.csv",
            "data/titanic_train.csv", "./data/titanic_train.csv", "../data/titanic_train.csv",
            # occasionally nested
            "titanic/train.csv", "./titanic/train.csv", "../titanic/train.csv",
            "../input/titanic/train.csv",
        )

    # try explicit candidates
    for p in candidates:
        if Path(p).is_file():
            return pd.read_csv(p)

    # tiny glob pass in common data dirs
    for root in (Path("."), Path(".."), Path("./data"), Path("../data")):
        hits = list(root.glob("**/*titanic*/*.csv")) + list(root.glob("**/*train*.csv")) + list(root.glob("**/*titanic*.csv"))
        for h in hits:
            try:
                df = pd.read_csv(h)
                if "Name" in df.columns:  # prefer a Kaggle-like file
                    return df
            except Exception:
                pass

    # fallback: seaborn (no Name column)
    import seaborn as sns
    return sns.load_dataset("titanic")

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to ensure BOTH Kaggle-style TitleCase and seaborn-style lowercase columns exist.
    """
    out = df.copy()

    # seaborn -> Kaggle titles (if needed)
    title_from_lower = {
        "survived": "Survived",
        "pclass": "Pclass",
        "sex": "Sex",
        "age": "Age",
        "sibsp": "SibSp",
        "parch": "Parch",
        "fare": "Fare",
        "embarked": "Embarked",
        "class": "Pclass",   # rarely used; keep Pclass as numeric if available
        "who": "Sex",        # not used, but avoid collisions
    }
    for low, title in title_from_lower.items():
        if low in out.columns and title not in out.columns:
            out[title] = out[low]

    # also guarantee lowercase mirrors exist
    for title in ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Name"]:
        if title in out.columns and title.lower() not in out.columns:
            out[title.lower()] = out[title]

    return out

# -----------------------------
# Exercise 1 — Demographics
# -----------------------------
def survival_demographics() -> pd.DataFrame:
    """
    Return LOWERCASE columns: pclass, sex, age_group, n_passengers, n_survivors, survival_rate.
    Includes rows for combos that have zero members (n_passengers = 0).
    """
    raw = load_titanic()
    df = _norm_cols(raw)

    # Age groups – categorical (ordered)
    bins = [-np.inf, 12, 19, 59, np.inf]
    labels = ["Child", "Teen", "Adult", "Senior"]
    age_group = pd.cut(df["age"], bins=bins, labels=labels, ordered=True)

    use = df.assign(age_group=age_group)[["pclass", "sex", "age_group", "survived"]]

    # group on observed data
    g = (
        use.dropna(subset=["pclass", "sex", "age_group"])
           .groupby(["pclass", "sex", "age_group"], observed=True)
           .agg(n_passengers=("survived", "size"),
                n_survivors=("survived", "sum"))
    )

    # reindex to ALL combinations so zero-member groups appear
    all_idx = pd.MultiIndex.from_product(
        [
            sorted(use["pclass"].dropna().unique()),
            sorted(use["sex"].dropna().unique()),
            pd.Categorical(labels, categories=labels, ordered=True)
        ],
        names=["pclass", "sex", "age_group"]
    )
    g = g.reindex(all_idx, fill_value=0)

    # survival_rate: 0 when n_passengers == 0
    g = g.assign(
        survival_rate=np.where(g["n_passengers"] > 0,
                               g["n_survivors"] / g["n_passengers"],
                               0.0)
    ).reset_index()

    # enforce lowercase column names and order
    g.columns = ["pclass", "sex", "age_group", "n_passengers", "n_survivors", "survival_rate"]
    # keep age_group as categorical
    g["age_group"] = pd.Categorical(g["age_group"], categories=labels, ordered=True)
    # and sort for readability
    g = g.sort_values(["pclass", "sex", "age_group"]).reset_index(drop=True)
    return g



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
    Return counts of last names (index = last name, value = count).
    Requires a 'Name' column; loader now aggressively searches for Kaggle CSV.
    """
    raw = load_titanic()
    df = _norm_cols(raw)
    if "Name" not in df.columns and "name" not in df.columns:
        return pd.Series(dtype="int64")

    name_series = df["Name"] if "Name" in df.columns else df["name"]
    last = name_series.astype(str).str.extract(r"^\s*([^,]+)\s*,", expand=False)
    return last.dropna().str.strip().value_counts()


def family_groups() -> pd.DataFrame:
    """
    Table grouped by family_size and Pclass with n_passengers, avg_fare, min_fare, max_fare.
    """
    raw = load_titanic()
    df = _norm_cols(raw)

    # family_size = SibSp + Parch + 1
    fam = (df["SibSp"].fillna(0).astype(int)
           + df["Parch"].fillna(0).astype(int) + 1)
    tmp = df.assign(family_size=fam)

    out = (
        tmp.groupby(["Pclass", "family_size"], dropna=False)
           .agg(n_passengers=("Fare", "size"),
                avg_fare=("Fare", "mean"),
                min_fare=("Fare", "min"),
                max_fare=("Fare", "max"))
           .reset_index()
           .sort_values(["Pclass", "family_size"])
           .reset_index(drop=True)
    )
    return out


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
    Add 'class_median' and 'older_passenger'. For rows where Age is NA,
    set older_passenger to <NA> (nullable boolean) so NA count matches Age NA count.
    """
    raw = load_titanic()
    df = _norm_cols(raw).copy()

    med = df.groupby("pclass", observed=True)["age"].median()
    df["class_median"] = df["pclass"].map(med)

    # nullable boolean with NA where age is NA
    older = pd.Series(pd.NA, index=df.index, dtype="boolean")
    mask = df["age"].notna() & df["class_median"].notna()
    older.loc[mask] = df.loc[mask, "age"] > df.loc[mask, "class_median"]
    df["older_passenger"] = older
    return df


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