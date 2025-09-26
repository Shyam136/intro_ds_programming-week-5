# app.py
import streamlit as st
import plotly.io as pio

from apputil import (
    survival_demographics,
    visualize_demographic,
    family_groups,
    visualize_families,
    last_names,
    determine_age_division,
    visualize_age_division,
)

st.set_page_config(page_title="Titanic: Week 5", layout="wide")
st.title("Week 5 — Titanic Patterns (Plotly + pandas)")

# ===== Exercise 1 =====
st.header("Exercise 1: Survival Patterns")

# Question prompt (put your own wording here)
st.write("Question: Do women and children in higher classes have notably higher survival rates than men across classes?")

demo_tbl = survival_demographics()
st.dataframe(demo_tbl, use_container_width=True)
fig1 = visualize_demographic(demo_tbl)
st.plotly_chart(fig1, use_container_width=True)

# ===== Exercise 2 =====
st.header("Exercise 2: Family Size and Wealth")

fam_tbl = family_groups()
st.dataframe(fam_tbl, use_container_width=True)

# Cross-check with last names
ln = last_names()
st.write("Top 10 last names (count):")
st.write(ln.head(10))

# Question prompt for families
st.write("Question: How does average fare change with family size across classes?")
fig2 = visualize_families(fam_tbl)
st.plotly_chart(fig2, use_container_width=True)

# ===== Bonus =====
with st.expander("Bonus: Older vs. younger within class"):
    df_age = determine_age_division()
    fig3 = visualize_age_division(df_age)
    st.plotly_chart(fig3, use_container_width=True)