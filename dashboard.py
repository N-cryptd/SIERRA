
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("SIERRA Experiment Dashboard")

# Load experiment results
try:
    df = pd.read_csv("experiment_results.csv")
    st.dataframe(df)

    # Plot results
    fig, ax = plt.subplots()
    ax.bar(df["name"], df["mean_reward"])
    ax.set_ylabel("Mean Reward")
    ax.set_title("Experiment Results")
    st.pyplot(fig)
except FileNotFoundError:
    st.warning("No experiment results found. Run `scripts/run_experiments.py` to generate results.")
