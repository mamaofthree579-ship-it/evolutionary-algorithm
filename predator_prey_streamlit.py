import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Evolutionary Energy Network", layout="wide")

st.title("🌍 Evolutionary Harmony Network")

# --- Sidebar controls ---
st.sidebar.header("Simulation Controls")
num_species = st.sidebar.slider("Number of Species / Groups", 3, 12, 6)
iterations = st.sidebar.slider("Simulation Steps", 10, 1000, 100)
imbalance_factor = st.sidebar.slider("Initial Imbalance", 0.1, 1.0, 0.5)
redistribution_rate = st.sidebar.slider("Redistribution Rate", 0.01, 0.5, 0.1)

# --- Initialize data ---
np.random.seed(42)
energy = np.abs(np.random.randn(num_species)) * imbalance_factor
energy /= np.sum(energy)

species = [f"Species {i+1}" for i in range(num_species)]
df = pd.DataFrame({"Species": species, "Energy": energy})

# --- Simulate redistribution ---
history = [df["Energy"].values.copy()]

for _ in range(iterations):
    mean_energy = np.mean(df["Energy"])
    df["Energy"] += redistribution_rate * (mean_energy - df["Energy"])
    df["Energy"] = np.clip(df["Energy"], 0, 1)
    df["Energy"] /= np.sum(df["Energy"])
    history.append(df["Energy"].values.copy())

# --- Convert to DataFrame for visualization ---
hist_array = np.array(history)
hist_df = pd.DataFrame(hist_array, columns=species)

# --- Plot Evolution over Time ---
fig = go.Figure()
for s in species:
    fig.add_trace(go.Scatter(y=hist_df[s], mode='lines', name=s))

fig.update_layout(
    title="Energy Redistribution and Harmony Over Time",
    xaxis_title="Iteration",
    yaxis_title="Normalized Energy Share",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# --- Display final equilibrium ---
st.subheader("🔹 Final Energy Distribution")
st.bar_chart(df.set_index("Species"))

# --- Summary Metrics ---
variance = np.var(df["Energy"])
st.markdown(f"**Energy Variance:** {variance:.6f}")
if variance < 0.001:
    st.success("System has reached near-perfect energetic harmony ✨")
else:
    st.info("Redistribution still in progress...")

# --- Optional Narrative Output ---
if st.checkbox("Show Interpretive Narrative"):
    st.markdown(
        """
        In this simulation, each ‘species’ represents a node within a greater field of
        collective resonance. Over time, excess energy disperses toward regions of need,
        stabilizing the system. The lower the variance, the greater the harmony achieved.
        """
    )
