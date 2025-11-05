import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -------------------------
# Model & configuration
# -------------------------
species_names = ["Naga", "AntPeople", "Winged", "Oceanic", "Crystalline", "HollowWalker"]

base_params = {
    "Naga": {"pop": 800, "int": 0.35, "tech": 0.25, "cohesion": 0.6},
    "AntPeople": {"pop": 1200, "int": 0.25, "tech": 0.15, "cohesion": 0.85},
    "Winged": {"pop": 400, "int": 0.5, "tech": 0.45, "cohesion": 0.5},
    "Oceanic": {"pop": 600, "int": 0.45, "tech": 0.35, "cohesion": 0.55},
    "Crystalline": {"pop": 200, "int": 0.6, "tech": 0.6, "cohesion": 0.4},
    "HollowWalker": {"pop": 150, "int": 0.4, "tech": 0.2, "cohesion": 0.3},
}

symbol_names = ["WingSymbol", "SpiralGlyph", "ThreePoint", "WaterSigil", "FlameMark", "CrystalWeave"]

symbol_effects = {
    "WingSymbol":     {"int": 0.6, "tech": 0.4, "cohesion": 0.1},
    "SpiralGlyph":    {"int": 0.4, "cohesion": 0.5, "tech": 0.1},
    "ThreePoint":     {"tech": 0.6, "cohesion": 0.2, "int": 0.2},
    "WaterSigil":     {"cohesion": 0.6, "int": 0.2, "tech": 0.2},
    "FlameMark":      {"tech": 0.5, "int": 0.3, "cohesion": -0.1},
    "CrystalWeave":   {"int": 0.5, "tech": 0.4, "cohesion": 0.3},
}

# -------------------------
# Simulation function
# -------------------------
def run_simulation(years=200, dt=1.0, symbol_strengths=None, species_symbol_map=None, noise=0.01):
    times = np.arange(0, years+1, dt)
    state = {s: {"pop": np.zeros_like(times),
                 "int": np.zeros_like(times),
                 "tech": np.zeros_like(times),
                 "cohesion": np.zeros_like(times)} for s in species_names}
    for s in species_names:
        state[s]["pop"][0] = base_params[s]["pop"]
        state[s]["int"][0] = base_params[s]["int"]
        state[s]["tech"][0] = base_params[s]["tech"]
        state[s]["cohesion"][0] = base_params[s]["cohesion"]
    K = 2000
    for t_i in range(1, len(times)):
        for s in species_names:
            pop = state[s]["pop"][t_i-1]
            intel = state[s]["int"][t_i-1]
            tech = state[s]["tech"][t_i-1]
            coh = state[s]["cohesion"][t_i-1]
            carrying_pressure = 1 - (pop / (K + 1))
            growth_rate = 0.01 + 0.005*coh + 0.002*tech
            dpop = pop * growth_rate * carrying_pressure
            dpop += 0.3 * intel * pop * 0.0005
            dpop += np.random.normal(0, noise * pop)
            new_pop = max(0.0, pop + dpop)
            base_int_gain = 0.001 + 0.0008 * coh
            base_tech_gain = 0.0012 + 0.0006 * intel
            base_coh_gain = 0.0008 + 0.0004 * (1 - abs(tech - intel))
            int_bonus = tech_bonus = coh_bonus = 0.0
            if species_symbol_map and symbol_strengths:
                syms = species_symbol_map.get(s, [])
                for sym in syms:
                    strength = float(symbol_strengths.get(sym, 0.0))
                    eff = symbol_effects.get(sym, {})
                    int_bonus += eff.get("int", 0.0) * strength
                    tech_bonus += eff.get("tech", 0.0) * strength
                    coh_bonus += eff.get("cohesion", 0.0) * strength
            dint = base_int_gain * (1 + int_bonus) + np.random.normal(0, noise*0.001)
            dtech = base_tech_gain * (1 + tech_bonus) + np.random.normal(0, noise*0.001)
            dcoh = base_coh_gain * (1 + coh_bonus) + np.random.normal(0, noise*0.001)
            dint += 0.0005 * tech * coh
            dtech += 0.0006 * intel * coh
            new_int = np.clip(intel + dint, 0.0, 1.5)
            new_tech = np.clip(tech + dtech, 0.0, 1.5)
            new_coh = np.clip(coh + dcoh, 0.0, 1.0)
            state[s]["pop"][t_i] = new_pop
            state[s]["int"][t_i] = new_int
            state[s]["tech"][t_i] = new_tech
            state[s]["cohesion"][t_i] = new_coh
    # DataFrame
    records = []
    for idx, t in enumerate(times):
        for s in species_names:
            records.append({
                "time": t,
                "species": s,
                "pop": state[s]["pop"][idx],
                "intelligence": state[s]["int"][idx],
                "technology": state[s]["tech"][idx],
                "cohesion": state[s]["cohesion"][idx],
            })
    return pd.DataFrame.from_records(records)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Evolutionary Symbolic Simulator", layout="wide")
st.title("Evolutionary Symbolic Simulation — visual symbolic modifiers & species evolution")

# Left: controls
with st.sidebar:
    st.header("Simulation controls")
    years = st.slider("Years to simulate", 50, 1000, 200, step=50)
    noise = st.slider("Simulation noise", 0.0, 0.05, 0.01, step=0.001)
    st.markdown("### Symbol strengths (global)")
    symbol_strengths = {}
    for sym in symbol_names:
        symbol_strengths[sym] = st.slider(sym, 0.0, 1.0, 0.3, step=0.01)
    st.markdown("---\nAssign symbols to species (multi-select):")
    species_symbol_map = {}
    for s in species_names:
        species_symbol_map[s] = st.multiselect(f"{s} symbols", symbol_names, default=[])
    run = st.button("Run simulation")

st.write("### Symbol effects (for reference)")
st.write(pd.DataFrame(symbol_effects).T)

# Run simulation and display results
if run:
    with st.spinner("Running simulation..."):
        df = run_simulation(years=years, symbol_strengths=symbol_strengths,
                            species_symbol_map=species_symbol_map, noise=noise)
    st.success("Simulation complete — rendering plots")

    # Create 4 plots (population, intelligence, technology, cohesion) using plotly
    def plot_metric(metric, title):
        fig = go.Figure()
        for s in species_names:
            d = df[df["species"]==s]
            fig.add_trace(go.Scatter(x=d["time"], y=d[metric], mode="lines", name=s))
        fig.update_layout(title=title, xaxis_title="Time (years)", yaxis_title=metric)
        return fig

    col1, col2 = st.columns(2)
    col1.plotly_chart(plot_metric("pop", "Population"), use_container_width=True)
    col2.plotly_chart(plot_metric("intelligence", "Intelligence"), use_container_width=True)

    col3, col4 = st.columns(2)
    col3.plotly_chart(plot_metric("technology", "Technology"), use_container_width=True)
    col4.plotly_chart(plot_metric("cohesion", "Cohesion"), use_container_width=True)

    # show end-state summary:
    last = df[df['time']==df['time'].max()]
    summary = last[['species','pop','intelligence','technology','cohesion']].set_index('species')
    st.write("### End-state summary")
    st.dataframe(summary.style.format({"pop":"{:.0f}", "intelligence":"{:.3f}", "technology":"{:.3f}", "cohesion":"{:.3f}"}))

    st.markdown("**Tip:** Adjust symbol strengths and assignment, then Run again to see different evolutionary trajectories.")
else:
    st.info("Adjust controls in the sidebar and press **Run simulation** to begin.")
