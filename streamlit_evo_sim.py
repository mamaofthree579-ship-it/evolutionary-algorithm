# streamlit_evo_sim.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import time
import math

# ---------------------------
# Utility & dataclasses
# ---------------------------

TRAIT_NAMES = [
    "mobility",          # 0 aquatic - 1 terrestrial
    "intelligence",      # 0..1 general problem solving
    "sociality",         # 0..1 social cooperation
    "tool_use",          # 0..1 propensity for tool manufacture
    "neural_complexity", # 0..1 costed internal complexity
    "bipedal_tendency",  # 0..1 tendency toward bipedalism
    "symbiosis",         # 0..1 tendency for symbiotic intelligence (mycelial, electric)
]

ENV_NAMES = [
    "water_level",       # 0 dry - 1 aquatic
    "temperature",       # normalized 0..1
    "predation",         # 0..1
    "resources",         # 0..1 abundance
    "social_opportunity",# 0..1
    "tech_pressure",     # 0..1 (pressure to develop tools/technology)
]

np.random.seed(42)

@dataclass
class Individual:
    genome: np.ndarray  # vector same length as TRAIT_NAMES
    phenotype: np.ndarray = field(default=None)  # phenotype may differ due to plasticity
    age: int = 0

    def __post_init__(self):
        if self.phenotype is None:
            self.phenotype = self.genome.copy()

@dataclass
class Population:
    individuals: List[Individual]

    def as_dataframe(self, trait_names=TRAIT_NAMES) -> pd.DataFrame:
        rows = []
        for i, ind in enumerate(self.individuals):
            row = {f"{t}": float(ind.phenotype[j]) for j, t in enumerate(trait_names)}
            row["age"] = ind.age
            row["id"] = i
            rows.append(row)
        return pd.DataFrame(rows)

# ---------------------------
# Core evolutionary functions
# ---------------------------

def init_population(n: int, genome_size: int) -> Population:
    inds = []
    for _ in range(n):
        genome = np.clip(np.random.normal(loc=0.5, scale=0.15, size=genome_size), 0.0, 1.0)
        inds.append(Individual(genome=genome))
    return Population(inds)

def fitness_function(ind: Individual, env: Dict[str, float], mode_flags: Dict[str,bool]) -> float:
    """
    Compute fitness as multi-factor match + synergy bonuses - cost penalties.

    - mobility matches water_level
    - intelligence, tool_use, neural_complexity give access to tech_pressure & resources but cost energy
    - sociality multiplies benefit from social_opportunity
    - symbiosis gains in special ecosystems (low resources, high mycelial avail)
    """
    g = ind.phenotype
    # map indices
    mobility = g[0]
    intelligence = g[1]
    sociality = g[2]
    tool_use = g[3]
    neural = g[4]
    bipedal = g[5]
    symbiosis = g[6]

    # environmental values
    water = env["water_level"]
    temp = env["temperature"]
    pred = env["predation"]
    res = env["resources"]
    social = env["social_opportunity"]
    tech = env["tech_pressure"]

    # basic match components
    # mobility match: if environment aquatic and mobility low -> good, else terrestrial
    mobility_match = 1.0 - abs(mobility - (1.0 - water))  # mobility=1 wants terrestrial; water=1 wants aquatic (mobility~0)
    # but scaled so high is good if close
    mobility_score = max(0.0, mobility_match)

    # intelligence/tool synergy: intelligence & tool_use beneficial when resources & tech pressure high
    intel_score = (intelligence * 0.6 + tool_use * 0.4) * (0.2 + 0.8 * (res * 0.6 + tech * 0.4))

    # social benefit: sociality * social opportunity
    social_score = sociality * social

    # symbiosis advantage in low-resource & specialized environments (if symbiosis mode on)
    symbiosis_score = 0.0
    if mode_flags.get("mycelial") or mode_flags.get("electric"):
        # symbiosis helps when resources low OR environment has water/soil connectivity
        env_connectivity = 0.5 * (1.0 - res) + 0.5 * water
        symbiosis_score = symbiosis * env_connectivity

    # bipedalism: beneficial for tool use + endurance running in terrestrial, penalized in water
    bipedal_score = bipedal * (1.0 - water) * (0.3 + 0.7 * tool_use)

    # costs: neural complexity and high intelligence cost energy, especially under predation
    cost = neural * 0.5 + intelligence * 0.2 + tool_use * 0.1
    pred_penalty = pred * 0.3 * (1.0 - sociality)  # predators hurt isolated individuals more

    base = 0.35 * mobility_score + 0.25 * intel_score + 0.2 * social_score + 0.1 * symbiosis_score + 0.1 * bipedal_score
    fitness = base - (0.6 * cost) - pred_penalty
    # clamp and add small floor
    fitness = max(0.001, fitness)
    return fitness

def behavioral_plasticity(ind: Individual, env: Dict[str, float], plasticity_strength: float) -> None:
    """
    Short-term phenotype adjustments due to learning and behavior that increase short-run fitness.
    A fraction of the phenotype is temporarily shifted toward an 'optimal' trait relative to the environment.
    """
    # compute "optimal" simple rules
    opt = np.zeros_like(ind.phenotype)
    # mobility optimal = reflect environment water level (0 aquatic->mobility=0)
    opt[0] = 1.0 - env["water_level"]
    # intelligence slightly up if tech pressure
    opt[1] = min(1.0, ind.phenotype[1] + env["tech_pressure"] * 0.2)
    # sociality adjusts to social_opportunity
    opt[2] = env["social_opportunity"]
    # tool_use responds to tech pressure and resource scarcity
    opt[3] = 0.5 * env["tech_pressure"] + 0.5 * (1.0 - env["resources"])
    # neural_complexity tends to increase slowly with intelligence
    opt[4] = ind.phenotype[4]  # less plastic
    # bipedal - respond to terrestrial bias
    opt[5] = max(0.0, ind.phenotype[5] * 0.9 + (1.0 - env["water_level"]) * 0.1)
    # symbiosis increases if resources low and connectivity high
    opt[6] = 0.5 * (1.0 - env["resources"]) + 0.5 * env["water_level"]

    # move phenotype some fraction toward opt
    ind.phenotype = np.clip(ind.phenotype + plasticity_strength * (opt - ind.phenotype), 0.0, 1.0)

def tournament_selection(pop: Population, env: Dict[str,float], mode_flags: Dict[str,bool], k=3) -> Individual:
    """
    Tournament selection returning a copied Individual (genome copied).
    """
    candidates = np.random.choice(pop.individuals, size=min(k, len(pop.individuals)), replace=False)
    best = None
    best_fit = -np.inf
    for c in candidates:
        f = fitness_function(c, env, mode_flags)
        if f > best_fit:
            best_fit = f
            best = c
    # return a deep copy individual with same genome
    return Individual(genome=best.genome.copy())

def crossover(parent_a: Individual, parent_b: Individual, rate=0.5) -> Individual:
    # single-point crossover for continuous genomes: do arithmetic crossover
    alpha = np.random.rand(parent_a.genome.size)
    child_genome = alpha * parent_a.genome + (1 - alpha) * parent_b.genome
    return Individual(genome=np.clip(child_genome, 0.0, 1.0))

def mutate(ind: Individual, mut_rate: float, mut_scale: float):
    # gaussian noise on genome with some genes mutated
    mask = np.random.rand(ind.genome.size) < mut_rate
    noise = np.random.normal(0, mut_scale, size=ind.genome.size)
    ind.genome = np.clip(ind.genome + mask * noise, 0.0, 1.0)
    # reset phenotype to genome baseline for next generation
    ind.phenotype = ind.genome.copy()

def step_generation(pop: Population, env: Dict[str,float], mode_flags: Dict[str,bool], params: Dict[str,Any]) -> Population:
    """
    One evolutionary generation: evaluate fitness, create next gen via selection/crossover/mutation,
    apply behavioral plasticity during lifetime for phenotype evaluation.
    """
    # lifetime plasticity applied before fitness calculations
    for ind in pop.individuals:
        behavioral_plasticity(ind, env, params["plasticity_strength"])

    # compute fitnesses
    fitnesses = np.array([fitness_function(ind, env, mode_flags) for ind in pop.individuals])
    # record relative fitness
    probs = fitnesses / fitnesses.sum()

    # reproduction
    new_inds = []
    pop_size = len(pop.individuals)
    for _ in range(pop_size):
        # selection by tournament (could also use roulette)
        p1 = tournament_selection(pop, env, mode_flags, k=params["tourney_k"])
        p2 = tournament_selection(pop, env, mode_flags, k=params["tourney_k"])
        child = crossover(p1, p2)
        mutate(child, mut_rate=params["mutation_rate"], mut_scale=params["mutation_scale"])
        new_inds.append(child)

    # age update
    for ind in new_inds:
        ind.age += 1

    return Population(new_inds)

# ---------------------------
# Streamlit UI / Controller
# ---------------------------

st.set_page_config(layout="wide", page_title="Evolutionary Simulator (Streamlit)")

st.title("Evolutionary Algorithm Simulator — Streamlit")
st.markdown(
    """
Interactive baseline simulator for trait-based evolutionary modeling.
Use the sidebar to configure environment, modes, and evolutionary parameters.
This is a modular scaffold to match the components of your research (intelligence pathways, symbioses, behavioral plasticity, etc.).
"""
)

# Sidebar controls
st.sidebar.header("Simulation control")
init_pop_size = st.sidebar.number_input("Initial population", min_value=20, max_value=2000, value=200, step=20)
gens = st.sidebar.number_input("Generations per run", min_value=1, max_value=2000, value=100, step=10)
random_seed = st.sidebar.number_input("Random seed", value=42)
np.random.seed(int(random_seed))

st.sidebar.header("Environment (values normalized 0..1)")
env = {
    "water_level": st.sidebar.slider("Water level (0 dry / 1 aquatic)", 0.0, 1.0, 0.3),
    "temperature": st.sidebar.slider("Temperature (0..1)", 0.0, 1.0, 0.5),
    "predation": st.sidebar.slider("Predation pressure", 0.0, 1.0, 0.4),
    "resources": st.sidebar.slider("Resource abundance", 0.0, 1.0, 0.6),
    "social_opportunity": st.sidebar.slider("Social opportunity", 0.0, 1.0, 0.5),
    "tech_pressure": st.sidebar.slider("Tech pressure", 0.0, 1.0, 0.3),
}

st.sidebar.header("Alternative intelligence / modes")
mode_mycelial = st.sidebar.checkbox("Enable mycelial / fungal networks", value=True)
mode_electric = st.sidebar.checkbox("Enable electrical communication (aquatic/electric)", value=True)
mode_vocal = st.sidebar.checkbox("Enable vocal/visual communication (land birds, primates)", value=True)
mode_flags = {"mycelial": mode_mycelial, "electric": mode_electric, "vocal": mode_vocal}

st.sidebar.header("Evolution parameters")
params = {
    "mutation_rate": st.sidebar.slider("Mutation rate (per gene)", 0.0, 0.5, 0.05),
    "mutation_scale": st.sidebar.slider("Mutation STD", 0.0, 0.5, 0.07),
    "tourney_k": int(st.sidebar.number_input("Tournament size", min_value=2, max_value=10, value=3)),
    "plasticity_strength": st.sidebar.slider("Behavioral plasticity strength", 0.0, 1.0, 0.12),
}

# Parallel scenarios controls
st.sidebar.header("Parallel scenarios")
n_scenarios = st.sidebar.number_input("Scenarios to run in parallel (sequential execution)", min_value=1, max_value=6, value=2)
scenario_variation = st.sidebar.selectbox("Vary what across scenarios?", options=["environment", "mode_flags", "mutation_rate", "none"])

# Buttons: run single or parallel
run_single = st.sidebar.button("Run single scenario")
run_parallel = st.sidebar.button("Run parallel scenarios")

# Pre-seed populations per scenario
def run_simulation(init_pop, env_local, mode_flags_local, params_local, generations):
    pop = init_pop
    history = []
    for g in range(generations):
        # record stats
        df = pop.as_dataframe()
        means = df[TRAIT_NAMES].mean().to_dict()
        stdevs = df[TRAIT_NAMES].std().to_dict()
        means.update({f"{k}_sd": v for k, v in stdevs.items()})
        means["population"] = len(pop.individuals)
        history.append(means)
        pop = step_generation(pop, env_local, mode_flags_local, params_local)
    return pop, pd.DataFrame(history)

# Create initial population
if run_single:
    st.sidebar.success("Running single scenario...")
    init_pop = init_population(init_pop_size, genome_size=len(TRAIT_NAMES))

    with st.spinner("Simulating..."):
        final_pop, hist = run_simulation(init_pop, env, mode_flags, params, int(gens))

    st.success("Simulation complete — displaying results")

    # Plots
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Trait means over time")
        fig, ax = plt.subplots(figsize=(8,4))
        for t in TRAIT_NAMES:
            ax.plot(hist[t], label=t)
        ax.legend(fontsize="small")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Trait mean")
        st.pyplot(fig)

    with col2:
        st.subheader("Trait distribution (final generation)")
        df_final = final_pop.as_dataframe()
        fig2, axs = plt.subplots(2, math.ceil(len(TRAIT_NAMES)/2), figsize=(12,5))
        axs = axs.flatten()
        for i, t in enumerate(TRAIT_NAMES):
            axs[i].hist(df_final[t], bins=20)
            axs[i].set_title(t)
        plt.tight_layout()
        st.pyplot(fig2)

    st.subheader("Population sample (final gen)")
    st.dataframe(df_final.sample(min(200, len(df_final))).reset_index(drop=True))

    st.subheader("History (first 10 rows)")
    st.dataframe(hist.head(10))

if run_parallel:
    st.sidebar.success("Running parallel scenarios...")
    results = []
    all_hist = {}
    for s in range(int(n_scenarios)):
        # vary scenario param
        env_s = env.copy()
        mode_s = mode_flags.copy()
        params_s = params.copy()
        if scenario_variation == "environment":
            # vary water level across scenarios as an example
            env_s["water_level"] = min(1.0, max(0.0, env["water_level"] + (s - (n_scenarios-1)/2) * 0.15))
        elif scenario_variation == "mode_flags":
            # toggle different modes
            toggles = list(mode_flags.keys())
            key = toggles[s % len(toggles)]
            mode_s[key] = not mode_s[key]
        elif scenario_variation == "mutation_rate":
            params_s["mutation_rate"] = min(0.5, max(0.0, params["mutation_rate"] + (s - (n_scenarios-1)/2) * 0.02))
        # init population (same seed for reproducibility offsets)
        init_pop = init_population(init_pop_size, genome_size=len(TRAIT_NAMES))
        final_pop, hist = run_simulation(init_pop, env_s, mode_s, params_s, int(gens))
        results.append((env_s, mode_s, params_s, final_pop, hist))
        all_hist[f"scenario_{s}"] = hist

    # summary plots: mean intelligence across scenarios
    st.subheader("Parallel scenario comparison: intelligences over time")
    fig3, ax3 = plt.subplots(figsize=(10,4))
    for i, (_, mode_s, _, _, hist) in enumerate(results):
        ax3.plot(hist["intelligence"], label=f"scenario {i}")
    ax3.legend()
    st.pyplot(fig3)

    # show final trait table per scenario
    st.subheader("Final trait means per scenario")
    rows = []
    for i, (env_s, mode_s, params_s, final_pop, hist) in enumerate(results):
        df_final = final_pop.as_dataframe()
        row = {"scenario": i}
        row.update({f"{t}_mean": float(df_final[t].mean()) for t in TRAIT_NAMES})
        row.update({"water_level": env_s["water_level"], "modes": str(mode_s), "mutation_rate": params_s["mutation_rate"]})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows))

st.markdown("---")
st.markdown("## Notes & next steps")
st.markdown(
    """
- This app is a scaffold. Replace fitness_function with more sophisticated ecological / neural / cultural models as required.
- You can plug in real datasets (fossil traits, primate behavioral datasets, cetacean communication metrics) to initialize genomes or define environmental histories.
- To simulate cultural inheritance, add an explicit cultural trait vector and horizontal transmission rules.
- For large populations or heavy parallel runs, move the simulation engine to a backend (FastAPI, Ray, or Dask) and stream results to Streamlit.
"""
)

st.markdown("If you'd like, I can now:")
st.markdown("- Add realistic modules for **bipedalism biomechanics** and **hair loss energetics** (Savanna vs aquatic scenarios).  \n- Integrate available datasets for **cephalopod neural maps** / **corvid problem solving** to seed alternative intelligence modes.  \n- Add an export to CSV / JSON and a small UI for saving parameter presets.")
if st.button("Ask for further customization"):
    st.info("Tell me which feature you'd like added (e.g., evolutionary pressure time series, genetic drift vs selection controls, cultural inheritance rules, or an interactive map).")
                     
