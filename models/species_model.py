# species_model.py
"""
Core species model primitives for simple evolutionary / ecological simulations.

Design:
- SpeciesModel: encapsulates state and parameterized per-timestep updates.
- Simple interactions: resource consumption, reproduction (logistic-like),
  mutation drift (affects behavior traits), and interspecies influence.

This is intentionally modular: add your behavioral, neural, or evolutionary modules here.
"""
import json
import math
import random
from typing import Dict, Any

class SpeciesModel:
    def __init__(self, descriptor: Dict[str, Any], initial_population: int = None, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.desc = descriptor
        self.id = descriptor.get("id")
        self.params = descriptor.get("simulation_parameters", {})
        self.behavior = descriptor.get("behavioral_traits", {})
        self.population = initial_population if initial_population is not None else max(10, int(self.params.get("carrying_capacity", 1000) * 0.05))
        self.carrying_capacity = self.params.get("carrying_capacity", 10000)
        self.resource_consumption = self.params.get("resource_consumption_per_capita", 1.0)
        self.reproduction_rate = self.params.get("reproduction_rate", 0.01)
        self.mutation_rate = self.params.get("mutation_rate", 0.001)
        # internal stats
        self.resource_reserve = self.population * self.resource_consumption * 1.0  # abstract
        self.traits = dict(self.behavior)  # copy of behavioral trait values that can drift
        self.history = []

    @classmethod
    def load_from_file(cls, path: str, initial_population: int = None, seed: int = None):
        with open(path, 'r', encoding='utf-8') as f:
            descriptor = json.load(f)
        return cls(descriptor, initial_population=initial_population, seed=seed)

    def step(self, external_pressure: float = 0.0, influence_factors: Dict[str, float] = None):
        """
        Single timestep update:
        - compute birth events (logistic)
        - compute death events (resource scarcity + external pressure)
        - allow trait drift (mutation)
        - optional influence from other species (increase/decrease cohesion, tool use)
        """
        # Reproduction: logistic-like growth modulated by resource and social_cohesion
        social = self.traits.get("social_cohesion_index", 0.5)
        r = self.reproduction_rate * (0.5 + 0.5 * social)  # social increases reproduction efficiency
        growth_potential = r * self.population * (1 - self.population / max(1, self.carrying_capacity))
        births = max(0, int(growth_potential))

        # Deaths: resource scarcity + external pressure (e.g., predation, climate)
        resource_factor = self.resource_reserve / max(1.0, self.population * self.resource_consumption)
        # if resources insufficient, increased mortality
        death_rate = (1.0 - min(1.0, resource_factor)) * 0.5 + external_pressure
        deaths = min(self.population, int(self.population * death_rate))

        # Apply influence factors from others (dictionary of trait adjustments)
        if influence_factors:
            for trait, delta in influence_factors.items():
                if trait in self.traits:
                    # small instantaneous change
                    self.traits[trait] = max(0.0, min(1.0, self.traits[trait] + delta))

        # Trait drift (mutation): small random walk
        for t in list(self.traits.keys()):
            if random.random() < self.mutation_rate:
                drift = random.gauss(0, 0.01)
                self.traits[t] = max(0.0, min(1.0, self.traits[t] + drift))

        # Update population and reserves
        self.population = max(0, self.population + births - deaths)
        # resources consumption and replenishment (very abstract)
        consumption = self.population * self.resource_consumption
        # small replenishment proportional to carrying capacity (environment baseline)
        replenishment = (self.carrying_capacity - self.population) / max(1, self.carrying_capacity) * (self.resource_consumption * 0.5 * self.population)
        self.resource_reserve = max(0.0, self.resource_reserve - consumption + replenishment)

        # record history
        record = {
            "population": self.population,
            "births": births,
            "deaths": deaths,
            "resource_reserve": self.resource_reserve,
            "traits": dict(self.traits)
        }
        self.history.append(record)
        return record

    def influence_others(self):
        """
        Compute a small influence vector that this species projects to others.
        For demonstration, influence is derived from key traits.
        Returns a dict of trait deltas keyed by trait name.
        """
        # Example: if high social cohesion, positively influence other's social_cohesion.
        out = {}
        social = self.traits.get("social_cohesion_index", 0.5)
        tool = self.traits.get("tool_use_probability", 0.5)
        exploratory = self.traits.get("exploratory_tendency", 0.5)
        # influences are scaled and signed
        out["social_cohesion_index"] = (social - 0.5) * 0.02
        out["tool_use_probability"] = (tool - 0.5) * 0.01
        out["exploratory_tendency"] = (exploratory - 0.5) * 0.01
        return out

    def summarize(self):
        last = self.history[-1] if self.history else {}
        return {
            "id": self.id,
            "population": self.population,
            "traits": dict(self.traits),
            "last_step": last
        }
