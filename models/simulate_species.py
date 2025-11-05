import os
import csv

def run_simulation(steps=200, seed=42, verbose=True):
    # instantiate models (initial populations chosen heuristically)
    naga = create_naga_model(initial_population=3000, seed=seed+1)
    ants = create_ant_people_model(initial_population=50000, seed=seed+2)
    winged = create_winged_model(initial_population=2000, seed=seed+3)
    trees = create_trees_model(initial_population=40000, seed=seed+4)

    species = [naga, ants, winged, trees]

    # header (for CSV output)
    headers = ["step"]
    for s in species:
        headers += [f"{s.id}_pop", f"{s.id}_res", f"{s.id}_soc"]

    if verbose:
        writer = csv.writer(open("simulation_log.csv", "w", newline=''))
        writer.writerow(headers)

    for step in range(steps):
        # compute pairwise influences (very simple: average influence from all others)
        influences = {}
        for s in species:
            influences[s.id] = s.influence_others()

        # each species receives the mean influence of others
        for s in species:
            mean_influence = {}
            for other in species:
                if other.id == s.id:
                    continue
                for trait, delta in influences[other.id].items():
                    mean_influence[trait] = mean_influence.get(trait, 0.0) + delta
            # normalize
            for k in mean_influence.keys():
                mean_influence[k] = mean_influence[k] / max(1, len(species)-1)
            # external pressure: simple environmental stress proportional to total pop / total carrying capacity
            total_pop = sum([x.population for x in species])
            total_capacity = sum([x.carrying_capacity for x in species])
            env_pressure = max(0.0, (total_pop / max(1, total_capacity)) - 0.6) * 0.5  # only when crowding past threshold
            s.step(external_pressure=env_pressure, influence_factors=mean_influence)

        # log
        row = [step]
        for s in species:
            row += [s.population, round(s.resource_reserve, 2), round(s.traits.get("social_cohesion_index", 0.0), 3)]
        if verbose:
            writer.writerow(row)

    # return models for inspection
    return species

if __name__ == "__main__":
    species = run_simulation(steps=300, seed=123, verbose=True)
    # print final summaries
    for s in species:
        print(s.summarize())
