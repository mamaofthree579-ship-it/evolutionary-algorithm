# trees_model.py
from species_model import SpeciesModel
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PATH = os.path.join(DATA_DIR, "trees.json")

def create_trees_model(initial_population=None, seed=3):
    return SpeciesModel.load_from_file(PATH, initial_population=initial_population, seed=seed)

if __name__ == "__main__":
    m = create_trees_model(initial_population=50000, seed=3)
    print(m.summarize())
