# winged_people_model.py
from species_model import SpeciesModel
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PATH = os.path.join(DATA_DIR, "winged_people.json")

def create_winged_model(initial_population=None, seed=2):
    return SpeciesModel.load_from_file(PATH, initial_population=initial_population, seed=seed)

if __name__ == "__main__":
    m = create_winged_model(initial_population=2000, seed=2)
    print(m.summarize())
