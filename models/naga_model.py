# naga_model.py
from species_model import SpeciesModel
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PATH = os.path.join(DATA_DIR, "naga.json")

def create_naga_model(initial_population=None, seed=0):
    return SpeciesModel.load_from_file(PATH, initial_population=initial_population, seed=seed)

if __name__ == "__main__":
    m = create_naga_model(initial_population=3000, seed=42)
    print(m.summarize())


---

ant_people_model.py

# ant_people_model.py
from species_model import SpeciesModel
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PATH = os.path.join(DATA_DIR, "ant_people.json")

def create_ant_people_model(initial_population=None, seed=1):
    return SpeciesModel.load_from_file(PATH, initial_population=initial_population, seed=seed)

if __name__ == "__main__":
    m = create_ant_people_model(initial_population=50000, seed=1)
    print(m.summarize())
  
