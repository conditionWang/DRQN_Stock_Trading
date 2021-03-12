import json, codecs
import numpy as np

DIRECTORY = "F:/Dev/Data/truefx/"

def hot_encoding(a):
    a_ = np.zeros(3, dtype=np.float32)
    a_[a + 1] = 1.
    return a_


def load_data_structure(file):
    return json.load(codecs.open(file, 'r', encoding='utf-8'))

def save_data_structure(structure, file):
    json.dump(structure, codecs.open(file, 'w', encoding='utf-8'), sort_keys=True, indent=4)