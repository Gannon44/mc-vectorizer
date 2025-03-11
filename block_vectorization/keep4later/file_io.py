import os
import numpy as np

def load_npy_files(directory):
    """
    Scans a directory for .npy files and loads them.
    Returns a dictionary mapping file path to numpy array.
    """
    npy_files = {}
    for fname in os.listdir(directory):
        if fname.endswith(".npy") or fname.endswith(".npy.txt"):
            path = os.path.join(directory, fname)
            try:
                arr = np.load(path, allow_pickle=True)
                npy_files[path] = arr
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    return npy_files

def save_structure(structure, path):
    """
    Saves a numpy array to the given path.
    """
    np.save(path, structure)
