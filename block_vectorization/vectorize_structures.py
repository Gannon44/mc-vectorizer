import os
import argparse
import numpy as np
import json

def load_npy_files(input_dir):
    npy_files = {}
    for fname in os.listdir(input_dir):
        if fname.endswith(".npy"):
            path = os.path.join(input_dir, fname)
            try:
                arr = np.load(path, allow_pickle=True)
                npy_files[path] = arr
            except Exception as e:
                print(f"Error loading {path}: {e}")
    return npy_files

def save_structure(arr, out_path):
    np.save(out_path, arr)

def build_lookup(structure, block_id2name, block_name2latent, latent_dim=32):
    """
    Build a lookup dict mapping each unique value in the structure to its latent vector.
    Handles mixed types (integers and strings) separately.
    """
    lookup = {}

    # Extract unique values while handling mixed types
    unique_vals = set(structure.flatten())  # Use set to avoid NumPy sorting issue

    for val in unique_vals:
        if isinstance(val, (int, np.integer)):  # If it's an integer (e.g., 0)
            if val == 0:
                vec = block_name2latent.get("Air", np.zeros(latent_dim, dtype=np.float32))
            else:
                print(f"Warning: Unexpected integer value '{val}' found in structure.")
                vec = np.zeros(latent_dim, dtype=np.float32)
        elif isinstance(val, str):  # If it's a string (block ID)
            block_name = block_id2name.get(val, None)
            if block_name is None:
                print(f"Warning: Block ID '{val}' not found in block_id2name mapping. Using zero vector.")
                vec = np.zeros(latent_dim, dtype=np.float32)
            else:
                vec = block_name2latent.get(block_name, np.zeros(latent_dim, dtype=np.float32))
        else:
            print(f"Warning: Unsupported data type '{type(val)}' encountered. Using zero vector.")
            vec = np.zeros(latent_dim, dtype=np.float32)

        lookup[val] = vec
    return lookup


def vectorize_structure(structure, lookup, latent_dim=32):
    """
    Given a 3D structure, return a 4D array (H, W, D, latent_dim) where every value is replaced 
    by its corresponding latent vector from lookup.
    """
    H, W, D = structure.shape
    output = np.zeros((H, W, D, latent_dim), dtype=np.float32)
    for val, vec in lookup.items():
        mask = structure == val
        output[mask] = vec  # Broadcasting the 32-dim vector into each position.
    return output

def main(input_dir, output_dir, block_id2name_path, block_name2latent_path, latent_dim=32):
    # Load mapping files.
    with open(block_id2name_path, "r") as f:
        block_id2name = json.load(f)
    with open(block_name2latent_path, "r") as f:
        block_name2latent_raw = json.load(f)
    # Convert latent vectors from lists to numpy arrays.
    block_name2latent = {k: np.array(v, dtype=np.float32) for k, v in block_name2latent_raw.items()}
    
    os.makedirs(output_dir, exist_ok=True)
    npy_files = load_npy_files(input_dir)
    print(f"Found {len(npy_files)} .npy files in '{input_dir}'")
    
    for file_path, structure in npy_files.items():
        print(f"Processing file: {file_path}...")
        # Build lookup: for each unique value, determine its latent vector.
        lookup = build_lookup(structure, block_id2name, block_name2latent, latent_dim)
        vec_structure = vectorize_structure(structure, lookup, latent_dim)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(output_dir, base_name + "_vectorized.npy")
        save_structure(vec_structure, out_path)
        print(f"Saved vectorized file to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize .npy structures to 32-channel latent representations.")
    parser.add_argument("--input_dir", type=str, default="input_structures",
                        help="Directory containing input .npy files (default: input_structures)")
    parser.add_argument("--output_dir", type=str, default="vectorized_structures",
                        help="Directory to save vectorized .npy files (default: vectorized_structures)")
    parser.add_argument("--block_id2name", type=str, default="mappings/block_id2name.json",
                        help="Path to block_id2name JSON mapping file (default: mappings/block_id2name.json)")
    parser.add_argument("--block_name2latent", type=str, default="mappings/block_name2latent.json",
                        help="Path to block_name2latent JSON mapping file (default: mappings/block_name2latent.json)")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimensionality of latent representation (default: 32)")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.block_id2name, args.block_name2latent, args.latent_dim)
