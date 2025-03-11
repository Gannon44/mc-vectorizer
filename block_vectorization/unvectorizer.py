import os
import json
import numpy as np
from scipy.spatial import KDTree

def load_mappings(block_name2latent_path, latent2block_name_path, block_name2id_path, block_id2color_path):
    """
    Loads the required mapping files.
    
    Returns:
      ordered_block_names: list of block names (order preserved)
      latent_array: NumPy array of latent vectors (float32)
      latent2block_name: dict mapping str(latent_vector) -> block name
      block_name2id: dict mapping block name -> block id
      block_id2color: dict mapping block id -> color (or None if not needed)
    """
    with open(block_name2latent_path, "r") as f:
        bn2latent = json.load(f)
    ordered_block_names = list(bn2latent.keys())
    latent_array = np.array([bn2latent[name] for name in ordered_block_names], dtype=np.float32)
    
    with open(latent2block_name_path, "r") as f:
        latent2block_name = json.load(f)
    with open(block_name2id_path, "r") as f:
        block_name2id = json.load(f)
    
    block_id2color = None
    if block_id2color_path:
        with open(block_id2color_path, "r") as f:
            block_id2color = json.load(f)
        
    return ordered_block_names, latent_array, latent2block_name, block_name2id, block_id2color

def reverse_vectorize_data(data, ordered_block_names, latent_array, latent2block_name, block_name2id, block_id2color, stop_at_block_id, latent_dim=32):
    """
    Given a single 4D NumPy array of shape (H, W, D, latent_dim), this function
    reverse–maps each latent vector:
      1. KDTree lookup finds the closest latent vector from our mapping.
      2. latent2block_name (or fallback to ordered list) recovers the block name.
      3. block_name2id converts the block name to a block id.
      4. If stop_at_block_id is False, block_id2color converts the block id to its color.
    
    Returns a new array of shape (H, W, D, C) where C=1 if stop_at_block_id is True
    (with block IDs encoded as strings in an object array) or C equals the color dimension.
    """
    if data.shape[-1] != latent_dim:
        raise ValueError(f"Expected latent dimension {latent_dim}, but got {data.shape[-1]}")
    
    H, W, D, _ = data.shape
    data_flat = data.reshape(-1, latent_dim)
    
    # Build KDTree for fast nearest–neighbor search.
    tree = KDTree(latent_array)
    distances, indices = tree.query(data_flat, k=1)
    
    results = []
    for idx in indices:
        # Get the nearest latent vector from our mapping.
        nearest_latent = latent_array[idx]
        key = str(nearest_latent.tolist())
        block_name = latent2block_name.get(key, ordered_block_names[idx])
        block_id = block_name2id.get(block_name, "unknown_block")
        if stop_at_block_id:
            results.append(block_id)
        else:
            # Map block id to color; default to black (or transparent black if using 4 channels)
            color = block_id2color.get(block_id, [0, 0, 0, 0])
            results.append(color)
    
    # Reshape results back into spatial dimensions.
    if stop_at_block_id:
        # We want to return an object array of strings.
        result_arr = np.array(results, dtype=object).reshape(H, W, D)
    else:
        result_arr = np.array(results, dtype=np.float32).reshape(H, W, D, -1)
    return result_arr

def reverse_vectorize(input_data, output=None,
                      block_name2latent_path="mappings/block_name2latent.json",
                      latent2block_name_path="mappings/latent2block_name.json",
                      block_name2id_path="mappings/block_name2id.json",
                      block_id2color_path="mappings/block_id2color.json",
                      stop_at_block_id=False, latent_dim=32):
    """
    Reverse–maps a vectorized structure (or structures) into either block IDs or colors.
    
    Parameters:
      input_data: Either
         - A string (file or directory path),
         - A NumPy array (single structure, shape (H, W, D, latent_dim)), or
         - A list/tuple of NumPy arrays.
      output: (Optional) If input_data is a file or folder path, then output is:
         - A string: directory (if input_data is a folder) or file path (if single file)
         If input_data is already loaded data, output is ignored and the processed data is returned.
      block_name2latent_path, latent2block_name_path, block_name2id_path, block_id2color_path:
         Paths to the JSON mapping files.
      stop_at_block_id: If True, the function stops at block IDs; otherwise it converts to color.
      latent_dim: Dimensionality of latent vectors (default: 32).
      
    Returns:
      If input_data is loaded data, returns the processed NumPy array (or list of arrays).
      Otherwise, processes and saves the files in the output folder (or file) and returns a dictionary
      mapping file names to processed arrays.
    """
    # Load mappings.
    ordered_block_names, latent_array, latent2block_name, block_name2id, block_id2color = load_mappings(
        block_name2latent_path, latent2block_name_path, block_name2id_path,
        None if stop_at_block_id else block_id2color_path
    )
    
    outputs = {}
    # Helper function to process a single array.
    def process_array(arr):
        return reverse_vectorize_data(arr, ordered_block_names, latent_array, latent2block_name,
                                      block_name2id, block_id2color, stop_at_block_id, latent_dim)
    
    # If input_data is a string, assume a file or folder.
    if isinstance(input_data, str):
        if os.path.isdir(input_data):
            # Process every .npy file in the folder.
            for fname in os.listdir(input_data):
                if fname.endswith(".npy"):
                    in_path = os.path.join(input_data, fname)
                    arr = np.load(in_path, allow_pickle=True)
                    result = process_array(arr)
                    outputs[fname] = result
                    if output:
                        out_dir = output
                        os.makedirs(out_dir, exist_ok=True)
                        out_fname = os.path.splitext(fname)[0] + "_reconstructed.npy"
                        np.save(os.path.join(out_dir, out_fname), result)
            return outputs
        elif os.path.isfile(input_data):
            arr = np.load(input_data, allow_pickle=True)
            result = process_array(arr)
            if output:
                # If output is a filename, save result there.
                np.save(output, result)
            return result
        else:
            raise ValueError("Input path is not a valid file or directory.")
    elif isinstance(input_data, np.ndarray):
        # Process a single loaded array.
        return process_array(input_data)
    elif isinstance(input_data, (list, tuple)):
        # Process each array in the list.
        return [process_array(arr) for arr in input_data]
    else:
        raise ValueError("Unsupported type for input_data.")

# If run as a script, use argparse to process a file or folder.
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Reverse vectorized structure(s) to block IDs or average colors."
    )
    parser.add_argument("--input_path", type=str, default="vectorized_structures",
                        help="Path to input .npy file or directory containing .npy files.")
    parser.add_argument("--output_path", type=str, default="reconstructed_output",
                        help="Path to output file (if input is file) or directory (if input is folder).")
    parser.add_argument("--block_name2latent", type=str, default="mappings/block_name2latent.json",
                        help="Path to block_name2latent.json mapping file.")
    parser.add_argument("--latent2block_name", type=str, default="mappings/latent2block_name.json",
                        help="Path to latent2block_name.json mapping file.")
    parser.add_argument("--block_name2id", type=str, default="mappings/block_name2id.json",
                        help="Path to block_name2id.json mapping file.")
    parser.add_argument("--block_id2color", type=str, default="mappings/block_id2color.json",
                        help="Path to block_id2color.json mapping file (ignored if --stop_at_block_id is set).")
    parser.add_argument("--stop_at_block_id", action="store_true",
                        help="If set, output will be block IDs instead of colors.")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimensionality of latent vectors (default: 32).")
    args = parser.parse_args()
    
    result = reverse_vectorize(
        input_data=args.input_path,
        output=args.output_path,
        block_name2latent_path=args.block_name2latent,
        latent2block_name_path=args.latent2block_name,
        block_name2id_path=args.block_name2id,
        block_id2color_path=args.block_id2color,
        stop_at_block_id=args.stop_at_block_id,
        latent_dim=args.latent_dim
    )
    # If running as script and processing a single file or loaded array, print a summary.
    if isinstance(result, np.ndarray):
        print(f"Processed single file. Output shape: {result.shape}")
    elif isinstance(result, dict):
        print(f"Processed {len(result)} files from folder '{args.input_path}'.")
    else:
        print("Processing complete.")

if __name__ == "__main__":
    main()
