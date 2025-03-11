import os
import json
import argparse
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import KDTree
from sentence_transformers import SentenceTransformer

# =============================================================================
# BLOCK NAME VECTORIZER CLASS & SUPPORT FUNCTIONS
# =============================================================================

class BlockNameVectorizer:
    """
    The BlockNameVectorizer builds a 32-dimensional latent representation for block names
    using three feature sets:
      1. A 1-channel order-based (positional) encoding (normalized index).
      2. A 15-channel semantic embedding using SentenceTransformer (reduced via UMAP).
      3. A 16-channel TF-IDF embedding (reduced via UMAP).

    The class also builds a KDTree for efficient reverse lookup (latent vector -> block name).
    
    Mappings can be precomputed and stored in JSON files, which can then be reloaded.
    """
    def __init__(self, csv_path=None, latent_dim=32, precomputed_dir=None):
        self.latent_dim = latent_dim
        self.block_to_latent = {}
        # If precomputed mappings exist, load them.
        if precomputed_dir:
            bn2l_path = os.path.join(precomputed_dir, "block_name2latent.json")
            l2bn_path = os.path.join(precomputed_dir, "latent2block_name.json")
            if os.path.exists(bn2l_path) and os.path.exists(l2bn_path):
                with open(bn2l_path, "r") as f:
                    mapping = json.load(f)
                # Convert stored lists back to numpy arrays.
                self.block_to_latent = {name: np.array(vec, dtype=np.float32) for name, vec in mapping.items()}
                self.block_names = list(self.block_to_latent.keys())
                self._build_kdtree()
                return
        # Otherwise, generate from CSV.
        if csv_path is None:
            raise ValueError("csv_path must be provided if precomputed mappings are not used.")
        df = pd.read_csv(csv_path, header=None)  # CSV without header; first column contains block names.
        self.block_names = df[0].tolist()
        self.block_to_latent = self._build_embeddings(self.block_names)
        self._build_kdtree()

    def _build_embeddings(self, block_names):
        n = len(block_names)
        # 1. Order-based (positional) encoding: normalized index (1 channel)
        order_embeddings = np.array([[i / n] for i in range(n)])  # shape (n, 1)
        
        # 2. Semantic embeddings: use SentenceTransformer then reduce to 15 dimensions via UMAP
        model = SentenceTransformer('all-MiniLM-L6-v2')
        semantic_raw = model.encode(block_names, convert_to_numpy=True, show_progress_bar=True)
        umap_sem = umap.UMAP(n_components=15, random_state=42)
        semantic_embeddings = umap_sem.fit_transform(semantic_raw)  # shape (n, 15)
        
        # 3. TF-IDF embeddings: vectorize block names then reduce to 16 dimensions via UMAP
        tfidf = TfidfVectorizer()
        tfidf_raw = tfidf.fit_transform(block_names).toarray()
        umap_tfidf = umap.UMAP(n_components=16, random_state=42)
        tfidf_embeddings = umap_tfidf.fit_transform(tfidf_raw)  # shape (n, 16)
        
        # Concatenate all features: 1 + 15 + 16 = 32 channels.
        combined_features = np.hstack([order_embeddings, semantic_embeddings, tfidf_embeddings])
        mapping = {name: combined_features[i] for i, name in enumerate(block_names)}
        return mapping

    def transform(self, block_name):
        """
        Returns the 32-dimensional latent vector for the given block name.
        If the block name is not found, returns a zero vector.
        """
        return self.block_to_latent.get(block_name, np.zeros(self.latent_dim, dtype=np.float32))
    
    def _build_kdtree(self):
        """
        Builds a KDTree for efficient reverse lookup (latent vector -> block name).
        """
        self._latent_array = np.array(list(self.block_to_latent.values()))
        self._ordered_block_names = list(self.block_to_latent.keys())
        self.kdtree = KDTree(self._latent_array)

    def reverse(self, vector):
        """
        Given a 32-dimensional latent vector, returns the closest block name by querying the KDTree.
        """
        distance, idx = self.kdtree.query(vector.reshape(1, -1), k=1)
        return self._ordered_block_names[idx[0]]


def save_mappings(vectorizer, output_dir):
    """
    Saves the block name to latent vector mapping and the reverse mapping to JSON files.
    Files saved:
      - block_name2latent.json
      - latent2block_name.json
    """
    os.makedirs(output_dir, exist_ok=True)
    block_name2latent = {name: emb.tolist() for name, emb in vectorizer.block_to_latent.items()}
    latent2block_name = {str(emb.tolist()): name for name, emb in vectorizer.block_to_latent.items()}
    
    bn2l_path = os.path.join(output_dir, "block_name2latent.json")
    l2bn_path = os.path.join(output_dir, "latent2block_name.json")
    with open(bn2l_path, "w") as f:
        json.dump(block_name2latent, f, indent=2)
    with open(l2bn_path, "w") as f:
        json.dump(latent2block_name, f, indent=2)
    print("Vectorizer built and mappings saved.")
    print(f"Block names -> latent vectors saved to '{bn2l_path}'.")
    print(f"Latent vectors -> block names saved to '{l2bn_path}'.")


def get_block_name_vectorizer(csv_path=None, latent_dim=32, precomputed_dir=None, recreate=False, prompt_on_overwrite=True):
    """
    Returns an instance of BlockNameVectorizer.
    
    If precomputed_dir is provided and mapping files exist (and recreate is False), the vectorizer is loaded from them.
    Otherwise, the vectorizer is generated from the CSV file.
    
    Parameters:
      csv_path: Path to CSV file containing block names (no header). Required if precomputed mappings are not used.
      latent_dim: Dimensionality of the latent vectors (default: 32).
      precomputed_dir: Directory where mapping files are stored.
      recreate: If True, force recreation of mappings from the CSV, overwriting existing files.
      prompt_on_overwrite: If True and recreation is forced, prompt the user for confirmation.
      
    Returns:
      An instance of BlockNameVectorizer.
    """
    if recreate and precomputed_dir:
        bn2l_path = os.path.join(precomputed_dir, "block_name2latent.json")
        l2bn_path = os.path.join(precomputed_dir, "latent2block_name.json")
        if os.path.exists(bn2l_path) or os.path.exists(l2bn_path):
            if prompt_on_overwrite:
                response = input("WARNING: Mapping files already exist and will be overwritten. Are you sure? (y/n): ")
                if response.lower() != "y":
                    print("Aborting recreation. Loading existing mappings instead.")
                    recreate = False
    if not recreate and precomputed_dir:
        bn2l_path = os.path.join(precomputed_dir, "block_name2latent.json")
        l2bn_path = os.path.join(precomputed_dir, "latent2block_name.json")
        if os.path.exists(bn2l_path) and os.path.exists(l2bn_path):
            print("Loading precomputed vectorizer from mapping files...")
            return BlockNameVectorizer(precomputed_dir=precomputed_dir, latent_dim=latent_dim)
    # Otherwise, generate new mappings from the CSV.
    if csv_path is None:
        raise ValueError("csv_path must be provided if precomputed mappings are not used or if recreating.")
    print("Generating vectorizer from CSV...")
    vectorizer = BlockNameVectorizer(csv_path=csv_path, latent_dim=latent_dim)
    if precomputed_dir:
        save_mappings(vectorizer, precomputed_dir)
    return vectorizer

# =============================================================================
# STRUCTURE VECTORIZATION FUNCTIONS
# =============================================================================

def load_npy_files(input_dir):
    """
    Loads all .npy files from the specified directory.
    
    Returns a dictionary mapping file paths to NumPy arrays.
    """
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

def vectorize_structure(input_data, block_vectorizer, block_id2name, latent_dim=32, output_path=None):
    """
    Converts a structure of block IDs into a 4D array (H, W, D, latent_dim) where each block is replaced by
    its 32-channel latent vector.

    The conversion process:
      - If a value is 0, it is interpreted as "Air".
      - Otherwise, the block ID is mapped (via block_id2name) to a block name, then converted via block_vectorizer.transform().

    Parameters:
      input_data: Either a file path (string), a directory path, a single NumPy array, or a list/tuple of arrays.
      block_vectorizer: An instance of BlockNameVectorizer.
      block_id2name: A dict mapping block IDs (strings) to block names.
      latent_dim: Dimensionality of the latent vectors (default: 32).
      output_path: (Optional) If provided and input_data is a file/directory, results will be saved to this path.
    
    Returns:
      The vectorized NumPy array, a list of arrays, or a dict mapping filenames to arrays (if processing files).
    """
    def process_array(arr):
        H, W, D = arr.shape
        output = np.zeros((H, W, D, latent_dim), dtype=np.float32)
        # For each unique value, determine its latent vector.
        unique_vals = set(arr.flatten())
        lookup = {}
        for val in unique_vals:
            if isinstance(val, (int, np.integer)):
                # Treat 0 as "Air"
                if val == 0:
                    lookup[val] = block_vectorizer.transform("Air")
                else:
                    print(f"Warning: Unexpected integer value '{val}' found in structure.")
                    lookup[val] = np.zeros(latent_dim, dtype=np.float32)
            elif isinstance(val, str):
                block_name = block_id2name.get(val, None)
                if block_name is None:
                    print(f"Warning: Block ID '{val}' not found in block_id2name mapping. Using zero vector.")
                    lookup[val] = np.zeros(latent_dim, dtype=np.float32)
                else:
                    lookup[val] = block_vectorizer.transform(block_name)
            else:
                print(f"Warning: Unsupported type '{type(val)}' encountered. Using zero vector.")
                lookup[val] = np.zeros(latent_dim, dtype=np.float32)
        for val, vec in lookup.items():
            mask = arr == val
            output[mask] = vec
        return output

    results = {}
    # If input_data is a string, treat as file or folder.
    if isinstance(input_data, str):
        if os.path.isdir(input_data):
            for fname in os.listdir(input_data):
                if fname.endswith(".npy"):
                    path = os.path.join(input_data, fname)
                    arr = np.load(path, allow_pickle=True)
                    result = process_array(arr)
                    results[fname] = result
                    if output_path:
                        out_dir = output_path
                        os.makedirs(out_dir, exist_ok=True)
                        out_fname = os.path.splitext(fname)[0] + "_vectorized.npy"
                        np.save(os.path.join(out_dir, out_fname), result)
            return results
        elif os.path.isfile(input_data):
            arr = np.load(input_data, allow_pickle=True)
            result = process_array(arr)
            if output_path:
                np.save(output_path, result)
            return result
        else:
            raise ValueError("Input path is not a valid file or directory.")
    elif isinstance(input_data, np.ndarray):
        return process_array(input_data)
    elif isinstance(input_data, (list, tuple)):
        return [process_array(arr) for arr in input_data]
    else:
        raise ValueError("Unsupported type for input_data.")

# =============================================================================
# STRUCTURE REVERSE–VECTORIZATION FUNCTIONS
# =============================================================================

def reverse_vectorize_structure(input_data, block_vectorizer, block_name2id, block_id2color=None,
                                stop_at_block_id=False, latent_dim=32, output_path=None):
    """
    Converts a latent vector structure (4D array: H x W x D x latent_dim) back into either block IDs
    or average color values.
    
    The conversion process:
      1. For each latent vector, the BlockNameVectorizer’s KDTree is used to find the nearest
         latent vector, which is then mapped to a block name.
      2. The block name is then mapped to a block ID via block_name2id.
      3. If stop_at_block_id is False, the block ID is further mapped to an average color using block_id2color.
    
    Parameters:
      input_data: Either a file/folder path (string), a NumPy array, or a list/tuple of arrays.
      block_vectorizer: An instance of BlockNameVectorizer.
      block_name2id: A dict mapping block names to block IDs.
      block_id2color: (Optional) A dict mapping block IDs to average colors. If None, only block IDs are returned.
      stop_at_block_id: If True, the output will be block IDs; otherwise, average colors.
      latent_dim: Dimensionality of the latent vectors.
      output_path: (Optional) If provided and input_data is a path, results will be saved.
    
    Returns:
      The reverse–vectorized data as a NumPy array, a list of arrays, or a dict mapping filenames to arrays.
    """
    def process_array(arr):
        if arr.shape[-1] != latent_dim:
            raise ValueError(f"Expected latent dimension {latent_dim}, but got {arr.shape[-1]}")
        H, W, D, _ = arr.shape
        arr_flat = arr.reshape(-1, latent_dim)
        distances, indices = block_vectorizer.kdtree.query(arr_flat, k=1)
        results = []
        for idx in indices:
            nearest_latent = block_vectorizer._latent_array[idx]
            key = str(nearest_latent.tolist())
            # Use the latent2block_name mapping (if available in block_vectorizer) or fallback.
            if hasattr(block_vectorizer, '_latent2block_name'):
                block_name = block_vectorizer._latent2block_name.get(key, block_vectorizer._ordered_block_names[idx])
            else:
                # Alternatively, use the reverse() method.
                block_name = block_vectorizer.reverse(nearest_latent)
            block_id = block_name2id.get(block_name, "unknown_block")
            if stop_at_block_id:
                results.append(block_id)
            else:
                # Map block id to color; default to black.
                color = block_id2color.get(block_id, [0, 0, 0, 0])
                results.append(color)
        if stop_at_block_id:
            return np.array(results, dtype=object).reshape(H, W, D)
        else:
            return np.array(results, dtype=np.float32).reshape(H, W, D, -1)
    
    outputs = {}
    if isinstance(input_data, str):
        if os.path.isdir(input_data):
            for fname in os.listdir(input_data):
                if fname.endswith(".npy"):
                    path = os.path.join(input_data, fname)
                    arr = np.load(path, allow_pickle=True)
                    result = process_array(arr)
                    outputs[fname] = result
                    if output_path:
                        out_dir = output_path
                        os.makedirs(out_dir, exist_ok=True)
                        out_fname = os.path.splitext(fname)[0] + "_reconstructed.npy"
                        np.save(os.path.join(out_dir, out_fname), result)
            return outputs
        elif os.path.isfile(input_data):
            arr = np.load(input_data, allow_pickle=True)
            result = process_array(arr)
            if output_path:
                np.save(output_path, result)
            return result
        else:
            raise ValueError("Input path is not a valid file or directory.")
    elif isinstance(input_data, np.ndarray):
        return process_array(input_data)
    elif isinstance(input_data, (list, tuple)):
        return [process_array(arr) for arr in input_data]
    else:
        raise ValueError("Unsupported type for input_data.")

# =============================================================================
# COMMAND-LINE INTERFACE (CLI)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Block Vectorization Module: vectorize structures (block IDs -> latent vectors) "
                    "and reverse-vectorize them (latent vectors -> block IDs or average colors)."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands: 'vectorize' or 'reverse'")
    
    # 'vectorize' subcommand
    vec_parser = subparsers.add_parser("vectorize", help="Convert block ID structures to latent vectors.")
    vec_parser.add_argument("--input_path", type=str, required=True,
                            help="Path to input .npy file or directory with structures (each voxel is a block ID or 0).")
    vec_parser.add_argument("--output_path", type=str, required=True,
                            help="Path to output .npy file (if single file) or directory (if folder).")
    vec_parser.add_argument("--block_id2name", type=str, default="mappings/block_id2name.json",
                            help="Path to block_id2name mapping file (maps block IDs to block names).")
    vec_parser.add_argument("--latent_dim", type=int, default=32,
                            help="Latent dimension (default: 32).")
    vec_parser.add_argument("--vectorizer_csv", type=str, default="mappings/ordered_block_names.csv",
                            help="CSV file (no header) with ordered block names for vectorizer generation.")
    vec_parser.add_argument("--vectorizer_dir", type=str, default="mappings",
                            help="Directory to load/save vectorizer mappings.")
    vec_parser.add_argument("--recreate_vectorizer", action="store_true",
                            help="Force recreation of vectorizer mappings from CSV.")
    
    # 'reverse' subcommand
    rev_parser = subparsers.add_parser("reverse", help="Convert latent vector structures back to block IDs or colors.")
    rev_parser.add_argument("--input_path", type=str, required=True,
                            help="Path to input .npy file or directory with latent vector structures.")
    rev_parser.add_argument("--output_path", type=str, required=True,
                            help="Path to output file (if single file) or directory (if folder).")
    rev_parser.add_argument("--block_name2id", type=str, default="mappings/block_name2id.json",
                            help="Path to block_name2id mapping file (maps block names to block IDs).")
    rev_parser.add_argument("--block_id2color", type=str, default="mappings/block_id2color.json",
                            help="Path to block_id2color mapping file (maps block IDs to average colors).")
    rev_parser.add_argument("--stop_at_block_id", action="store_true",
                            help="If set, output will be block IDs rather than colors.")
    rev_parser.add_argument("--latent_dim", type=int, default=32,
                            help="Latent dimension (default: 32).")
    rev_parser.add_argument("--vectorizer_csv", type=str, default="mappings/ordered_block_names.csv",
                            help="CSV file with ordered block names for vectorizer generation.")
    rev_parser.add_argument("--vectorizer_dir", type=str, default="mappings",
                            help="Directory to load/save vectorizer mappings.")
    rev_parser.add_argument("--recreate_vectorizer", action="store_true",
                            help="Force recreation of vectorizer mappings from CSV.")
    
    args = parser.parse_args()
    
    if args.command == "vectorize":
        # Load or generate the BlockNameVectorizer instance.
        vectorizer = get_block_name_vectorizer(csv_path=args.vectorizer_csv,
                                               latent_dim=args.latent_dim,
                                               precomputed_dir=args.vectorizer_dir,
                                               recreate=args.recreate_vectorizer)
        # Load block_id2name mapping.
        with open(args.block_id2name, "r") as f:
            block_id2name = json.load(f)
        result = vectorize_structure(args.input_path, vectorizer, block_id2name,
                                     latent_dim=args.latent_dim, output_path=args.output_path)
        print("Vectorization complete.")
    elif args.command == "reverse":
        vectorizer = get_block_name_vectorizer(csv_path=args.vectorizer_csv,
                                               latent_dim=args.latent_dim,
                                               precomputed_dir=args.vectorizer_dir,
                                               recreate=args.recreate_vectorizer)
        with open(args.block_name2id, "r") as f:
            block_name2id = json.load(f)
        block_id2color = None
        if not args.stop_at_block_id:
            with open(args.block_id2color, "r") as f:
                block_id2color = json.load(f)
        result = reverse_vectorize_structure(args.input_path, vectorizer, block_name2id,
                                             block_id2color=block_id2color,
                                             stop_at_block_id=args.stop_at_block_id,
                                             latent_dim=args.latent_dim,
                                             output_path=args.output_path)
        print("Reverse vectorization complete.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
