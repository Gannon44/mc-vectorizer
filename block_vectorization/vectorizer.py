import os
import json
import argparse
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import KDTree
from sentence_transformers import SentenceTransformer

class BlockNameVectorizer:
    def __init__(self, csv_path=None, latent_dim=32, precomputed_dir=None):
        """
        Builds a vectorizer for block names using features:
          1. 1 channel of order-based embedding (normalized index)
          2. 15 channels of semantic embeddings (using a SentenceTransformer, reduced via UMAP)
          3. 16 channels of TF-IDF embeddings (reduced via UMAP)
        
        The final latent vector is 32-dimensional.
        
        If precomputed_dir is provided and contains both mapping files 
        ("block_name2latent.json" and "latent2block_name.json"), the vectorizer is built 
        from those mappings.
        
        Args:
          csv_path: Path to a CSV file containing an ordered list of block names (no header).
          latent_dim: Target dimension (should be 32).
          precomputed_dir: Directory containing precomputed mapping files.
        """
        self.latent_dim = latent_dim
        self.block_to_latent = {}
        if precomputed_dir:
            bn2l_path = os.path.join(precomputed_dir, "block_name2latent.json")
            l2bn_path = os.path.join(precomputed_dir, "latent2block_name.json")
            if os.path.exists(bn2l_path) and os.path.exists(l2bn_path):
                with open(bn2l_path, "r") as f:
                    mapping = json.load(f)
                # Convert list representations to numpy arrays.
                self.block_to_latent = {name: np.array(vec, dtype=np.float32) for name, vec in mapping.items()}
                self.block_names = list(self.block_to_latent.keys())
                self._build_kdtree()
                return  # Done loading precomputed mappings.
        
        # Otherwise, generate from CSV.
        if csv_path is None:
            raise ValueError("csv_path must be provided if precomputed mappings are not used.")
        df = pd.read_csv(csv_path, header=None)  # no headers
        self.block_names = df[0].tolist()  # assume the first (only) column contains block names
        self.block_to_latent = self._build_embeddings(self.block_names)
        self._build_kdtree()
    
    def _build_embeddings(self, block_names):
        n = len(block_names)
        # 1. Positional encoding: normalized index (1 channel)
        order_embeddings = np.array([[i / n] for i in range(n)])  # shape (n, 1)
        
        # 2. LLM embeddings: encode using SentenceTransformer then reduce to 15 dims via UMAP
        model = SentenceTransformer('all-MiniLM-L6-v2')
        semantic_raw = model.encode(block_names, convert_to_numpy=True, show_progress_bar=True)
        umap_sem = umap.UMAP(n_components=15, random_state=42)
        semantic_embeddings = umap_sem.fit_transform(semantic_raw)  # shape (n, 15)
        
        # 3. TF-IDF embeddings: vectorize and reduce to 16 dims via UMAP
        tfidf = TfidfVectorizer()
        tfidf_raw = tfidf.fit_transform(block_names).toarray()
        umap_tfidf = umap.UMAP(n_components=16, random_state=42)
        tfidf_embeddings = umap_tfidf.fit_transform(tfidf_raw)  # shape (n, 16)
        
        # Concatenate: 1 + 15 + 16 = 32 channels
        combined_features = np.hstack([order_embeddings, semantic_embeddings, tfidf_embeddings])
        mapping = {name: combined_features[i] for i, name in enumerate(block_names)}
        return mapping

    def transform(self, block_name):
        """
        Returns the latent vector for the given block name.
        If not found, returns a zero vector.
        """
        return self.block_to_latent.get(block_name, np.zeros(self.latent_dim, dtype=np.float32))
    
    def _build_kdtree(self):
        # Build an array of latent vectors in the same order as the block names.
        self._latent_array = np.array(list(self.block_to_latent.values()))
        self._ordered_block_names = list(self.block_to_latent.keys())
        self.kdtree = KDTree(self._latent_array)

    def reverse(self, vector):
        """
        Given a latent vector, efficiently finds the closest block name using a KDTree.
        """
        distance, idx = self.kdtree.query(vector.reshape(1, -1), k=1)
        return self._ordered_block_names[idx[0]]


def save_mappings(vectorizer, output_dir):
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
    
    If precomputed_dir is provided and mapping files exist and recreate is False,
    the vectorizer will be loaded from those mappings. Otherwise, it will be generated
    from the CSV file.
    
    Parameters:
      csv_path: Path to CSV file with ordered block names (no header). Required if precomputed_dir is not used.
      latent_dim: Dimensionality of the latent vectors (default: 32).
      precomputed_dir: Directory where mapping files are stored.
      recreate: If True, force recreation of the mappings from the CSV file.
      prompt_on_overwrite: If True and recreate is True, prompt the user before overwriting existing mappings.
    
    Returns:
      An instance of BlockNameVectorizer.
    """
    if recreate and precomputed_dir:
        bn2l_path = os.path.join(precomputed_dir, "block_name2latent.json")
        l2bn_path = os.path.join(precomputed_dir, "latent2block_name.json")
        if os.path.exists(bn2l_path) or os.path.exists(l2bn_path):
            if prompt_on_overwrite:
                response = input("WARNING: Mapping files already exist and will be overwritten. "
                                 "Are you sure you want to recreate them? (y/n): ")
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


# --- Command-Line Interface ---
def main():
    parser = argparse.ArgumentParser(description="Build or load block name vectorizer mappings.")
    parser.add_argument("--csv_path", type=str, default="mappings/ordered_block_names.csv",
                        help="Path to CSV file with ordered block names (no header).")
    parser.add_argument("--output_dir", type=str, default="mappings",
                        help="Directory to save/load mapping files.")
    parser.add_argument("--recreate", action="store_true",
                        help="Force recreation of vectorizer mappings from CSV, overwriting existing mapping files.")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimensionality of the latent vectors (default: 32).")
    args = parser.parse_args()
    
    vectorizer = get_block_name_vectorizer(csv_path=args.csv_path,
                                           latent_dim=args.latent_dim,
                                           precomputed_dir=args.output_dir,
                                           recreate=args.recreate,
                                           prompt_on_overwrite=True)
    print("Vectorizer ready.")
    
if __name__ == '__main__':
    main()
