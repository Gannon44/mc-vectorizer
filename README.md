# MC Vectorizer
🚀 A Python library for encoding and decoding Minecraft structures using diffusion-ready vector representations. The current implementation handles a custom .npy structure, but I plan to make it more generalizable in the future.

## Features
✅ **Block Name Vectorization** – Converts block names to 32-dimensional latent vectors.  
✅ **Efficient Reverse Mapping** – Uses KDTree for fast latent-to-block lookups.  
✅ **Structure Encoding** – Converts Minecraft `.npy` structures into trainable representations.  
✅ **Structure Decoding** – Recovers block IDs or average colors from latent vectors.  
✅ **Command-Line & API Support** – Use as a CLI tool or integrate into your Python code.

## Overview:

The `block_vectorization` module provides a (very incomplete) framework for converting Minecraft structure data between two formats:
1. **Vectorization:**  
   Convert a 3D structure (where each voxel contains either a block ID string or 0) into a 4D latent representation. Each block is represented by a 32‑channel vector built from:
   - Order-based (positional) encoding,
   - Semantic embeddings (from a SentenceTransformer, reduced via UMAP),
   - TF-IDF embeddings (reduced via UMAP).

2. **Reverse–Vectorization:**  
   Convert the 32‑channel latent representation back into a human–readable format by:
   - Using a KDTree (built from the latent mappings) to find the closest latent vector,
   - Mapping that latent vector to a block name,
   - Converting the block name to a block ID (using a provided mapping), and
   - Optionally mapping the block ID to its average color.

The module is designed for both programmatic use (via single function calls) and via the command–line interface.

## Module Components:

1. **BlockNameVectorizer Class:**  
   - **Purpose:** Build and store a mapping from block names to 32-dimensional latent vectors.
   - **Key Methods:**
     - `transform(block_name)`: Returns the latent vector for a given block name.
     - `reverse(vector)`: Returns the closest block name for a given latent vector using an internal KDTree.
   - **Mapping Persistence:**  
     The vectorizer can save its mappings to JSON files (`block_name2latent.json` and `latent2block_name.json`) and load them from a specified directory. Use `get_block_name_vectorizer()` to obtain an instance, optionally forcing recreation from a CSV file.

2. **Vectorization Functions:**
   - **`vectorize_structure(input_data, block_vectorizer, block_id2name, latent_dim=32, output_path=None)`**
     - **Input:** A 3D structure provided as a file path, directory, single NumPy array, or list/tuple of arrays. Each voxel is either a block ID (string) or 0.
     - **Processing:**  
       For each voxel:
         - If the value is 0, it is interpreted as "Air."
         - Otherwise, the block ID is mapped to a block name using the `block_id2name` dictionary, then the vectorizer converts it to a latent vector.
     - **Output:** A 4D NumPy array with shape (H, W, D, 32) or a set of such arrays saved to disk if `output_path` is provided.
     
3. **Reverse–Vectorization Functions:**
   - **`reverse_vectorize_structure(input_data, block_vectorizer, block_name2id, block_id2color=None, stop_at_block_id=False, latent_dim=32, output_path=None)`**
     - **Input:** A 4D latent vector structure (or collection thereof) provided as a file path, directory, or already loaded array(s).
     - **Processing:**  
       For each latent vector:
         - The KDTree in `block_vectorizer` finds the closest latent vector.
         - This latent vector is mapped to a block name.
         - The block name is converted to a block ID using `block_name2id`.
         - Optionally (if `stop_at_block_id` is False) the block ID is further mapped to an average color using `block_id2color`.
     - **Output:** A structure containing either block IDs (if `stop_at_block_id` is True) or average color values.
     
4. **Command–Line Interface (CLI):**
   - The module supports two subcommands:
     - **`vectorize`:**  
       Convert block ID structures to latent vectors.
       - **Arguments:**
         - `--input_path`: Input file or folder containing .npy files with block IDs.
         - `--output_path`: Where to save the resulting .npy file(s).
         - `--block_id2name`: Path to a JSON file mapping block IDs to block names.
         - `--vectorizer_csv`: CSV file (no header) containing the ordered list of block names.
         - `--vectorizer_dir`: Directory to load or save vectorizer mappings.
         - `--recreate_vectorizer`: (Flag) Force recreation of vectorizer mappings from CSV.
         - `--latent_dim`: Latent vector dimensionality (default 32).
     - **`reverse`:**  
       Convert latent vector structures back to block IDs or average colors.
       - **Arguments:**
         - `--input_path`: Input file or folder containing latent vector .npy files.
         - `--output_path`: Output file or folder.
         - `--block_name2id`: Path to JSON mapping from block names to block IDs.
         - `--block_id2color`: Path to JSON mapping from block IDs to average colors.
         - `--stop_at_block_id`: (Flag) If set, the output will be block IDs; otherwise, average colors.
         - `--vectorizer_csv`, `--vectorizer_dir`, `--recreate_vectorizer`, `--latent_dim`: As above.
         
## Programmatic API Usage:

To use the module within your Python code, import the functions and classes:

```python
from block_vectorization import get_block_name_vectorizer, vectorize_structure, reverse_vectorize_structure

# Obtain a BlockNameVectorizer (load precomputed mappings if available)
vectorizer = get_block_name_vectorizer(csv_path="mappings/ordered_block_names.csv",
                                       precomputed_dir="mappings", latent_dim=32)

# To vectorize a single loaded structure (NumPy array):
with open("input_structure.npy", "rb") as f:
    structure = np.load(f, allow_pickle=True)
# block_id2name is a dict mapping block IDs to block names.
with open("mappings/block_id2name.json", "r") as f:
    block_id2name = json.load(f)
vectorized = vectorize_structure(structure, vectorizer, block_id2name, latent_dim=32)

# To reverse–vectorize back to block IDs:
with open("mappings/block_name2id.json", "r") as f:
    block_name2id = json.load(f)
# Optionally, load block_id2color mapping for color conversion.
with open("mappings/block_id2color.json", "r") as f:
    block_id2color = json.load(f)
reversed_data = reverse_vectorize_structure(vectorized, vectorizer, block_name2id,
                                              block_id2color=block_id2color, stop_at_block_id=False, latent_dim=32)
```

## Command–Line Usage:

From the terminal, you can run the module using:

- **Vectorization:**
  ```bash
  python block_vectorization.py vectorize \
      --input_path input_structures \
      --output_path vectorized_structures \
      --block_id2name mappings/block_id2name.json \
      --vectorizer_csv mappings/ordered_block_names.csv \
      --vectorizer_dir mappings \
      --latent_dim 32
  ```
  
- **Reverse–Vectorization:**
  ```bash
  python block_vectorization.py reverse \
      --input_path vectorized_structures \
      --output_path reconstructed_output \
      --block_name2id mappings/block_name2id.json \
      --block_id2color mappings/block_id2color.json \
      --vectorizer_csv mappings/ordered_block_names.csv \
      --vectorizer_dir mappings \
      --latent_dim 32
  ```
  To output block IDs rather than colors, add the flag `--stop_at_block_id`.
