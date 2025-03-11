import os
import argparse
import numpy as np
from file_io import load_npy_files, save_structure
from vectorizer import BlockVectorizer, build_unique_block_ids
from augmentation import augment_structure

def process_structure(structure, vectorizer):
    # Vectorize the structure.
    # We assume structure is a 3D numpy array.
    # Create an output array of shape (H, W, D, 32) where for each block:
    # if block==0 then output vector is zeros, else the 32-dim vector from vectorizer.
    H, W, D = structure.shape
    vectorized = np.zeros((H, W, D, vectorizer.dim), dtype=np.float32)
    
    # Process nonzero (non-empty) blocks.
    # For performance, we use numpy indexing.
    non_empty = structure != 0
    unique_blocks = np.unique(structure[non_empty])
    # Build a lookup mapping (for each block id, its 32-dim vector)
    embed_lookup = {block_id: vectorizer.transform(block_id) for block_id in unique_blocks}
    
    # Apply lookup vector for each nonzero entry.
    # (We loop over unique block ids; if many blocks share the same id, this is efficient.)
    for block_id, vec in embed_lookup.items():
        mask = structure == block_id
        vectorized[mask] = vec  # broadcast the 32-dim vector
    return vectorized

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all .npy files from the input directory.
    npy_files = load_npy_files(input_dir)
    
    # Build the global list of block ids by scanning all structures.
    unique_block_ids = build_unique_block_ids(npy_files)
    print(f"Found {len(unique_block_ids)} unique block ids.")
    
    # Build vectorizer using UMAP (and your chosen semantic/TFIDF features).
    vectorizer = BlockVectorizer(unique_block_ids, dim=32)
    
    for file_path, structure in npy_files.items():
        print(f"Processing {file_path}...")
        # Vectorize the structure (each block id becomes a 32-dim vector)
        vec_structure = process_structure(structure, vectorizer)
        
        # Augment the structure to get 13 variants.
        augmented_variants = augment_structure(vec_structure)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save each augmented variant.
        for i, aug in enumerate(augmented_variants):
            out_fname = f"{base_name}_aug{i}.npy"
            out_path = os.path.join(output_dir, out_fname)
            save_structure(aug, out_path)
            print(f"Saved augmented file: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and augment Minecraft structures.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save augmented .npy files")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
