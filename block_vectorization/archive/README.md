## **📂 Archive Folder**  

This folder contains **older scripts and exploratory work** related to Minecraft block vectorization and dataset processing. These files are **not actively used in the main package**, but they are kept here for reference in case they are needed later.  

### **📜 Contents**  

- **`block_vectorization.ipynb`**  
  - Initial **exploration notebook** for Minecraft block vectorization.  
  - Contains early experiments with different vectorization techniques.  

- **`process_structures.py`**  
  - An early attempt at **processing and augmenting the dataset**.  
  - **Relies on an outdated `vectorization.py`** that no longer works with the current pipeline.  
  - Kept for reference on dataset augmentation strategies.  

- **`augmentation.py`**  
  - Another early script for **augmenting dataset structures** (flips, rotations, etc.).  
  - Also depends on an older version of `vectorization.py`.  
  - May be useful for future dataset augmentation needs.  

- **`file_io.py`**  
  - **Scans a directory** for `.npy` files and loads them.  
  - Returns a **dictionary mapping file paths to NumPy arrays**.  
  - Can be useful for bulk file loading.  

- **`npy_reader.py`**  
  - **Reads a `.npy` file and outputs a `.txt` file** for convenient viewing.  
  - Helpful for quickly inspecting dataset contents in a human-readable format.  

---

### **🛑 Notes**  
- These scripts **are not actively maintained** and may not work with the latest codebase.  
- If integrating any of these files, **check for compatibility issues** with newer versions of `vectorization.py`.  
- If any of these functionalities are needed in the future, consider **rewriting them** to align with the new structure.  

