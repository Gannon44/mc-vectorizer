import numpy as np
import sys

def print_npy_file(file_path):
    try:
        # Load the .npy file with allow_pickle=True to handle pickled data
        data = np.load(file_path, allow_pickle=True)
        
        # Check if the data is 2D or not, and if so, remove rows that are all zeros
        if data.ndim == 2:
            data = data[~np.all(data == 0, axis=1)]  # Remove rows that are all zeros
        
        # Ensure that the entire array is printed without truncation
        np.set_printoptions(threshold=np.inf, edgeitems=10)  # Modify print options to avoid truncation
        
        # Print the content to the console
        print("Data loaded from", file_path)
        print(data)
        write_file = open("output.txt", "w")
        write_file.write(str(data))
        write_file.close()
    
    except Exception as e:
        print(f"Error loading .npy file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_npy.py <path_to_npy_file>")
    else:
        file_path = sys.argv[1]
        print_npy_file(file_path)
