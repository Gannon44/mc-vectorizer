import csv
import json
import sys

# Function to convert CSV to JSON
def csv_to_json(csv_file, json_file):
    # Initialize an empty dictionary to hold the data
    result = {}

    # Open the CSV file for reading
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        
        # Skip the header row
        next(reader)
        
        # Process each row in the CSV
        for row in reader:
            block_name = row[0]
            r, g, b, a = map(float, row[1:])
            result[block_name] = (r, g, b, a)
    
    # Write the resulting dictionary to a JSON file
    with open(json_file, mode='w') as file:
        json.dump(result, file, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python id2color.py <path_to_csv_file>")
    else:
        csv_file = sys.argv[1]
        csv_to_json(csv_file, "mappings/block_id2color.json")