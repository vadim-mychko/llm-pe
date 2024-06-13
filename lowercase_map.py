import json

# Function to convert JSON content to lowercase
def json_to_lowercase(input_file, output_file):
    with open(input_file, 'r') as f:
        json_data = json.load(f)

    # Convert all keys and values to lowercase
    lowercase_json_data = convert_dict_to_lowercase(json_data)

    with open(output_file, 'w') as f:
        json.dump(lowercase_json_data, f, indent=4)

# Function to recursively convert dictionary keys and values to lowercase
def convert_dict_to_lowercase(data):
    if isinstance(data, dict):
        return {key.lower(): convert_dict_to_lowercase(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_dict_to_lowercase(item) for item in data]
    elif isinstance(data, str):
        return data.lower()
    else:
        return data

# Example usage
input_file = 'data/name_maps/ml25M_100_map.json'
output_file = 'data/name_maps/ml25M_100_map_new.json'
json_to_lowercase(input_file, output_file)