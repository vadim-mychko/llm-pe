import json
import re

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
        # return {clean_movie_name(key).lower(): convert_dict_to_lowercase(value) for key, value in data.items()} # Use for movie dataset
    elif isinstance(data, list):
        return [convert_dict_to_lowercase(item) for item in data]
    elif isinstance(data, str):
        return data.lower()
    else:
        return data
    
def clean_movie_name(movie_name):
        # Define the regex pattern to match ", The" before the year in brackets
        pattern = r', the(?=\s*\(\d{4}\))'
        # Replace the pattern with an empty string
        cleaned_name = re.sub(pattern, '', movie_name)
        # print("Cleaned name: ", cleaned_name)
        return cleaned_name

# Example usage
input_file = 'data/name_maps/ml25M_100_map.json'
output_file = 'data/name_maps/ml25M_100_map_no_the.json'
json_to_lowercase(input_file, output_file)