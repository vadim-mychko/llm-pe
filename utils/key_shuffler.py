import json
import random

class KeyShuffler():

    def __init__(self, file_name):
        with open(file_name) as json_data:
            file_data = json.load(json_data)
        self.keys = file_data['keys']

    def get_next_key(self):
        # Doing random so we can reduce collisions with concurrent runs - don't want to create a more sophisticated algo
        return random.choice(self.keys)