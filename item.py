'''
The Item class represents an item. 

It has the following fields: 
_id - unique int id
reviews - List of Strins with a natural language description of the item
'''

class Item:
    def __init__(self, id, reviews):
        self._id = id
        self.reviews = reviews
