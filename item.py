'''
The Item class represents an item. 

It has the following fields: 
_id - unique int id
desc - String with a natural language description of the item
'''

class Item:
    def __init__(self, id, desc):
        self._id = id
        self.desc = desc
