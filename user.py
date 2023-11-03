'''
The User class represents a user. 

They have the following fields: 
_id - unique int id
top_items - List of Item ids indicating the user's top few items
user_desc - String with a natural language description of the user
'''

class User:
    def __init__(self, id, top_items, user_desc):
        self._id = id
        self.top_items = top_items
        self.user_desc = user_desc

    def get_response(self, query):
        print(query)
        response = input("Enter Response: ")