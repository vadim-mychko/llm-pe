from item import Item
from user import User

'''
PrefLoop is preference elicitation system. 
items - A list of all Items
user_profile - string with the natural language user profile
util - List with utility value beliefs for each item
responses - List of user query/response strings
'''

class PrefLoop:

    def __init__(self, items, user_profile="", util, responses):
        self.items = items
        self.user_profile = user_profile
        self.util = util
        self.responses = responses

    '''
    Gets the top k items with UCB
    '''
    def ucb_get_items(k=3):
        pass

    '''
    Generates a query based on the current utility values.
    '''
    def get_query():
        pass

    '''
    Updates the belief state and user profile based on the user's response
    '''
    def update_belief(): 
        pass

    '''
    Runs the full recommendation loop until a recommendation is made.
    '''
    def rec_loop(u):
        pass
    
