from item import Item
from user import User
from pe_modules.base_pe_module import BasePEModule

'''
JointPEModule is a child class of PEModule, the core module for our preference elicitation task.
JointPEModule models the utility conditioned on all responses.

items - A list of all Items
util - List with utility value beliefs for each item
responses - List of user query/response strings
user_profile - string with the natural language user profile
'''

class JointPEModule(BasePEModule):

    def __init__(self, config, dataloader):
        super.__init__(config)
        self.items = dataloader
        self.user_profile = ""
        self.util = [0.5] * len(dataloader)
        self.responses = []

    '''
    Gets the top k items with UCB
    '''
    def ucb_get_items(self, k=3):
        pass

    '''
    Generates a query based on the current utility values.
    '''
    def query_selection(self):
        pass

    '''
    Updates the belief state and user profile based on the user's response
    '''
    def belief_update(self, query, response): 
        pass
