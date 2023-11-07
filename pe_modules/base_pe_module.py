from item import Item
from user import User
import logging
from utils.logging import setup_logging
import abc

'''
BasePEModule is the abstract base class for the core module for our preference elicitation task. 
'''

class BasePEModule(abc.ABC):

    def __init__(self, config):
        self.logger: logging.Logger = setup_logging(self.__class__.__name__)
        self.config = config

    def get_top_items(self, k=5):
        pass

    '''
    Generates a query based on the current utility values and the provided set of items.
    '''
    def query_selection(self):
        return "Abstract base class"

    '''
    Updates the belief state and user profile based on the user's response
    '''
    def belief_update(self, query, response): 
        return

