from item import Item
from user import User
import logging
import llms
from utils.logging import setup_logging
import abc

'''
BasePEModule is the abstract base class for the core module for our preference elicitation task. 
'''

class BasePEModule(abc.ABC):

    def __init__(self, config, debug=False):
        self.logger: logging.Logger = setup_logging(self.__class__.__name__, config)
        self.config = config
        self.debug=debug

        llm_module = llms.LLM_CLASSES[self.config['llm']['llm_name']]
        self.llm = llm_module(config)

    def get_top_items(self, k=5):
        pass

    '''
    Generates a query based on the current utility values and the provided set of items.
    '''
    def query_selection(self):
        return NotImplementedError("Abstract Base Class")

    '''
    Updates the belief state and user profile based on the user's response
    '''
    def belief_update(self, query, response): 
        return NotImplementedError("Abstract Base Class")


    def pe_loop(self):
        return NotImplementedError("Abstract Base Class")