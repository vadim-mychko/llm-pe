import logging
import llms
from utils.setup_logging import setup_logging
import abc
import jinja2

'''
BasePEModule is the abstract base class for the core module for our preference elicitation task. 
Declared attributes:
'''

class BasePEModule(abc.ABC):

    def __init__(self, config, dataloader):
        self.logger: logging.Logger = setup_logging(self.__class__.__name__, config)
        self.config = config
        self.interactions = [] 
        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath='./templates'))
        self.items = dataloader.get_data()

        llm_module = llms.LLM_CLASSES[self.config['llm']['llm_name']]
        self.llm = llm_module(config)
        self.recs = []



    '''
    Get the IDs of the top k recommended items
    '''
    def get_top_items(self, k=5):
        pass

    '''
    Generates a query based on the current utility values and the provided set of items.
    '''
    def get_query(self):
        return NotImplementedError("Abstract Base Class")
    
    '''
    Update the model's beliefs, etc based on the user's response
    '''
    def update_from_response(self,query, response):
        return NotImplementedError("Abstract Base Class")
    
    def reset(self):
        self.interactions = []
        self.recs = []

    def get_last_results(self):
        return NotImplementedError