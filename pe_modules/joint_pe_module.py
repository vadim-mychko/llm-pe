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
        self.util = [[0.5, 0.2]] * len(dataloader) # default mu=0.5 sigma=0.2 Fix this later
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
        #query_items = self.ucb_get_items() #implement later, for now pick top 3 items by utility mean
        query_items = self.top_k_means()
        
        query = "Generate a yes or no query that best distinguishes between the following items: \n"
        for i, item in enumerate(query_items):
            item_str = "Item %d: " % i
            query += item_str
            query += item['desc']
            query += "\n"

        response = self.llm.make_request(query)
        return response

    '''
    Updates the belief state and user profile based on the user's response
    '''
    def belief_update(self): 


    def pe_loop(self):
        query = self.query_selection()
        print(query)
        response = input("Your response: ")
        self.responses.append({'query': query, 'response': response}) # maybe fix datatype later for scalability?
        self.belief_update() # TODO: Confirm which items were being updated with belief update

        pass
