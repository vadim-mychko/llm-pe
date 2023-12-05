from item import Item
from users.user import User
import math
from pe_modules.base_pe_module import BasePEModule
import jinja2

'''
JointPEModule is a child class of PEModule, the core module for our preference elicitation task.
JointPEModule models the utility conditioned on all responses.

items - A list of all Items
util - List with utility value beliefs for each item
responses - List of user query/response strings
user_profile - string with the natural language user profile -> Possibly removable?
'''

class JointPEModule(BasePEModule):

    def __init__(self, config, dataloader):
        super().__init__(config)
        self.items = dataloader.get_data()
        self.user_profile = ""
        self.util = [0.5] * len(dataloader) # Until we figure out belief update, we'll just use a utility value in [0,1]
        self.responses = []
        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath='./templates'))

    '''
    Gets the top k items with UCB
    '''
    def ucb_get_items(self, k=3):
        pass

    '''
    Return the items with the top k utility means. Just for ease of implementation
    '''
    def get_top_items(self, k=3):
        top_k_idx = sorted(range(len(self.util)), key=lambda i: self.util[i])[-k:]
        top_k_items = []
        for idx in top_k_idx:
            top_k_items.append(self.items[idx])
        return top_k_items

    '''
    Get the LLM to generate a query for the user based on the current utility values.
    '''
    def query_selection(self):
        # Get the top 3 items and generate the query to differentiate between them
        #query_items = self.ucb_get_items() #implement later, for now pick top 3 items by utility mean
        query_items = self.get_top_items()
        
        template_file = self.config['query']['query_selection_template_file']
        query_template = self.jinja_env.get_template(template_file)
        context = {
            "items": query_items
        }
        query = query_template.render(context)

        response = self.llm.make_request(query, temperature=self.config['llm']['temperature'])
        return response
    
    '''
    Update the belief state at self.utils[item_idx] given the T/F probabilities.
    '''
    def update_from_probs(self, item_idx, true_prob, false_prob):
        # TODO: For now we'll do some very sketchy hack that just changes the mean
        momentum = 0.3
        
        # This is super hacky, we'll fix it -> Try update fully every time
        # TODO: Look at log probs for each item given interactions w 1- / 2 values
        new_util = ((1-momentum) * self.util[item_idx]) + momentum * (true_prob / (true_prob + false_prob)) 
        self.util[item_idx] = new_util
        return
        
    
    '''
    Queries the LLM on if a user would like the item given the response history. Returns a dict with the log probs of true/false
    that the user will like the item based on item_reviews given the response history in self.responses
    '''
    def get_item_pref(self, item_reviews):
        # Create query from jinja template
        pref_query_file = self.config['llm']['pref_template_file']
        query_template = self.jinja_env.get_template(pref_query_file)
        context = {
            "interactions": self.responses,
            "reviews": item_reviews
        }
        final_query = query_template.render(context)

        # We want more than just the top logprob so set logprobs to something higher than 0
        response = self.llm.make_request(final_query, logprobs=10)
        log_probs = self.llm.get_log_probabilities()

        # Extract T/F logprobs 
        true_prob = 0.0
        false_prob = 0.0

        #TODO: Add an option to turn off stripping and lowercasing.

        for token_pos in log_probs: # Iterate over each token value for each token position
            for token_val in token_pos.keys():
                trimmed_token_val = token_val.strip().lower()
                 # If token is 'true' or 'false' after lowercasing and stripping, add to corresponding probability total
                if trimmed_token_val == "true":
                    token_log_prob = token_pos[token_val]
                    token_prob = math.exp(token_log_prob)
                    true_prob += token_prob
                if trimmed_token_val == "false":
                    token_log_prob = token_pos[token_val]
                    token_prob = math.exp(token_log_prob)
                    false_prob += token_prob 

        # TODO: Use Mustafa's update method -> Confirm how this would make sense in our context
        # true_prob = (1 + true_prob) / 2
        # false_prob = (1 - false_prob) / 2
            
        return {'true': true_prob, 'false': false_prob}

    '''
    Updates the belief state for all items based on all of the user's past response
    '''
    def belief_update(self): 
        for item_idx, item in enumerate(self.items):
            # Get logprobs of user liking item based on reviews and user's interactions
            probs = self.get_item_pref(item['reviews'])
            # Perform update
            self.update_from_probs(item_idx, probs['true'], probs['false'])
        return

    '''
    pe_loop is the core functionality of the preference elicitation module. It repeatedly selects a query,
    gets the user's response, then updates beliefs. Currently, it is set to run a fixed number of times before printing
    the top 3 items
    '''
    def pe_loop(self):
        for i in range(3):
            query = self.query_selection()
            print(query)
            response = input("Your response: ")
            self.responses.append({'query': query, 'response': response}) 
            self.belief_update() 
            debug_str = "Utilities at turn %s: %s" % (i, str(self.util))
            self.logger.debug(debug_str) # Log all utilities to debugger - works for small datasets


        top_items = self.get_top_items(3)
        print("Top ranked items:")
        for i, item in enumerate(top_items):
            print(i, item['name'])

        return top_items
