from item import Item
from user import User
import math
from pe_modules.base_pe_module import BasePEModule
import jinja2

'''
JointPEModule is a child class of PEModule, the core module for our preference elicitation task.
JointPEModule models the utility conditioned on all responses.

items - A list of all Items
util - List with utility value beliefs for each item
responses - List of user query/response strings
user_profile - string with the natural language user profile
'''

class JointPEModule(BasePEModule):

    def __init__(self, config, dataloader, debug=False):
        #TODO: Move debug to config
        super().__init__(config, debug)
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
    def top_k_utils(self, k=3):
        top_k_idx = sorted(range(len(self.util)), key=lambda i: self.util[i])[-2:]
        top_k_items = []
        for idx in top_k_idx:
            top_k_items.append(self.items[idx])
        return top_k_items

    '''
    Generates a query based on the current utility values.
    '''
    def query_selection(self):
        #query_items = self.ucb_get_items() #implement later, for now pick top 3 items by utility mean
        query_items = self.top_k_utils()
        
        template_file = self.config['llm']['query_selection_template_file']
        query_template = self.jinja_env.get_template(template_file)
        context = {
            "items": query_items
        }
        query = query_template.render(context)

        response = self.llm.make_request(query)
        return response
    
    '''
    Do the actual belief update process given the probabilities.
    '''
    def update_from_probs(self, item_idx, true_prob, false_prob):
        # TODO: For now we'll do some hack that just changes the mean
        momentum = 0.3
        
        # This is super hacky, we'll fix it
        new_util = ((1-momentum) * self.util[item_idx]) + momentum * (true_prob / (true_prob + false_prob))
        self.util[item_idx] = new_util
        return
        
    
    '''
    Queries the LLM on if a user would like the item given the response history. Expects part of the query
    string to be passed in query, like in the belief_update function. Returns a dict with the log probs of true/false
    for the user liking the description of the item based on the response history in resp_hist_str
    '''
    def get_item_pref(self, item_reviews):
        pref_query_file = self.config['llm']['pref_template_file']
        query_template = self.jinja_env.get_template(pref_query_file)
        context = {
            "interactions": self.responses,
            "reviews": item_reviews
        }
        final_query = query_template.render(context)

        response = self.llm.make_request(final_query, logprobs=10)
        log_probs = self.llm.get_log_probabilities()

        # Extract T/F logprobs and return. Will need to pool positive/negative tokens
        true_prob = 0.0
        false_prob = 0.0

        #TODO: Add an option to turn off stripping and lowercasing

        for token_pos in log_probs: # Iterate over each token value for each token position
            for token_val in token_pos.keys():
                trimmed_token_val = token_val.strip().lower()
                if trimmed_token_val == "true":
                    token_log_prob = token_pos[token_val]
                    token_prob = math.exp(token_log_prob)
                    true_prob += token_prob
                if trimmed_token_val == "false":
                    token_log_prob = token_pos[token_val]
                    token_prob = math.exp(token_log_prob)
                    false_prob += token_prob
            
        return {'true': true_prob, 'false': false_prob}

    '''
    Updates the belief state for all items and user profile based on the user's response
    '''
    def belief_update(self): 
        for item_idx, item in enumerate(self.items):
            # Get logprobs
            probs = self.get_item_pref(item['reviews'])
            # Perform update
            self.update_from_probs(item_idx, probs['true'], probs['false'])
        return


    def pe_loop(self):
        for i in range(4):
            query = self.query_selection()
            print(query)
            response = input("Your response: ")
            self.responses.append({'query': query, 'response': response}) # maybe fix datatype later for scalability?
            self.belief_update() 
            if (self.debug):
                self.logger.debug("Utilities at turn", i, self.util)


        top_items = self.get_top_items(5)
        print("Top ranked items:")
        for i, item in enumerate(top_items):
            print(i, item)

        return top_items
