from item import Item
from user import User
import math
from pe_modules.base_pe_module import BasePEModule
import jinja2

#TODO: Remove these after debugging
# import matplotlib.pyplot as plt
# from scipy.stats import beta
# import numpy as np

'''
BetaBernPEModule is a child class of PEModule, the core module for our preference elicitation task.
BetaBernPEModule uses Armin's proposed belief update.

items - A list of all Items
util - List with utility value beliefs for each item
responses - List of user query/response strings
'''

class BetaPEModule(BasePEModule):

    def __init__(self, config, dataloader):
        super().__init__(config)
        self.items = dataloader.get_data()
        self.user_profile = ""
        self.util = [{"alpha": 1, "beta": 1}] * len(dataloader) # Store the parameters of a beta-bernoulli distribution for the utility of each item
        self.responses = []
        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath='./templates'))
        self.like_probs = [1] * len(dataloader) # This is the probability of each item being in the group the user prefers based on repeated spliting into half-spaces on aspects
        self.aspects = []
        self.ucb_counts = [1] * len(dataloader) # start with one to avoid division by zero? TODO: confirm w Armin
        self.turn = 0 # Counts the number of turns that have passed

    '''
    Gets the top k items with UCB using tradeoff parameter c
    '''
    def ucb_get_items(self, k=3, c=1):
        scores = [(util['alpha'] / (util['alpha'] + util['beta']) ) for util in self.util]

        for i in range(len(scores)):
            scores[i] += math.sqrt(2 * math.log(self.turn) / self.ucb_counts[i]) # TODO: Try to clean this up 
        
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i])[-2:]

        top_k_items = []
        for idx in reversed(top_k_idx):
            top_k_items.append(self.items[idx])
        return top_k_items

    '''
    Return the items with the top k utility means. Just for ease of initial implementation
    '''
    def get_top_items(self, k=3):
        # Get indices of the top k items by means
        # import pdb; pdb.set_trace()
        top_k_idx = sorted(range(len(self.util)), key=lambda i: (self.util[i]['alpha'] / (self.util[i]['alpha'] + self.util[i]['beta']) ))[-k:]
        top_k_items = []
        for idx in reversed(top_k_idx):
            top_k_items.append(self.items[idx])
        return top_k_items

    '''
    Get the LLM to generate a query for the user based on the current utility values.
    '''
    def query_selection(self):
        # Get the top 3 items and generate the query to differentiate between them
        #query_items = self.ucb_get_items() #implement later, for now pick top 3 items by utility mean
        query_items = self.ucb_get_items(k=3, c=0.2)

        debug_str = "Selecting query based on items with id %d and id %d" % (query_items[0]['id'], query_items[1]['id']) # TODO: Could maybe plot utility distn instead?
        self.logger.debug(debug_str) # Log all utilities to debugger - works for small datasets
        
        template_file = self.config['llm']['query_selection_template_file']
        query_template = self.jinja_env.get_template(template_file)
        context = {
            "items": query_items,
            "interactions": self.responses,
            "aspects": self.aspects
        }
        query = query_template.render(context)

        response = self.llm.make_request(query)
        return response
    
    '''
    Returns a string with the attribute that the llm extracted from the most recent query
    '''
    def get_last_aspect(self):
        # An aspect based on the user's response to the last query
        
        template_file = self.config['llm']['aspect_extraction_template_file']
        query_template = self.jinja_env.get_template(template_file)
        context = {
            "interaction": self.responses[-1]
        }
        query = query_template.render(context)

        response = self.llm.make_request(query)
        return response
    
    def entailment_check(self, aspect, item):

        template_file = self.config['llm']['entailment_template_file']
        query_template = self.jinja_env.get_template(template_file)
        context = {
            "reviews": item['reviews'],
            "aspect": aspect,
            "interactions": self.responses
        }
        query = query_template.render(context)

        response = self.llm.make_request(query, logprobs=0)

        response = response.strip().lower()
        
        if response == "yes":
            return True
        elif response == "no":
            return False
        else:
            return None # Could throw an error instead. Could also switch to sum of logprobs 


    '''
    Updates the belief state for all items
    '''
    def belief_update(self): 
        # Query LLM to extract aspect from the last query -> based on the yes/no of the response this is true/false respectively
        aspect = self.get_last_aspect()
        self.aspects.append(aspect)
        
        #  Perform entailment between the aspect and each item to get a halfspace - 
        #  e.g. , for item i1, prompt an LLM with "Could <aspect> be used to describe i1? Reply Yes/No.'' - anton
        for i, item in enumerate(self.items):
            # Perform entailment on item, returns a Boolean indicating if this aspect entails this item
            entailment = self.entailment_check(aspect, item)
            
            # Debug Log
            debug_str = aspect
            if (entailment):
                debug_str += " does entail item %d" % i
            else:
                debug_str += " does NOT entail item %d" % i
            self.logger.debug(debug_str) # Log all utilities to debugger - works for small datasets

            # If user responded "no", then invert the entailment assessment. TODO: This is a little sketchy, clear with the others
            if self.responses[-1]['response'] == "no":
                entailment = not entailment

            # Armin's belief update
            # Update like_probs to indicate probability of preference for item's aspect space
            if entailment:
                self.like_probs[i] *= 0.9
            else: 
                self.like_probs[i] *= 0.1
            # import pdb; pdb.set_trace()
            new_alpha = self.util[i]['alpha'] + self.like_probs[i] # new_alpha = old_alpha + L
            new_beta = 1 - self.like_probs[i] + self.util[i]['beta'] # new_beta = 1 - L + old_beta
            self.util[i] = {'alpha': new_alpha, 'beta': new_beta}
        pass

    '''
    pe_loop is the core functionality of the preference elicitation module. It repeatedly selects a query,
    gets the user's response, then updates beliefs. Currently, it is set to run a fixed number of times before printing
    the top 3 items
    '''
    def pe_loop(self):
        # Plot utility TODO: Remove when past initial stages

        for i in range(5):
            self.turn += 1
            query = self.query_selection()
            print("QUERY:", query)
            response = input("Your response (ONLY yes or no): ")
            self.responses.append({'query': query, 'response': response}) 
            self.belief_update() 
            debug_str = "Utilities at turn %s: %s" % (i, str(self.util)) # TODO: Could maybe plot utility distn instead?
            self.logger.debug(debug_str) # Log all utilities to debugger - works for small datasets
            debug_str = "Like_probs at turn %s: %s" % (i, str(self.like_probs)) 
            self.logger.debug(debug_str) 

            # Plotting utility
            # for i, util in enumerate(self.util):
            #     x = np.linspace(beta.ppf(0.01, util['alpha'], util['beta']), beta.ppf(0.99, util['alpha'], util['beta']), 100)
            #     plt.plot(x, beta.pdf(x, util['alpha'], util['beta']), label='%d' % i)
            # plt.show()

        # import pdb; pdb.set_trace()

        top_items = self.get_top_items(3)
        print("Top ranked items:")
        for i, item in enumerate(top_items):
            print("Rank %d: Item #%d" % (i,item['id']))
            for review in item['reviews']:
                print("   Review:", review)

        return top_items
