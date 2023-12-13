from pe_modules.base_pe_module import BasePEModule
import math
import heapq
import math
import item_scorers
import history_preprocessors


'''
The DTPEModule is a subclass of BasePEModule which conducts preference elicitation using
a combined decision theory and LLM approach.
'''
class DTPEModule(BasePEModule):

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)

        item_scorer_class = item_scorers.ITEM_SCORER_CLASSES[config['item_scoring']['item_scorer_name']]
        self.item_scorer = item_scorer_class(config)

        if config['item_scoring']['preprocess_query']:
            history_preprocessor_class = history_preprocessors.HISTORY_PREPROCESSOR_CLASSES[config['item_scoring']['history_preprocessor_name']]
            self.history_preprocessor = history_preprocessor_class(config)


        self.belief = {}
        for id in self.items:
            self.belief[id] = {"alpha": 0.5, "beta": 0.5} # Set initial belief state
        # TODO: Set up item selection method from config

        # These fields are only used to maintain information for the return_dict
        self.all_beliefs = [] # List containing full belief state at the end of each turn (i.e. after belief update)
        self.queried_items = [] # List of items queried at each turn
        self.recs = [] # List of recommended items at each step. Number recommended is from config
        self.aspects = []

    '''
    Generate an aspect from the item description on which to query 
    '''
    def get_aspect(self, item_desc):
        template_file = self.config['query']['aspect_gen_template']
        prompt_template = self.jinja_env.get_template(template_file)

        context = {
            "item_desc": item_desc, # Item description for the given item
            "aspects": self.aspects
        }
        prompt = prompt_template.render(context)

        aspect_pair = self.llm.make_request(prompt, temperature=self.config['llm']['temperature'])
        print(aspect_pair)

        return aspect_pair

    '''
    Generates a query based on the current utility values and the provided set of items.
    '''
    def get_query(self):
        ITEM_SELECTION_MAP = {
        'greedy': self.item_selection_greedy,
        }  
        
        # Run item selection to get the item to generate from
        item_selection_method = ITEM_SELECTION_MAP[self.config['query']['item_selection']]
        item_ids = item_selection_method() #NOTE: item_id is a single-item list with the item_id of the top idx
        self.queried_items.append(item_ids)
        item_desc = self.items[item_ids[0]]['description'] 
        
        # Get the aspect
        aspect_pair = self.get_aspect(item_desc)

        # Clean the aspect, value pair (assuming correct format)
        aspect_list = aspect_pair.split(",")
        for i in range(len(aspect_list)):
            aspect_list[i] = aspect_list[i].strip()
        
        aspect_dict = {"aspect": aspect_list[0], "value": aspect_list[1]}
        self.aspects.append(aspect_dict)

        # Generate query from aspect and item_desc
        template_file = self.config['query']['query_gen_template']
        prompt_template = self.jinja_env.get_template(template_file)

        context = {
            "item_desc": item_desc, # Item description for the given item
            "aspect": aspect_dict
        }
        prompt = prompt_template.render(context)

        # self.logger.debug(prompt)

        user_query = self.llm.make_request(prompt, temperature=self.config['llm']['temperature'])
        self.logger.debug(user_query)
        return user_query
    
    '''
    Get the IDs of the top k recommended items
    '''
    def get_top_items(self, k=5):
        top_k_ids = heapq.nlargest(k, self.items, key=lambda i: (self.belief[i]['alpha'] / (self.belief[i]['alpha'] + self.belief[i]['beta']) ))
        return top_k_ids
    
    
    '''
    Update the model's beliefs, etc based on the user's response
    '''
    def update_from_response(self, query, response):
        self.interactions.append({"query": query, 
                                  "response":response, 
                                  "aspect": self.aspects[-1]['aspect'], 
                                  "value": self.aspects[-1]['value']
                                })

        # Use either full history or just last response 
        preference = [self.interactions[-1]] if self.config['pe']['response_update']=="individual" else self.interactions

        #optional preprocessing
        if self.config['item_scoring']['preprocess_query']:
            preference = self.history_preprocessor.preprocess(preference)
            self.logger.debug("Preference: %s" % preference)

        #ANTON Dec 12 TODO: set truncation warnings
        #get like_prob for all items
        like_probs = self.item_scorer.score_items(preference, self.items) 

        for item_id in self.items:
            new_alpha = self.belief[item_id]['alpha'] + like_probs[item_id] # new_alpha = old_alpha + L
            new_beta = 1 - like_probs[item_id] + self.belief[item_id]['beta'] # new_beta = 1 - L + old_beta
            self.belief[item_id] = {'alpha': new_alpha, 'beta': new_beta}

            self.logger.debug("Like probs for item %s: %f, updated alpha = %f and beta = %f" % (item_id, like_probs[item_id], self.belief[item_id]['alpha'], self.belief[item_id]['beta']))

        self.all_beliefs.append(self.belief)

        # Append the top k items to self.recs for return_dict
        k = self.config['pe']['num_recs']
        top_recs = self.get_top_items(k)
        self.recs.append(top_recs)


    def reset(self):
        super().reset()
        self.belief = {}
        for id in self.items:
            self.belief[id] = {"alpha": 0.5, "beta": 0.5}
        self.queried_items = []
        self.all_beliefs = []


    '''
    the following item_selection_x() methods use different pointwise item selection methods. Each 
    method returns the item_id on which to query, based on the selection method.
    '''
    # Select the item with the highest expected utility.
    def item_selection_greedy(self):
        top_id = heapq.nlargest(1, self.items, key=lambda i: (self.belief[i]['alpha'] / (self.belief[i]['alpha'] + self.belief[i]['beta'])))
        return top_id

    def item_selection_random(self):
        raise NotImplementedError

    def item_selection_entropy_reduction(self):
        raise NotImplementedError

    def thompson_sampling(beliefs):
        raise NotImplementedError
    
    def get_last_results(self):
        results = {"rec_items": self.recs, 
                   "conv_hist": self.interactions, 
                   "queried_items": self.queried_items, 
                   "belief_states": self.all_beliefs,
                   "aspects": self.aspects}
        return results