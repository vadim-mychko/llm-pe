from pe_modules.base_pe_module import BasePEModule
import math
import heapq
import math


'''
The DTPEModule is a subclass of BasePEModule which conducts preference elicitation using
a combined decision theory and LLM approach.
'''
class DTPEModule(BasePEModule):

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.belief = {}
        for id in self.items:
            self.belief[id] = {"alpha": 0.5, "beta": 0.5} # Set initial belief state
        # TODO: Set up item selection method from config

    '''
    Generates a query based on the current utility values and the provided set of items.
    '''
    def get_query(self):
        ITEM_SELECTION_MAP = {
        'greedy': self.item_selection_greedy,
        }  
        
        item_selection_method = ITEM_SELECTION_MAP[self.config['query']['item_selection']]

        item_ids = item_selection_method() #NOTE: item_id is a list with the item_id of the top idx
        # import pdb; pdb.set_trace()
        item_desc = self.items[item_ids[0]]['description'] 
        # self.logger.debug("Generating query from item %d with first desc %s" % (item_idx[0], item_desc))

        template_file = self.config['query']['query_gen_template']
        prompt_template = self.jinja_env.get_template(template_file)

        context = {
            "item_desc": item_desc, # Item description for the given item
            "interactions": self.interactions 
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
    
    # Check if the user will like an item based on the item description and the full/partial interaction history
    # Return the probbility that the user will like the item as a float.
    def get_like_probs(self, item): 

        template_file = self.config['llm']['like_probs_template']
        query_template = self.jinja_env.get_template(template_file)

        # Use either full history or just last response
        interaction_history = [self.interactions[-1]] if self.config['pe']['response_update']=="individual" else self.interactions

        context = {
            "item_desc": item['description'],
            "interactions": interaction_history
        }
        query = query_template.render(context)

        # self.logger.debug(query)

        # response = self.llm.make_request(query, logprobs=0)
        response = self.llm.make_request(query, logprobs=1)

        
        response = response.strip().lower()

        full_logprobs = self.llm.get_full_logprobs() 

        # NOTE: We're assuming here that response is one word, which it should be. Take logprobs of the first word

        log_probs = full_logprobs['token_logprobs'][0]

        probs = math.exp(log_probs)

        like_probs = 0.0

        if (response == "true"):
            like_probs = (1 + probs) / 2
        elif (response == "false"):
            like_probs = (1 - probs) / 2
        else:
            raise ValueError("Expected LLM response to be either true or false")
        
        # self.logger.debug("Like Probs Query: %s Response: %s Like Probs: %f" % (query, response, like_probs))
        
        return like_probs

        # # Extract T/F logprobs 
        # true_prob = 0.0
        # false_prob = 0.0

        # for token_pos in log_probs: # Iterate over each token value for each token position
        #     for token_val in token_pos.keys():
        #         trimmed_token_val = token_val.strip().lower()
        #          # If token is 'true' or 'false' after lowercasing and stripping, add to corresponding probability total
        #         if trimmed_token_val == "true":
        #             token_log_prob = token_pos[token_val]
        #             token_prob = math.exp(token_log_prob)
        #             true_prob += token_prob
        #         if trimmed_token_val == "false":
        #             token_log_prob = token_pos[token_val]
        #             token_prob = math.exp(token_log_prob)
        #             false_prob += token_prob 
    
    '''
    Update the model's beliefs, etc based on the user's response
    '''
    def update_from_response(self, query, response):
        self.interactions.append({"query": query, "response":response})

        # Update this for new class
        for item_id in self.items:
            like_probs = self.get_like_probs(self.items[item_id])

            new_alpha = self.belief[item_id]['alpha'] + like_probs # new_alpha = old_alpha + L
            new_beta = 1 - like_probs + self.belief[item_id]['beta'] # new_beta = 1 - L + old_beta
            self.belief[item_id] = {'alpha': new_alpha, 'beta': new_beta}

            self.logger.debug("Like probs for item %s: %f, updated alpha = %f and beta = %f" % (item_id, like_probs, self.belief[item_id]['alpha'], self.belief[item_id]['beta']))

    def reset(self):
        super().reset()
        self.belief = {}
        for id in self.items:
            self.belief[id] = {"alpha": 0.5, "beta": 0.5}


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