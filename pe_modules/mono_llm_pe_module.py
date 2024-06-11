from pe_modules.base_pe_module import BasePEModule
import timeit

'''
The MonoLLMPEModule is a subclass of BasePEModule which conducts preference elicitation using
a monolithic LLM architecture. There is no maintained belief state, the LLM does all inference
based on internal knowledge and item descriptions.
'''
class MonoLLMPEModule(BasePEModule):

    def __init__(self, config, dataloader, llm):
        super().__init__(config, dataloader, llm)
        self.total_llm_time = 0.0
        self.total_entailment_time = 0.0

    '''
    Get the IDs of the top k recommended items
    '''
    def get_top_items(self, k=5):
        template_file = self.config['mono_llm']['top_items_template']
        query_template = self.jinja_env.get_template(template_file)
        context = {
            "num_items": k,
            "items": self.items, 
            "interactions": self.interactions,
        }
        query = query_template.render(context)

        self.logger.debug(query)

        start = timeit.default_timer()
        response = self.llm.make_request(query)
        self.logger.debug(response)
        stop = timeit.default_timer()
        self.total_entailment_time += (stop - start)
        return response # Just use the full string, we'll parse it in eval_manager
    
    # Have LLM record the strings, then have the evaluator process the results

    '''
    Generates a query based on the current utility values and the provided set of items.
    '''
    def get_query(self):
        template_file = self.config['mono_llm']['query_gen_template']
        query_template = self.jinja_env.get_template(template_file)
        context = {
            "items": self.items,
            "interactions": self.interactions,
        }
        query = query_template.render(context)

        #self.logger.debug(query)
        
        start = timeit.default_timer()
        response = self.llm.make_request(query)
        stop = timeit.default_timer()
        self.total_llm_time += (stop - start)
        return response
    
    '''
    Update the model's beliefs, etc based on the user's response
    '''
    def update_from_response(self, query, response):
        self.interactions.append({"query": query, "response": response})
        # Get top recommended items
        top_items = self.get_top_items(self.config['pe']['num_recs'])
        self.recs.append(top_items)

    def get_last_results(self):
        return {'conv_hist': self.interactions, 'rec_items': self.recs}