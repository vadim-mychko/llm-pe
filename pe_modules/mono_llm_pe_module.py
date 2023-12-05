from pe_modules.base_pe_module import BasePEModule


'''
The MonoLLMPEModule is a subclass of BasePEModule which conducts preference elicitation using
a monolithic LLM architecture. There is no maintained belief state, the LLM does all inference
based on internal knowledge and item descriptions.
'''
class MonoLLMPEModule(BasePEModule):

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)

    '''
    Get the top k recommended items
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

        response = self.llm.make_request(query)
        return response

    '''
    Generates a query based on the current utility values and the provided set of items.
    '''
    def get_query(self):
        template_file = self.config['mono_llm']['query_gen_template']
        query_template = self.jinja_env.get_template(template_file)
        # import pdb; pdb.set_trace()
        context = {
            "items": self.items,
            "interactions": self.interactions,
        }
        query = query_template.render(context)

        response = self.llm.make_request(query)
        return response
    
    '''
    Update the model's beliefs, etc based on the user's response
    '''
    def update_from_response(self, query, response):
        self.interactions.append({"query": query, "response": response})
    