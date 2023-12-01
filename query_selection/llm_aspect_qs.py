from query_selection.llm_base_qs import LLMQuerySelection

# This QS Module will get an aspect first, then get a query from that
class LLMAspectQuerySelection(LLMQuerySelection):

    def __init__(self):
        super().__init__()
        self.history['aspects'] = [] # Add an aspects list in self.history
    
    '''
    We expect query info to be a dict with the following structure:
    "items": list of lists of item descriptions
    '''
    def get_query(self, query_info):
        # Get aspect from LLM
        template_file = self.config['llm']['aspect_from_items_template_file']
        query_template = self.jinja_env.get_template(template_file)
        context = {
            "items": query_info['items'],
            "interactions": self.history['queries'],
            "aspects": self.history['aspects']
        }
        aspect_query = query_template.render(context)

        query_aspect = self.llm.make_request(query)


        # Generate query from aspect via LLM
        template_file = self.config['llm']['query_from_aspect_template_file']
        query_template = self.jinja_env.get_template(template_file)
        context = {
            "aspect": query_aspect
        }
        query = query_template.render(context)

        response = self.llm.make_request(query)
        return response