from query_selection.llm_base_qs import LLMQuerySelection


# This class will generate the query straight from the items
class LLMItemQuerySelection(LLMQuerySelection):
    
    '''
    We expect query info to be a dict with the following structure:
    "items": list of lists of item descriptions
    '''
    def get_query(self, query_info):
        template_file = self.config['llm']['query_selection_template_file']
        query_template = self.jinja_env.get_template(template_file)
        context = {
            "items": query_info['items'],
            "interactions": self.responses,
        }
        query = query_template.render(context)

        response = self.llm.make_request(query)
        return response