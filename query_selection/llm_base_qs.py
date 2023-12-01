import abc

'''
This is the ABC for Query selection using an LLM.
Args:
    config - the config file's contents
    llm - the LLMBase that we will use to get the query
    jinja_env - the Jinja Env to use for templates
'''

class LLMQuerySelection(abc.ABC):

    def __init__(self, config, llm, jinja_env):
        self.llm = llm
        self.config = config
        self.jinja_env = jinja_env
        self.history = {"queries": []} # Store past interactions

    '''
    get_query prompts the LLM to get the query to ask the user to determine their preferences over query_items
    Args: 
        query_items - the item descriptions to inspire the query 
    Returns:
        response - the query to ask the user to best separate query_items
    
    '''
    def get_query(self, query_info):
        raise NotImplementedError
    
    # Get the past history of queries, aspects, etc in a dict form
    def get_history(self):
        return self.history