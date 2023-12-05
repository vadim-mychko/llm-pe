

'''
The purpose of the DialogueSimulator class is to run the dialogue between the user and the
PE_module. 
'''

class DialogueSimulator:
    
    def __init__(self, config):
        self.config = config

    def run_dialogue(self, user, pe_module):
        recs = []
        for i in range(self.config['dialogue_sim']['num_turns']):
            query = pe_module.get_query()
            response = user.get_response(query)
            pe_module.update_from_response(query, response)
            rec = pe_module.get_top_items(k=self.config['dialogue_sim']['num_recs'])
            recs.append(rec) # TODO: This is slow, change for larger datasets
        return recs
