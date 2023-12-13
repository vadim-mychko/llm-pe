

'''
The purpose of the DialogueSimulator class is to run the dialogue between the user and the
PE_module. 
'''

class DialogueSimulator:
    
    def __init__(self, config):
        self.config = config

    def run_dialogue(self, user, pe_module):
        for dialogue_turn_num in range(self.config['dialogue_sim']['num_turns']):
            query = pe_module.get_query()
            response = user.get_response(query)
            pe_module.update_from_response(query, response)
        results = pe_module.get_last_results()
        return results
