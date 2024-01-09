from utils.setup_logging import setup_logging
import logging

'''
The purpose of the DialogueSimulator class is to run the dialogue between the user and the
PE_module. 
'''

class DialogueSimulator:
    
    def __init__(self, config):
        self.config = config
        self.logger: logging.Logger = setup_logging(self.__class__.__name__, config)

    def run_dialogue(self, user, pe_module):
        for dialogue_turn_num in range(self.config['dialogue_sim']['num_turns']):
            query = pe_module.get_query()
            response = user.get_response(query)
            pe_module.update_from_response(query, response)
        results = pe_module.get_last_results()
        self.logger.debug(f"Query LLM Time: {pe_module.total_llm_time}  |  Entailment Time: {pe_module.total_entailment_time}  |  User LLM Time: {user.total_llm_time}" )
        return results
