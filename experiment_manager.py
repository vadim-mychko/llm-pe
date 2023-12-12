import os
from users.base_user import UserSim
from users.llm_user import LLMUserSim
from dialogue_sim import DialogueSimulator
import pe_modules
import users
import dataloaders
import llms
import jinja2
import argparse
import yaml


from utils.logging import setup_logging


class ExperimentManager():

    def __init__(self):

        self.logger = None

    '''
    Dump to JSON
    '''
    def dump_results(self):
        raise NotImplementedError

    # This class should eventually use the UserSim class to allow manual experiments. Not implementing now bc low priority
    def manual_experiment(self, config):
        raise NotImplementedError

    '''
    Run the experiment specified by the config file
    '''

    def run_single_experiment(self, config):
        self.logger = setup_logging(self.__class__.__name__, config)

        # Dataloader Class
        dataloader_class = dataloaders.DATALOADER_CLASSES[config['data']['data_loader_name']]
        # Load item data
        item_dataloader = dataloader_class(config['data']['data_path'], config) 
        items = item_dataloader.get_data()
        # Load user data
        user_dataloader = dataloader_class(config['data']['user_path'], config) 
        user_data = user_dataloader.get_data()
        # Set up other stuff
        llm_module = llms.LLM_CLASSES[config['llm']['llm_name']]
        llm = llm_module(config)

        # Dialogue Sim
        dial_sim = DialogueSimulator(config)

        # PE Module
        pe_module_class = pe_modules.PE_MODULE_CLASSES[config['pe']['pe_module_name']]
        pe_module = pe_module_class(config, item_dataloader)

        runs = []
        for user_id, user_item_ids in user_data.items():
            # Add relevant item descriptions to a list
            item_descs = []
            for item_id in user_item_ids:
                item_descs.append(items[item_id]['description'])
            # Create user simulator and dialogue simulator
            user_sim = LLMUserSim(config, item_descs, llm)

            # Reset pe_module
            pe_module.reset()
                
            # Run Dialogue
            recs = dial_sim.run_dialogue(user_sim, pe_module)
            
            # TODO: Temp fix for difference in output format
            for rec_turn in range(len(recs)):
                if (config['pe']['pe_module_name'] == "DT"):
                    runs.append({'user_id': user_id,'turn': rec_turn, 'correct_items': user_item_ids, 'rec_items': recs[rec_turn][0]})
                    # self.logger.info("Recommendations at turn %d: %s" % (turn, recs[turn][0]['id']))
                else:
                    # self.logger.info("Recommendations at turn %d: %s" % (turn, recs[turn]))
                    runs.append({'user_id': user_id, 'turn': rec_turn, 'correct_items': user_item_ids, 'rec_items': recs[rec_turn]})

        #TODO: Change output format and dump to JSON
        print(runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-config", "--config_path", type=str, default="./configs/david_rest_config.yaml")
    parser.add_argument("-config", "--config_path", type=str, default="./configs/david_base_config.yaml")

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path))

    experiment_manager = ExperimentManager()
    # experiment_manager.run_multi_experiment()
    experiment_manager.run_single_experiment(config)