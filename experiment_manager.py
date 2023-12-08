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

    def __init__(self, config):

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

        # Dataloader
        dataloader = dataloaders.DATALOADER_CLASSES[self.config['data']['data_loader_name']]
        self.dataloader = dataloader(config) # Could force to be DataLoader class here?
        self.data = self.dataloader.get_data()
        self.jinja = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath='./templates'))
        llm_module = llms.LLM_CLASSES[self.config['llm']['llm_name']]
        self.llm = llm_module(config)

    def run_single_experiment(self):
        dial_sim = DialogueSimulator(self.config)
        
        # SELECT USER SIM AND PE MODULE TYPES
        user = UserSim()
        if (self.config['dialogue_sim']['sim_type'] == "llm"):
            item_num = 0 # NOTE: SET THE ITEM NUMBER HERE
            user = LLMUserSim(self.config, self.data[item_num]['description'], self.llm, self.jinja)
            self.logger.debug("First Description of Top Item: %s" % self.data[item_num]['description'][0])
            
        pe_module_class = pe_modules.PE_MODULE_CLASSES[self.config['pe']['pe_module_name']]
        pe_module = pe_module_class(self.config, self.dataloader)

        # Run Dialogue
        recs = dial_sim.run_dialogue(user, pe_module)
        # TODO: Temp fix for difference in output format
        for turn in range(len(recs)):
            if (self.config['pe']['pe_module_name'] == "DT"):
                self.logger.info("Recommendations at turn %d: %s" % (turn, recs[turn][0]['id']))
            else:
                self.logger.info("Recommendations at turn %d: %s" % (turn, recs[turn]))

    def run_multi_experiment(self):
        dial_sim = DialogueSimulator(self.config)
        
        # SELECT USER SIM AND PE MODULE TYPES
        item_num = 0
        user = LLMUserSim(self.config, self.data[item_num]['description'], self.llm, self.jinja)

        runs = []

        pe_module_class = pe_modules.PE_MODULE_CLASSES[self.config['pe']['pe_module_name']]
        pe_module = pe_module_class(self.config, self.dataloader)
#
        for run_num in range(len(self.data)):
        # for run_num in range(2):
            pe_module.reset()
            # import pdb; pdb.set_trace()
            user.set_top_item(self.data[run_num]['description'])
            self.logger.debug("First Description of Top Item: %s" % self.data[run_num]['description'][0])
                
            # Run Dialogue
            recs = dial_sim.run_dialogue(user, pe_module)
            # TODO: Temp fix for difference in output format
            for rec_turn in range(len(recs)):
                if (self.config['pe']['pe_module_name'] == "DT"):
                    runs.append({'run': run_num,'turn': rec_turn, 'correct_item': run_num, 'rec_item': recs[rec_turn][0]['id']})
                    # self.logger.info("Recommendations at turn %d: %s" % (turn, recs[turn][0]['id']))
                else:
                    # self.logger.info("Recommendations at turn %d: %s" % (turn, recs[turn]))
                    runs.append({'run': run_num, 'turn': rec_turn, 'correct_item': self.data[run_num]['id'], 'rec_item': recs[rec_turn]})

        self.logger.info("RESULTS: ######################################")
        for run in runs:
            if (self.config['pe']['pe_module_name'] == "DT"):
                self.logger.info("Run %d - Turn %d - Correct Item %d - Recommended Item %d" % (run['run'], run['turn'], run['correct_item'], run['rec_item']))
            else:
                self.logger.info("Run %d - Turn %d - Correct Item %d - Recommended Item %s" % (run['run'], run['turn'], run['correct_item'], run['rec_item']))

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-config", "--config_path", type=str, default="./configs/david_rest_config.yaml")
    parser.add_argument("-config", "--config_path", type=str, default="./configs/david_base_config.yaml")

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path))

    experiment_manager = ExperimentManager(config)
    # experiment_manager.run_multi_experiment()
    experiment_manager.run_single_experiment()