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

    def run_experiment(self):
        dial_sim = DialogueSimulator(self.config)
        
        # SELECT USER SIM AND PE MODULE TYPES
        user = UserSim()
        if (self.config['dialogue_sim']['sim_type'] == "llm"):
            user = LLMUserSim(self.config, self.data[3]['description'], self.llm, self.jinja)
            
        pe_module_class = pe_modules.PE_MODULE_CLASSES[self.config['pe']['pe_module_name']]
        pe_module = pe_module_class(self.config, self.dataloader)

        # Run Dialogue
        recs = dial_sim.run_dialogue(user, pe_module)
        for turn in range(len(recs)):
            print("Recommendations at turn %d: " % turn, recs[turn])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-config", "--config_path", type=str, default="./configs/david_rest_config.yaml")
    parser.add_argument("-config", "--config_path", type=str, default="./configs/david_base_config.yaml")

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path))

    experiment_manager = ExperimentManager(config)
    experiment_manager.run_experiment()