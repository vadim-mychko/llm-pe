import os
from users.base_user import UserSim
from users.llm_user import LLMUserSim
from dialogue_sim import DialogueSimulator
import pe_modules
import users
import dataloaders
import llms
import argparse
import yaml
import json


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
    
    def run_experiments(self,exp_dir: str):
        """
        This function takes a directory path and runs experiments based on configuration files in the subdirectories.
        It expects each subdirectory to contain a 'config.yaml' file, which will be used to load experiment settings.
        
        The function will iterate over the subdirectories of the given directory in the order they appear in the directory.
        For each subdirectory, it will open and read the 'config.yaml' file as a configuration for an experiment.
        The 'config.yaml' file should be in a format that can be parsed into a Python dictionary.

        Parameters:
        exp_dir (str): A string specifying the path of the directory containing experiment subdirectories.

        Returns:
        None
        """

        # os.walk() will yield a tuple containing directory path, 
        # directory names and file names in the directory.
        for root, dirs, files in os.walk(exp_dir):
            # we are interested in directories only
            for directory in dirs:
                config_file_path = os.path.join(root, directory, "config.yaml")
                dir_path = os.path.join(root, directory)
                self.run_single_experiment(config_file_path, dir_path = dir_path)

    '''
    Run the experiment specified by the config file
    '''
    def run_single_experiment(self, config_path, dir_path):
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config = yaml.safe_load(config_file)
        
        self.logger = setup_logging(self.__class__.__name__, config)

        # Dataloader Class
        item_dataloader_class = dataloaders.DATALOADER_CLASSES[config['data']['data_loader_name']]
        user_dataloader_class = dataloaders.DATALOADER_CLASSES[config['data']['user_loader_name']]
        # Load item data
        item_dataloader = item_dataloader_class(config['data']['data_path'], config) 
        items = item_dataloader.get_data()
        # Load user data
        user_dataloader = user_dataloader_class(config['data']['user_path'], config) 
        user_data = user_dataloader.get_data()
        # Set up other stuff
        llm_module = llms.LLM_CLASSES[config['llm']['llm_name']]
        llm = llm_module(config)

        # Dialogue Sim
        dial_sim = DialogueSimulator(config)

        # PE Module
        pe_module_class = pe_modules.PE_MODULE_CLASSES[config['pe']['pe_module_name']]
        pe_module = pe_module_class(config, item_dataloader)

        runs = {}
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
            results = dial_sim.run_dialogue(user_sim, pe_module)
            runs[user_id] = results
        
        output_file = open(os.path.join(dir_path, "results.json"), "w")
        json.dump(runs, output_file)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    experiment_manager = ExperimentManager()

    parser = argparse.ArgumentParser()

    parser.add_argument("-exp_dir", "--experiment_dir", type=str)

    args = parser.parse_args()

    # experiment_manager.run_experiments(args.experiment_dir)
    experiment_manager.run_experiments("/Users/david/Documents/Research2324/Sanner/llm-pe/experiments/example/generation_test")