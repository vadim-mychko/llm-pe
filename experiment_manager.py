import os
from pe_modules.joint_pe_module import JointPEModule
from pe_modules.beta_pe_module import BetaPEModule
import dataloaders
import llms
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

    def run_experiment(self):

        # TODO: Add support for polymorphism later
        #pe_module = JointPEModule(self.config, self.dataloader)
        pe_module = BetaPEModule(self.config, self.dataloader)

        result = pe_module.pe_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-config", "--config_path", type=str, default="./configs/david_rest_config.yaml")
    parser.add_argument("-config", "--config_path", type=str, default="./configs/david_base_config.yaml")

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path))

    experiment_manager = ExperimentManager(config)
    experiment_manager.run_experiment()