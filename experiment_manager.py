import os
from pe_modules.joint_pe_module import JointPEModule
import dataloaders
import llms


from utils.logging import setup_logging


class ExperimentManager():

    def __init__(self, config):

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

        # Dataloader
        dataloader = dataloaders.DATALOADER_CLASSES[self.config['data']['data_loader_name']]
        self.data_loader = dataloader(config) # Could force to be DataLoader class here?

    def run_experiment(self):

        # TODO: Add support for polymorphism later
        pe_module = JointPEModule(self.config, self.dataloader)

        result = pe_module.pe_loop()