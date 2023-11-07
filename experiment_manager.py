import os
from pe_modules.joint_pe_module import JointPEModule


from utils.logging import setup_logging


class ExperimentManager():

    def __init__(self, config):

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

        # TODO: Dataloader

    def run_experiment(self):

       # TODO: Add support for polymorphism later
       pe_module = JointPEModule(self.config, self.dataloader)

       for x in range(5):
           query = pe_module.get_query()
           # ASK LLM -> maybe this should be either a separate class or a method for pe_module
           response = "Get from LLM"
           pe_module.belief_update(query, response)
